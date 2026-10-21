"""int8 models made by Sequential.quantize(), and C header export."""

import math
import re
import struct
import shutil
from pathlib import Path
from typing import Optional, Tuple, Union

from pyann._bindings.ffi import ffi, lib
from pyann.core.network import as_rows
from pyann.core.tensor import Tensor
from pyann.utils.compat import ArrayLike, has_numpy, to_array
from pyann.exceptions import InvalidParameterError, NetworkError, raise_for_error_code


def runtime_header_path() -> Path:
    """Path of ``ann_rt.h``, the runtime that exported C headers include.

    Wheels ship it in ``pyann/include``; development builds use the copy in
    the ``ann/`` submodule.
    """
    package_dir = Path(__file__).resolve().parent.parent
    for candidate in (package_dir / "include" / "ann_rt.h",
                      package_dir.parent.parent / "ann" / "ann_rt.h"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("ann_rt.h not found; reinstall pylibann or check out the ann/ submodule")


def _prefix_for(filepath: Path, prefix: Optional[str]) -> str:
    """The C identifier naming the header's symbols: prefix, or one made from the file name."""
    if prefix is None:
        prefix = re.sub(r"\W", "_", filepath.stem)
        if not prefix or prefix[0].isdigit():
            prefix = "model_" + prefix
    if not re.fullmatch(r"[A-Za-z_]\w*", prefix, flags=re.ASCII) or len(prefix) >= 64:
        raise ValueError(f"prefix must be a C identifier shorter than 64 characters, got {prefix!r}")
    return prefix


def export_c_header(net_ptr, q_ptr, filepath: Union[str, Path], prefix: Optional[str],
                    runtime: bool) -> Path:
    """Write a model header (int8 when q_ptr is given) and, if asked, ann_rt.h next to it."""
    filepath = Path(filepath)
    result = lib.ann_export_c_header(net_ptr, q_ptr, str(filepath).encode(),
                                     _prefix_for(filepath, prefix).encode())
    raise_for_error_code(result, f"Failed to export C header to {filepath}")
    if runtime:
        target = filepath.resolve().parent / "ann_rt.h"
        source = runtime_header_path()
        if target != source:
            shutil.copyfile(source, target)
    return filepath


class QuantizedModel:
    """An int8 version of a trained Sequential model.

    Made by :meth:`Sequential.quantize`. Weights are int8 with one scale per
    output channel, activations are int8 with one scale and zero point per
    tensor, and sums are int32, so the model runs on integer-only hardware.
    It does not depend on the model it came from.

    Example::

        q = model.quantize(x_train[:1000])
        print(model.evaluate(x_test, y_test), q.evaluate(x_test, y_test))
        q.export_c_header("model.h")      # writes model.h and ann_rt.h
    """

    def __init__(self, ptr, input_shape: Optional[Tuple[int, ...]], input_size: int, output_size: int):
        self._ptr = ptr
        self._input_shape = input_shape
        self.input_size = input_size
        self.output_size = output_size

        in_scale, in_zp = ffi.new("real *"), ffi.new("int *")
        out_scale, out_zp = ffi.new("real *"), ffi.new("int *")
        lib.ann_quantized_get_io_params(ptr, in_scale, in_zp, out_scale, out_zp)
        self.input_scale = float(in_scale[0])
        self.input_zero_point = int(in_zp[0])
        self.output_scale = float(out_scale[0])
        self.output_zero_point = int(out_zp[0])

    def _rows(self, x: ArrayLike, name: str, columns: int):
        """x as a list or 2-D array of rows, checked against the column count."""
        arr = as_rows(x, self._input_shape, name)
        if has_numpy():
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2 or arr.shape[1] != columns:
                raise ValueError(f"{name} must have {columns} values per sample, got shape {arr.shape}")
            return arr
        if arr and not isinstance(arr[0], (list, tuple)):
            arr = [arr]
        for row in arr:
            if len(row) != columns:
                raise ValueError(f"{name} has a row of {len(row)} values but the model expects {columns}")
        return arr

    def predict(self, x: ArrayLike) -> ArrayLike:
        """Run the int8 model on real inputs.

        The inputs are quantized, the model runs in integer math, and the
        outputs are dequantized (with softmax applied if the model has it).
        The results match the exported header's ``<prefix>_predict_float``.

        Returns:
            Predictions, (samples x outputs)
        """
        rows = self._rows(x, "x", self.input_size)
        if has_numpy():
            import numpy as np
            x_f32 = np.ascontiguousarray(rows, dtype=np.float32)
            out = np.zeros((len(x_f32), self.output_size), dtype=np.float32)
            src = ffi.cast("real *", x_f32.ctypes.data)
            dst = ffi.cast("real *", out.ctypes.data)
            for r in range(len(x_f32)):
                result = lib.ann_quantized_predict(self._ptr, src + r * self.input_size,
                                                   dst + r * self.output_size)
                raise_for_error_code(result, "Quantized prediction failed")
            return out

        inputs = ffi.new("real[]", self.input_size)
        outputs = ffi.new("real[]", self.output_size)
        results = []
        for row in rows:
            for i, v in enumerate(row):
                inputs[i] = float(v)
            raise_for_error_code(lib.ann_quantized_predict(self._ptr, inputs, outputs),
                                 "Quantized prediction failed")
            results.append([outputs[k] for k in range(self.output_size)])
        return results

    def quantize_input(self, x: ArrayLike) -> ArrayLike:
        """Quantize real inputs to the model's int8 input: round(x / scale) + zero_point.

        Rounds exactly like the C runtime (float32 math, halves away from
        zero), so ``predict_int8(quantize_input(x))`` is what ``predict(x)``
        computes internally.
        """
        rows = self._rows(x, "x", self.input_size)
        if has_numpy():
            import numpy as np
            v = np.asarray(rows, dtype=np.float32) / np.float32(self.input_scale) + np.float32(self.input_zero_point)
            q = np.where(v >= 0, np.floor(v + np.float32(0.5)), -np.floor(np.float32(0.5) - v))
            return np.clip(q, -128, 127).astype(np.int8)

        def f32(value: float) -> float:
            return struct.unpack("f", struct.pack("f", value))[0]

        def quantize(value: float) -> int:
            v = f32(f32(f32(value) / self.input_scale) + self.input_zero_point)
            q = math.floor(f32(v + 0.5)) if v >= 0 else -math.floor(f32(0.5 - v))
            return max(-128, min(127, q))

        return [[quantize(float(v)) for v in row] for row in rows]

    def dequantize_output(self, y: ArrayLike) -> ArrayLike:
        """Convert int8 outputs to real values: scale * (y - zero_point) (no softmax)."""
        if has_numpy():
            import numpy as np
            return (self.output_scale * (np.asarray(y, dtype=np.float32) - self.output_zero_point)).astype(np.float32)
        return [[self.output_scale * (int(v) - self.output_zero_point) for v in row] for row in y]

    def predict_int8(self, x_q: ArrayLike) -> ArrayLike:
        """Run the int8 model on int8 inputs, giving int8 outputs.

        Softmax outputs are the int8 logits (their argmax is the class). The
        results match the exported header's ``<prefix>_predict`` bit for bit.

        Args:
            x_q: int8 inputs, e.g. from :meth:`quantize_input`
        """
        rows = self._rows(x_q, "x_q", self.input_size)
        if has_numpy():
            import numpy as np
            arr = np.asarray(rows)
            if arr.size and (arr.min() < -128 or arr.max() > 127):
                raise ValueError("x_q values must be in the int8 range [-128, 127]")
            x_i8 = np.ascontiguousarray(arr, dtype=np.int8)
            out = np.zeros((len(x_i8), self.output_size), dtype=np.int8)
            src = ffi.cast("signed char *", x_i8.ctypes.data)
            dst = ffi.cast("signed char *", out.ctypes.data)
            for r in range(len(x_i8)):
                result = lib.ann_quantized_predict_int8(self._ptr, src + r * self.input_size,
                                                        dst + r * self.output_size)
                raise_for_error_code(result, "Quantized prediction failed")
            return out

        inputs = ffi.new("signed char[]", self.input_size)
        outputs = ffi.new("signed char[]", self.output_size)
        results = []
        for row in rows:
            for i, v in enumerate(row):
                if not -128 <= int(v) <= 127:
                    raise ValueError("x_q values must be in the int8 range [-128, 127]")
                inputs[i] = int(v)
            raise_for_error_code(lib.ann_quantized_predict_int8(self._ptr, inputs, outputs),
                                 "Quantized prediction failed")
            results.append([outputs[k] for k in range(self.output_size)])
        return results

    def evaluate(self, x: ArrayLike, y: ArrayLike) -> float:
        """Classification accuracy of the int8 model (compare with Sequential.evaluate)."""
        x_tensor = Tensor(self._rows(x, "x", self.input_size))
        y_tensor = Tensor(to_array(y))
        accuracy = lib.ann_quantized_evaluate_accuracy(self._ptr, x_tensor._ptr, y_tensor._ptr)
        if accuracy < 0.0:
            raise InvalidParameterError("Data shape does not match the model's input and output sizes")
        return accuracy

    def export_c_header(self, filepath: Union[str, Path], prefix: Optional[str] = None,
                        runtime: bool = True) -> Path:
        """Export the int8 model as a self-contained C header.

        The header holds the parameters as ``static const`` arrays and defines
        ``<prefix>_predict`` (int8 in and out), ``<prefix>_predict_float`` and
        ``<prefix>_argmax``, plus size and quantization macros. It needs
        ``ann_rt.h``, which is copied next to it unless ``runtime`` is False.

        Args:
            filepath: Output file, e.g. ``"model.h"``
            prefix: C identifier naming the header's symbols (default: from
                the file name)
            runtime: Also write ``ann_rt.h`` into the same directory

        Returns:
            The header's path
        """
        return export_c_header(ffi.NULL, self._ptr, filepath, prefix, runtime)

    def __repr__(self) -> str:
        return f"QuantizedModel(inputs={self.input_size}, outputs={self.output_size}, int8)"

    def __del__(self):
        if getattr(self, "_ptr", None) is not None and self._ptr != ffi.NULL:
            lib.ann_free_quantized(self._ptr)
            self._ptr = None
