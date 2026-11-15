"""Weight storage formats: low-bit weights for small exported models."""

from enum import IntEnum
from typing import Union


class WeightFormat(IntEnum):
    """How a Dense or conv layer stores its weights, matching libann's
    Weight_format enum (see Sequential.set_weight_format).

    The low-bit formats keep one scale per output channel (each row of the
    weights):

    - INT4: integers -7..7 times max|row| / 7
    - TERNARY: -1, 0 or +1 times the mean |w| of the weights kept; weights
      with |w| <= 0.7 * mean |row| become 0
    - BINARY: -1 or +1 times mean |row|
    """

    FLOAT = 0     # WEIGHT_FORMAT_FLOAT - real weights, int8 when quantized (default)
    INT4 = 1      # WEIGHT_FORMAT_INT4 - 4 bits per weight
    TERNARY = 2   # WEIGHT_FORMAT_TERNARY - 2 bits per weight
    BINARY = 3    # WEIGHT_FORMAT_BINARY - 1 bit per weight

    @classmethod
    def from_string(cls, name: str) -> "WeightFormat":
        """Convert a name ('float', 'int4', 'ternary', 'binary'; any case) to
        a WeightFormat.

        Raises:
            ValueError: If name is not a format
        """
        name_map = {
            "float": cls.FLOAT,
            "int4": cls.INT4,
            "ternary": cls.TERNARY,
            "binary": cls.BINARY,
        }
        key = name.lower()
        if key not in name_map:
            valid = ", ".join(name_map)
            raise ValueError(f"Unknown weight format '{name}'. Valid options: {valid}")
        return name_map[key]

    @property
    def bits(self) -> int:
        """Bits per weight in a quantized model (8 for FLOAT, which quantizes to int8)."""
        return {WeightFormat.FLOAT: 8, WeightFormat.INT4: 4, WeightFormat.TERNARY: 2, WeightFormat.BINARY: 1}[self]


def parse_weight_format(fmt: Union[str, WeightFormat]) -> WeightFormat:
    """Parse a weight format argument (a name or a WeightFormat)."""
    if isinstance(fmt, WeightFormat):
        return fmt
    if isinstance(fmt, str):
        return WeightFormat.from_string(fmt)
    raise TypeError(f"format must be str or WeightFormat, got {type(fmt).__name__}")
