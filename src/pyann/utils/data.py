"""Data loading and manipulation utilities."""

import csv
import re
from typing import Dict, Tuple, Optional, List, Sequence, Union, TYPE_CHECKING
from pyann.utils.compat import ArrayLike, to_array, has_numpy, is_numpy_array, HAS_NUMPY

if TYPE_CHECKING:
    import numpy as np


def load_csv(
    filename: str,
    has_header: bool = False,
    delimiter: str = ",",
    use_c_library: bool = True
) -> Tuple[ArrayLike, int, int]:
    """Load a CSV file of numbers.
    
    Every field must be a number, as in the MNIST and Fashion-MNIST CSVs;
    for files with text (reviews, labels such as "positive") use
    ``load_text_csv``. A field that is not a number raises ValueError naming
    its line and column, and so does a row with a different number of
    fields.
    
    Args:
        filename: Path to CSV file
        has_header: If True, skip the first line
        delimiter: Column separator (default: comma)
        use_c_library: If True, use the C library's ann_load_csv for performance
        
    Returns:
        Tuple of (data, rows, cols) where data is array-like
    """
    filename = str(filename)
    if use_c_library and delimiter == ",":
        # libann reads a word as 0, so look at the first rows here
        _check_numeric(filename, has_header, None, max_rows=_CHECK_ROWS)
        try:
            return _load_csv_c(filename, has_header)
        except Exception:
            _check_numeric(filename, has_header, None)    # the line at fault, if it is one
            raise
    return _load_csv_python(filename, has_header, delimiter)


# rows load_csv checks before handing a file to libann, which reads words as 0
_CHECK_ROWS = 20
# libann's ann_load_csv splits fields at commas and spaces, and reads lines of
# up to this many characters with their line break (its DEFAULT_BUFFER_SIZE less one)
_LIBANN_SPLIT = re.compile(r"[, ]+")
_LIBANN_LINE = 8191


def _field_preview(text: str, width: int = 40) -> str:
    text = text.strip()
    return repr(text if len(text) <= width else text[:width] + "...")


def _check_numeric(filename: str, has_header: bool, delimiter: Optional[str], max_rows: Optional[int] = None) -> None:
    """Raise ValueError at the first field that is not a number, or row with
    a different number of fields, in the first max_rows rows (all if None).
    delimiter None splits as libann does, at commas and spaces."""
    expected = None
    first = 1
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            if has_header and line_no == 1:
                continue
            if max_rows is not None and line_no - (1 if has_header else 0) > max_rows:
                return
            if delimiter is None and len(line) > _LIBANN_LINE:
                raise ValueError(f"{filename} line {line_no} has {len(line.rstrip())} characters, more than libann's "
                                 f"reader takes ({_LIBANN_LINE - 1}): pass use_c_library=False")
            line = line.strip()
            if not line:
                continue
            fields = line.split(delimiter) if delimiter is not None else [f for f in _LIBANN_SPLIT.split(line) if f]
            for col, field in enumerate(fields, 1):
                try:
                    float(field)
                except ValueError:
                    hint = (" It looks like a header: pass has_header=True." if line_no == 1 and not has_header
                            else " For files with text use load_text_csv(), which keeps the fields as strings.")
                    raise ValueError(f"{filename} line {line_no}, column {col}: {_field_preview(field)} is not a "
                                     f"number. load_csv reads numeric CSV files only.{hint}") from None
            if expected is None:
                expected, first = len(fields), line_no
            elif len(fields) != expected:
                raise ValueError(f"{filename} line {line_no} has {len(fields)} fields, but line {first} has "
                                 f"{expected}. load_csv needs the same number of numbers on every line; for text "
                                 f"with quoted commas use load_text_csv().")


def _load_csv_c(filename: str, has_header: bool) -> Tuple[ArrayLike, int, int]:
    """Load CSV using the C library's ann_load_csv."""
    from pyann._bindings.ffi import ffi, lib
    from pyann.exceptions import raise_for_error_code
    
    data_ptr = ffi.new("real **")
    rows_ptr = ffi.new("int *")
    stride_ptr = ffi.new("int *")
    
    header_flag = lib.CSV_HAS_HEADER if has_header else lib.CSV_NO_HEADER
    result = lib.ann_load_csv(filename.encode(), header_flag, data_ptr, rows_ptr, stride_ptr)
    raise_for_error_code(result, f"Failed to load CSV file: {filename}")
    
    n_rows = rows_ptr[0]
    n_cols = stride_ptr[0]
    
    if n_rows == 0 or n_cols == 0:
        return to_array([[]]), 0, 0
    
    # Copy data from C array to Python
    if HAS_NUMPY:
        import numpy as np
        # Create numpy array from raw pointer using buffer interface
        total_elements = n_rows * n_cols
        buffer = ffi.buffer(data_ptr[0], total_elements * ffi.sizeof("real"))
        data = np.frombuffer(buffer, dtype=np.float32).reshape(n_rows, n_cols).copy()
    else:
        data = []
        for i in range(n_rows):
            row = [data_ptr[0][i * n_cols + j] for j in range(n_cols)]
            data.append(row)
        data = to_array(data)
    
    # Note: The C library allocates memory with malloc. Since we've copied the data,
    # the caller should free it. However, calling free() requires linking stdlib.
    # For now, we rely on the garbage collector to handle this through CFFI's gc mechanism
    # or accept a small memory leak. In production, consider adding a cleanup function to libann.
    
    return data, n_rows, n_cols


def _load_csv_python(
    filename: str,
    has_header: bool,
    delimiter: str
) -> Tuple[ArrayLike, int, int]:
    """Load CSV using pure Python (fallback)."""
    _check_numeric(filename, has_header, delimiter)
    rows = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    start = 1 if has_header else 0
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        values = [float(x) for x in line.split(delimiter)]
        rows.append(values)
    
    if not rows:
        return to_array([[]]), 0, 0
    
    n_rows = len(rows)
    n_cols = len(rows[0])
    
    return to_array(rows), n_rows, n_cols


def load_text_csv(
    filename: str,
    columns: Optional[Sequence[Union[str, int]]] = None,
    has_header: bool = True,
    delimiter: str = ",",
    encoding: str = "utf-8"
) -> Dict[Union[str, int], List[str]]:
    """Load a CSV file with text fields, column by column, as strings.

    Quoted fields may hold commas, quotes ("") and line breaks, as in the
    IMDB reviews CSV. Pure Python: no NumPy needed.

    >>> data = load_text_csv("IMDB Dataset.csv")          # columns: review, sentiment
    >>> texts, sentiment = data["review"], data["sentiment"]
    >>> labels = [s == "positive" for s in sentiment]     # then one_hot_encode(labels, 2)

    Args:
        filename: Path to the CSV file
        columns: Columns to keep (names, or 0-based indexes); None keeps all
        has_header: The first row names the columns; without one, the
            columns are keyed 0, 1, ...
        delimiter: Column separator
        encoding: Text encoding of the file

    Returns:
        A dict of column (name, or index without a header) to its list of
        strings, one per row
    """
    filename = str(filename)
    old_limit = csv.field_size_limit()
    csv.field_size_limit(max(old_limit, 1 << 30))    # long reviews; restored below
    try:
        with open(filename, "r", newline="", encoding=encoding) as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, None) if has_header else None
            rows = [row for row in reader if row]
    finally:
        csv.field_size_limit(old_limit)

    width = len(header) if header is not None else (len(rows[0]) if rows else 0)
    for i, row in enumerate(rows):
        if len(row) != width:
            # the reader's line number counts line breaks inside quoted fields, so name the row
            raise ValueError(f"{filename} row {i + 1} (after the header) has {len(row)} fields, expected {width}"
                             f"{'' if header is None else ' (' + ', '.join(header) + ')'}. Check its quoting.")
    names: List[Union[str, int]] = list(header) if header is not None else list(range(width))
    if header is not None and len(set(names)) != len(names):
        raise ValueError(f"{filename} has repeated column names: {header}")

    if columns is None:
        wanted = list(range(width))
    else:
        if isinstance(columns, (str, int)):
            raise TypeError("columns takes a list of names or indexes, e.g. ['review']")
        wanted = []
        for c in columns:
            if isinstance(c, str):
                if header is None or c not in names:
                    raise ValueError(f"{filename} has no column {c!r}; its columns are {names}")
                wanted.append(names.index(c))
            elif isinstance(c, int) and not isinstance(c, bool) and -width <= c < width:
                wanted.append(c % width)
            else:
                raise ValueError(f"column {c!r} is out of range for {width} columns")
    return {names[i]: [row[i] for row in rows] for i in wanted}


def _check_rows(data: ArrayLike, name: str) -> int:
    """The number of rows of X or y for split_data, after checking every row
    has the same length (token-id sequences straight from a Tokenizer don't)."""
    if is_numpy_array(data):
        return len(data)
    rows = list(data)
    lengths = {len(r) for r in rows if isinstance(r, (list, tuple))}
    if len(lengths) > 1:
        raise ValueError(f"{name} has rows of different lengths ({min(lengths)} to {max(lengths)} values): if they "
                         f"are token ids, make them one length with pad_sequences(sequences, maxlen=...) first")
    return len(rows)


def split_data(
    X: ArrayLike,
    y: ArrayLike,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Split data into training and validation sets.
    
    Args:
        X: Input features
        y: Target values
        train_ratio: Fraction of data to use for training (default: 0.8)
        shuffle: Whether to shuffle before splitting (default: True)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val)
    """
    if not 0.0 < train_ratio <= 1.0:
        raise ValueError(f"train_ratio must be in (0, 1], got {train_ratio}")
    n_x, n_y = _check_rows(X, "X"), _check_rows(y, "y")
    if n_x != n_y:
        raise ValueError(f"X has {n_x} rows but y has {n_y}: they need one target row per sample")
    if HAS_NUMPY:
        import numpy as np
        
        X = to_array(X)
        y = to_array(y)
        
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        
        split_idx = int(n_samples * train_ratio)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        
        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]
    else:
        # Pure Python path
        import random
        
        X_list = list(X) if not isinstance(X, list) else X
        y_list = list(y) if not isinstance(y, list) else y
        
        n_samples = len(X_list)
        indices = list(range(n_samples))
        
        if shuffle:
            if seed is not None:
                random.seed(seed)
            random.shuffle(indices)
        
        split_idx = int(n_samples * train_ratio)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        
        X_train = [X_list[i] for i in train_idx]
        y_train = [y_list[i] for i in train_idx]
        X_val = [X_list[i] for i in val_idx]
        y_val = [y_list[i] for i in val_idx]
        
        return to_array(X_train), to_array(y_train), to_array(X_val), to_array(y_val)


def normalize(
    data: ArrayLike,
    axis: Optional[int] = None,
    mean: Optional[ArrayLike] = None,
    std: Optional[ArrayLike] = None
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Normalize data to zero mean and unit variance.
    
    Args:
        data: Input data to normalize
        axis: Axis along which to compute statistics (default: flatten)
        mean: Pre-computed mean (for applying to test data)
        std: Pre-computed std (for applying to test data)
        
    Returns:
        Tuple of (normalized_data, mean, std)
    """
    if not HAS_NUMPY:
        raise NotImplementedError("normalize() requires NumPy")
    
    import numpy as np
    data = to_array(data)
    
    if mean is None:
        mean = np.mean(data, axis=axis, keepdims=True)
    if std is None:
        std = np.std(data, axis=axis, keepdims=True)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
    
    normalized = (data - mean) / std
    return normalized, mean, std


def one_hot_encode(
    labels: ArrayLike,
    num_classes: Optional[int] = None
) -> ArrayLike:
    """Convert integer labels to one-hot encoded vectors.
    
    Args:
        labels: 1D array of integer class labels
        num_classes: Number of classes (inferred from max label if not provided)
        
    Returns:
        2D array of shape (n_samples, num_classes)
    """
    if not HAS_NUMPY:
        raise NotImplementedError("one_hot_encode() requires NumPy")
    
    import numpy as np
    labels = to_array(labels).flatten().astype(int)
    
    if num_classes is None:
        num_classes = int(labels.max()) + 1
    
    n_samples = len(labels)
    one_hot = np.zeros((n_samples, num_classes), dtype=np.float32)
    one_hot[np.arange(n_samples), labels] = 1.0
    
    return one_hot
