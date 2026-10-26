"""Tests for data utilities."""

import pytest
from pyann.utils.data import load_csv, load_text_csv, split_data, one_hot_encode
from pyann.utils.compat import has_numpy, get_shape
import tempfile
import os


class TestLoadCsv:
    """Tests for CSV loading."""
    
    def test_load_simple_csv(self):
        """Test loading a simple CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("1.0,2.0,3.0\n")
            f.write("4.0,5.0,6.0\n")
            f.write("7.0,8.0,9.0\n")
            fname = f.name
        
        try:
            data, rows, cols = load_csv(fname)
            assert rows == 3
            assert cols == 3
        finally:
            os.unlink(fname)
    
    def test_load_csv_with_header(self):
        """Test loading CSV with header."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("a,b,c\n")
            f.write("1.0,2.0,3.0\n")
            f.write("4.0,5.0,6.0\n")
            fname = f.name
        
        try:
            data, rows, cols = load_csv(fname, has_header=True)
            assert rows == 2
            assert cols == 3
        finally:
            os.unlink(fname)


class TestSplitData:
    """Tests for data splitting."""
    
    @pytest.mark.skipif(not has_numpy(), reason="NumPy required")
    def test_split_basic(self):
        """Test basic data split."""
        import numpy as np
        
        X = np.arange(100).reshape(50, 2).astype(np.float32)
        y = np.arange(50).reshape(50, 1).astype(np.float32)
        
        X_train, y_train, X_val, y_val = split_data(X, y, train_ratio=0.8, shuffle=False)
        
        assert X_train.shape[0] == 40
        assert X_val.shape[0] == 10
        assert y_train.shape[0] == 40
        assert y_val.shape[0] == 10
    
    @pytest.mark.skipif(not has_numpy(), reason="NumPy required")
    def test_split_with_seed(self):
        """Test reproducible split with seed."""
        import numpy as np
        
        X = np.arange(100).reshape(50, 2).astype(np.float32)
        y = np.arange(50).reshape(50, 1).astype(np.float32)
        
        X_train1, _, _, _ = split_data(X, y, shuffle=True, seed=42)
        X_train2, _, _, _ = split_data(X, y, shuffle=True, seed=42)
        
        assert np.allclose(X_train1, X_train2)


class TestOneHotEncode:
    """Tests for one-hot encoding."""
    
    @pytest.mark.skipif(not has_numpy(), reason="NumPy required")
    def test_one_hot_basic(self):
        """Test basic one-hot encoding."""
        import numpy as np
        
        labels = np.array([0, 1, 2, 1, 0])
        one_hot = one_hot_encode(labels, num_classes=3)
        
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ], dtype=np.float32)
        
        assert np.allclose(one_hot, expected)
    
    @pytest.mark.skipif(not has_numpy(), reason="NumPy required")
    def test_one_hot_infer_classes(self):
        """Test inferring number of classes."""
        import numpy as np
        
        labels = np.array([0, 1, 2])
        one_hot = one_hot_encode(labels)
        
        assert one_hot.shape == (3, 3)


class TestUtilsPackage:
    """pyann.utils exports every data helper, as the README imports them."""

    def test_readme_data_utilities(self, tmp_path):
        np = pytest.importorskip("numpy")
        from pyann.utils import load_csv, normalize, one_hot_encode, split_data
        path = tmp_path / "data.csv"
        path.write_text("a,b,label\n" + "".join(f"{i},{2 * i},{i % 3}\n" for i in range(10)))
        data, rows, cols = load_csv(str(path), has_header=True)
        X, labels = data[:, :-1], data[:, -1]
        X_norm, mean, std = normalize(X, axis=0)
        assert np.allclose(X_norm.mean(axis=0), 0, atol=1e-6) and mean.shape == (1, 2)
        y = one_hot_encode(labels, num_classes=10)
        assert y.shape == (10, 10) and y[4, 1] == 1.0
        X_train, y_train, X_val, y_val = split_data(X_norm, y, train_ratio=0.8)
        assert X_train.shape == (8, 2) and y_val.shape == (2, 10)


IMDB_LIKE = (
    'review,sentiment\n'
    '"One of the reviewers said, ""really"" good.<br /><br />Yes",positive\n'
    '"A wonderful little production.\nA line break inside",positive\n'
    'Basically there\'s a family,negative\n'
)



def write_csv(path, text):
    """Write text with exactly these line breaks (write_text makes them \\r\\n on Windows)."""
    path.write_bytes(text.encode("utf-8"))


class TestLoadCsvErrors:
    """load_csv names the field or line at fault instead of misreading text."""

    @pytest.mark.parametrize("use_c", [True, False])
    def test_text_field(self, tmp_path, use_c):
        path = tmp_path / "imdb.csv"
        write_csv(path, IMDB_LIKE)
        with pytest.raises(ValueError, match=r"line 2, column 1: .*not a number.*load_text_csv"):
            load_csv(str(path), has_header=True, use_c_library=use_c)

    def test_text_label_column(self, tmp_path):
        # libann alone would read "positive" as 0 without complaint
        path = tmp_path / "labels.csv"
        path.write_text("1,2,positive\n3,4,negative\n")
        with pytest.raises(ValueError, match=r"line 1, column 3: 'positive'"):
            load_csv(str(path))

    def test_header_hint(self, tmp_path):
        path = tmp_path / "header.csv"
        path.write_text("a,b\n1,2\n")
        with pytest.raises(ValueError, match="has_header=True"):
            load_csv(str(path))

    def test_ragged_rows(self, tmp_path):
        path = tmp_path / "ragged.csv"
        path.write_text("1,2,3\n4,5\n")
        with pytest.raises(ValueError, match="line 2 has 2 fields, but line 1 has 3"):
            load_csv(str(path))

    def test_problem_past_the_first_rows(self, tmp_path):
        # found when libann fails, after the quick check of the first rows
        path = tmp_path / "late.csv"
        path.write_text("1,2\n" * 50 + "1,2,3\n")
        with pytest.raises(ValueError, match="line 51 has 3 fields"):
            load_csv(str(path))

    def test_line_too_long_for_libann(self, tmp_path):
        path = tmp_path / "wide.csv"
        row = ",".join(["0.5"] * 3000) + "\n"
        path.write_text(row * 2)
        with pytest.raises(ValueError, match="use_c_library=False"):
            load_csv(str(path))
        data, rows, cols = load_csv(str(path), use_c_library=False)
        assert (rows, cols) == (2, 3000)

    def test_numbers_still_load(self, tmp_path):
        # spaces separate fields for libann, as before
        path = tmp_path / "ok.csv"
        path.write_text("1, 2 ,3\n4 5 6\n-1e-3,nan,inf\n")
        data, rows, cols = load_csv(str(path))
        assert (rows, cols) == (3, 3)


class TestLoadTextCsv:
    """load_text_csv keeps every field as a string, quoting and all."""

    def test_columns_by_name(self, tmp_path):
        path = tmp_path / "imdb.csv"
        write_csv(path, IMDB_LIKE)
        data = load_text_csv(path)
        assert list(data) == ["review", "sentiment"]
        assert data["sentiment"] == ["positive", "positive", "negative"]
        assert data["review"][0] == 'One of the reviewers said, "really" good.<br /><br />Yes'
        assert data["review"][1] == "A wonderful little production.\nA line break inside"

    def test_windows_line_breaks(self, tmp_path):
        path = tmp_path / "crlf.csv"
        write_csv(path, IMDB_LIKE.replace("\n", "\r\n"))
        data = load_text_csv(path)
        assert data["sentiment"] == ["positive", "positive", "negative"]
        assert data["review"][1] == "A wonderful little production.\r\nA line break inside"   # kept as written
        numbers = tmp_path / "crlf_numbers.csv"
        write_csv(numbers, "a,b\r\n1,2\r\n3,4\r\n")
        assert load_csv(str(numbers), has_header=True)[1:] == (2, 2)

    def test_selecting_columns(self, tmp_path):
        path = tmp_path / "imdb.csv"
        write_csv(path, IMDB_LIKE)
        assert list(load_text_csv(path, columns=["sentiment"])) == ["sentiment"]
        assert list(load_text_csv(path, columns=[-1, 0])) == ["sentiment", "review"]
        with pytest.raises(ValueError, match="no column 'label'"):
            load_text_csv(path, columns=["label"])
        with pytest.raises(ValueError, match="out of range"):
            load_text_csv(path, columns=[2])
        with pytest.raises(TypeError):
            load_text_csv(path, columns="review")

    def test_without_a_header(self, tmp_path):
        path = tmp_path / "plain.tsv"
        path.write_text("a b\tx\nc\ty\n")
        data = load_text_csv(path, has_header=False, delimiter="\t")
        assert data == {0: ["a b", "c"], 1: ["x", "y"]}

    def test_long_fields(self, tmp_path):
        # longer than the csv module's default limit (131072 characters), which is restored after
        import csv
        path = tmp_path / "long.csv"
        path.write_text("text,label\n" + '"' + "word " * 40000 + '",1\n')
        limit = csv.field_size_limit()
        assert len(load_text_csv(path)["text"][0]) == 200000
        assert csv.field_size_limit() == limit

    def test_rows_of_another_width(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("review,sentiment\nfine,positive\nan unquoted, comma,negative\n")
        with pytest.raises(ValueError, match="row 2 .* 3 fields, expected 2"):
            load_text_csv(path)

    @pytest.mark.skipif(not has_numpy(), reason="NumPy required")
    def test_into_a_text_model(self, tmp_path):
        import numpy as np
        from pyann import Tokenizer, pad_sequences, strip_html
        path = tmp_path / "imdb.csv"
        write_csv(path, IMDB_LIKE)
        data = load_text_csv(path)
        tok = Tokenizer(preprocess=strip_html).fit_on_texts(data["review"])
        assert "br" not in tok.word_index
        x = pad_sequences(tok.texts_to_sequences(data["review"]), maxlen=8)
        y = one_hot_encode(np.array([s == "positive" for s in data["sentiment"]]), num_classes=2)
        X_train, y_train, X_val, y_val = split_data(x, y, train_ratio=2 / 3, seed=1)
        assert X_train.shape == (2, 8) and y_val.shape == (1, 2)


class TestSplitDataErrors:
    def test_unpadded_sequences(self):
        y = [[1.0], [0.0], [1.0]]
        with pytest.raises(ValueError, match="different lengths.*pad_sequences"):
            split_data([[1, 2, 3], [4], [5, 6]], y)

    def test_row_counts_differ(self):
        with pytest.raises(ValueError, match="X has 3 rows but y has 2"):
            split_data([[1.0], [2.0], [3.0]], [[1.0], [0.0]])

    def test_train_ratio(self):
        with pytest.raises(ValueError, match="train_ratio"):
            split_data([[1.0], [2.0]], [[1.0], [0.0]], train_ratio=0.0)
        with pytest.raises(ValueError, match="train_ratio"):
            split_data([[1.0], [2.0]], [[1.0], [0.0]], train_ratio=80)
