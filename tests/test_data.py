"""Tests for data utilities."""

import pytest
from pyann.utils.data import load_csv, split_data, one_hot_encode
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
