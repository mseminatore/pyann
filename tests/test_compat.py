"""Tests for compatibility utilities."""

import pytest
from pyann.utils.compat import (
    has_numpy, is_numpy_array, to_array, from_array, 
    get_shape, flatten
)


class TestHasNumpy:
    """Tests for NumPy detection."""
    
    def test_has_numpy_returns_bool(self):
        """Test that has_numpy returns a boolean."""
        result = has_numpy()
        assert isinstance(result, bool)


class TestToArray:
    """Tests for to_array conversion."""
    
    def test_list_to_array(self):
        """Test converting list to array."""
        data = [[1.0, 2.0], [3.0, 4.0]]
        result = to_array(data)
        shape = get_shape(result)
        assert shape == (2, 2)
    
    def test_flat_list_to_array(self):
        """Test converting flat list to array."""
        data = [1.0, 2.0, 3.0]
        result = to_array(data)
        shape = get_shape(result)
        # Flat list becomes 1-row array or stays 1D
        assert (len(shape) == 2 and shape[1] == 3) or (len(shape) == 1 and shape[0] == 3)
    
    @pytest.mark.skipif(not has_numpy(), reason="NumPy required")
    def test_numpy_passthrough(self):
        """Test NumPy array passthrough."""
        import numpy as np
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        result = to_array(data)
        assert result.dtype == np.float32


class TestGetShape:
    """Tests for get_shape."""
    
    def test_2d_list_shape(self):
        """Test shape of 2D list."""
        data = [[1, 2, 3], [4, 5, 6]]
        shape = get_shape(data)
        assert shape == (2, 3)
    
    def test_1d_list_shape(self):
        """Test shape of 1D list."""
        data = [1, 2, 3]
        shape = get_shape(data)
        assert shape == (3,)
    
    @pytest.mark.skipif(not has_numpy(), reason="NumPy required")
    def test_numpy_shape(self):
        """Test shape of NumPy array."""
        import numpy as np
        data = np.zeros((5, 10))
        shape = get_shape(data)
        assert shape == (5, 10)


class TestFlatten:
    """Tests for flatten."""
    
    def test_flatten_2d_list(self):
        """Test flattening 2D list."""
        data = [[1, 2], [3, 4]]
        result = flatten(data)
        assert result == [1, 2, 3, 4]
    
    def test_flatten_1d_list(self):
        """Test flattening 1D list."""
        data = [1, 2, 3]
        result = flatten(data)
        assert result == [1.0, 2.0, 3.0]
    
    @pytest.mark.skipif(not has_numpy(), reason="NumPy required")
    def test_flatten_numpy(self):
        """Test flattening NumPy array."""
        import numpy as np
        data = np.array([[1, 2], [3, 4]])
        result = flatten(data)
        assert result == [1, 2, 3, 4]
