"""Tests for loss functions."""

import pytest
from pyann.losses import Loss, parse_loss


class TestLoss:
    """Tests for Loss enum."""
    
    def test_enum_values(self):
        """Test that enum values match C library."""
        assert Loss.MSE == 0
        assert Loss.CATEGORICAL_CROSS_ENTROPY == 1
    
    def test_from_string(self):
        """Test string to enum conversion."""
        assert Loss.from_string("mse") == Loss.MSE
        assert Loss.from_string("MSE") == Loss.MSE
        assert Loss.from_string("categorical_cross_entropy") == Loss.CATEGORICAL_CROSS_ENTROPY
        assert Loss.from_string("cce") == Loss.CCE
        assert Loss.from_string("cross_entropy") == Loss.CROSS_ENTROPY
    
    def test_from_string_invalid(self):
        """Test invalid string raises ValueError."""
        with pytest.raises(ValueError):
            Loss.from_string("invalid")
    
    def test_aliases(self):
        """Test loss function aliases."""
        assert Loss.MEAN_SQUARED_ERROR == Loss.MSE
        assert Loss.CCE == Loss.CATEGORICAL_CROSS_ENTROPY
        assert Loss.CROSS_ENTROPY == Loss.CATEGORICAL_CROSS_ENTROPY


class TestParseLoss:
    """Tests for parse_loss helper."""
    
    def test_parse_string(self):
        """Test string parsing."""
        assert parse_loss("mse") == Loss.MSE
        assert parse_loss("categorical_cross_entropy") == Loss.CATEGORICAL_CROSS_ENTROPY
    
    def test_parse_enum(self):
        """Test enum passthrough."""
        assert parse_loss(Loss.MSE) == Loss.MSE
    
    def test_parse_invalid_type(self):
        """Test invalid type raises TypeError."""
        with pytest.raises(TypeError):
            parse_loss(123)
