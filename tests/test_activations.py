"""Tests for activation functions."""

import pytest
from pyann.activations import Activation, parse_activation


class TestActivation:
    """Tests for Activation enum."""
    
    def test_enum_values(self):
        """Test that enum values match C library."""
        assert Activation.NONE == 0
        assert Activation.SIGMOID == 1
        assert Activation.RELU == 2
        assert Activation.LEAKY_RELU == 3
        assert Activation.TANH == 4
        assert Activation.SOFTSIGN == 5
        assert Activation.SOFTMAX == 6
    
    def test_from_string(self):
        """Test string to enum conversion."""
        assert Activation.from_string("relu") == Activation.RELU
        assert Activation.from_string("RELU") == Activation.RELU
        assert Activation.from_string("sigmoid") == Activation.SIGMOID
        assert Activation.from_string("softmax") == Activation.SOFTMAX
        assert Activation.from_string("leaky_relu") == Activation.LEAKY_RELU
        assert Activation.from_string("leakyrelu") == Activation.LEAKY_RELU
    
    def test_from_string_invalid(self):
        """Test invalid string raises ValueError."""
        with pytest.raises(ValueError):
            Activation.from_string("invalid")
    
    def test_linear_alias(self):
        """Test LINEAR is alias for NONE."""
        assert Activation.LINEAR == Activation.NONE


class TestParseActivation:
    """Tests for parse_activation helper."""
    
    def test_parse_none(self):
        """Test None returns NONE."""
        assert parse_activation(None) == Activation.NONE
    
    def test_parse_string(self):
        """Test string parsing."""
        assert parse_activation("relu") == Activation.RELU
        assert parse_activation("sigmoid") == Activation.SIGMOID
    
    def test_parse_enum(self):
        """Test enum passthrough."""
        assert parse_activation(Activation.SOFTMAX) == Activation.SOFTMAX
    
    def test_parse_invalid_type(self):
        """Test invalid type raises TypeError."""
        with pytest.raises(TypeError):
            parse_activation(123)
