"""Tests for layer classes."""

import pytest
from pyann.core.layers import Layer, Input, Dense, parse_layer
from pyann.activations import Activation


class TestInput:
    """Tests for Input layer."""
    
    def test_input_shape(self):
        """Test input shape storage."""
        layer = Input(shape=(784,))
        assert layer.shape == (784,)
        assert layer.units == 784
    
    def test_input_2d_shape(self):
        """Test 2D input shape flattening."""
        layer = Input(shape=(28, 28))
        assert layer.units == 784


class TestDense:
    """Tests for Dense layer."""
    
    def test_basic_dense(self):
        """Test basic dense layer creation."""
        layer = Dense(64)
        assert layer.units == 64
        assert layer.activation is None
        assert layer.activation_type == Activation.NONE
    
    def test_dense_with_activation(self):
        """Test dense layer with activation."""
        layer = Dense(64, activation="relu")
        assert layer.units == 64
        assert layer.activation_type == Activation.RELU
    
    def test_dense_with_enum_activation(self):
        """Test dense layer with enum activation."""
        layer = Dense(64, activation=Activation.SIGMOID)
        assert layer.activation_type == Activation.SIGMOID
    
    def test_dense_with_input_shape(self):
        """Test dense layer with input shape."""
        layer = Dense(128, activation="relu", input_shape=(784,))
        assert layer.input_shape == (784,)
        assert layer.input_units == 784
    
    def test_dense_with_dropout(self):
        """Test dense layer with dropout."""
        layer = Dense(64, dropout=0.5)
        assert layer.dropout == 0.5
    
    def test_invalid_units(self):
        """Test invalid units raises ValueError."""
        with pytest.raises(ValueError):
            Dense(0)
        with pytest.raises(ValueError):
            Dense(-1)
    
    def test_invalid_dropout(self):
        """Test invalid dropout raises ValueError."""
        with pytest.raises(ValueError):
            Dense(64, dropout=1.0)
        with pytest.raises(ValueError):
            Dense(64, dropout=-0.1)


class TestParseLayer:
    """Tests for parse_layer helper."""
    
    def test_parse_layer_instance(self):
        """Test layer passthrough."""
        original = Dense(64, activation="relu")
        result = parse_layer(original)
        assert result is original
    
    def test_parse_int(self):
        """Test int to Dense conversion."""
        layer = parse_layer(64)
        assert isinstance(layer, Dense)
        assert layer.units == 64
    
    def test_parse_tuple(self):
        """Test tuple to Dense conversion."""
        layer = parse_layer((64, "relu"))
        assert isinstance(layer, Dense)
        assert layer.units == 64
        assert layer.activation_type == Activation.RELU
    
    def test_parse_invalid(self):
        """Test invalid input raises TypeError."""
        with pytest.raises(TypeError):
            parse_layer("invalid")
