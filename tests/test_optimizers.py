"""Tests for optimizers."""

import pytest
from pyann.optimizers import (
    OptimizerType, Optimizer, SGD, Momentum, RMSProp, AdaGrad, Adam,
    parse_optimizer
)


class TestOptimizerType:
    """Tests for OptimizerType enum."""
    
    def test_enum_values(self):
        """Test that enum values match C library."""
        assert OptimizerType.SGD == 0
        assert OptimizerType.MOMENTUM == 1
        assert OptimizerType.RMSPROP == 2
        assert OptimizerType.ADAGRAD == 3
        assert OptimizerType.ADAM == 4
    
    def test_from_string(self):
        """Test string to enum conversion."""
        assert OptimizerType.from_string("sgd") == OptimizerType.SGD
        assert OptimizerType.from_string("adam") == OptimizerType.ADAM
        assert OptimizerType.from_string("ADAM") == OptimizerType.ADAM
    
    def test_from_string_invalid(self):
        """Test invalid string raises ValueError."""
        with pytest.raises(ValueError):
            OptimizerType.from_string("invalid")


class TestOptimizers:
    """Tests for optimizer classes."""
    
    def test_sgd_defaults(self):
        """Test SGD default learning rate."""
        opt = SGD()
        assert opt.learning_rate == 0.05
        assert opt.optimizer_type == OptimizerType.SGD
    
    def test_sgd_custom_lr(self):
        """Test SGD with custom learning rate."""
        opt = SGD(lr=0.01)
        assert opt.learning_rate == 0.01
    
    def test_adam_defaults(self):
        """Test Adam default learning rate."""
        opt = Adam()
        assert opt.learning_rate == 0.001
        assert opt.optimizer_type == OptimizerType.ADAM
    
    def test_momentum_defaults(self):
        """Test Momentum defaults."""
        opt = Momentum()
        assert opt.learning_rate == 0.01
    
    def test_rmsprop_defaults(self):
        """Test RMSProp defaults."""
        opt = RMSProp()
        assert opt.learning_rate == 0.001
    
    def test_adagrad_defaults(self):
        """Test AdaGrad defaults."""
        opt = AdaGrad()
        assert opt.learning_rate == 0.01


class TestParseOptimizer:
    """Tests for parse_optimizer helper."""
    
    def test_parse_none(self):
        """Test None returns Adam."""
        opt = parse_optimizer(None)
        assert isinstance(opt, Adam)
    
    def test_parse_string(self):
        """Test string parsing."""
        opt = parse_optimizer("adam")
        assert isinstance(opt, Adam)
        
        opt = parse_optimizer("sgd")
        assert isinstance(opt, SGD)
    
    def test_parse_optimizer_type(self):
        """Test OptimizerType parsing."""
        opt = parse_optimizer(OptimizerType.ADAM)
        assert isinstance(opt, Adam)
    
    def test_parse_optimizer_instance(self):
        """Test optimizer passthrough."""
        original = Adam(lr=0.01)
        result = parse_optimizer(original)
        assert result is original
    
    def test_parse_invalid_type(self):
        """Test invalid type raises TypeError."""
        with pytest.raises(TypeError):
            parse_optimizer(123)
