"""Tests for callbacks."""

import pytest
from pyann.callbacks import (
    StepDecay, ExponentialDecay, CosineAnnealing,
    step_decay_fn, exponential_decay_fn, cosine_annealing_fn,
    EarlyStopping
)


class TestStepDecay:
    """Tests for step decay scheduler."""
    
    def test_step_decay_basic(self):
        """Test basic step decay."""
        params = StepDecay(step_size=10, gamma=0.5)
        
        # Epoch 0-9: no decay
        assert step_decay_fn(0, 0.1, params) == pytest.approx(0.1)
        assert step_decay_fn(9, 0.1, params) == pytest.approx(0.1)
        
        # Epoch 10-19: one decay
        assert step_decay_fn(10, 0.1, params) == pytest.approx(0.05)
        assert step_decay_fn(19, 0.1, params) == pytest.approx(0.05)
        
        # Epoch 20+: two decays
        assert step_decay_fn(20, 0.1, params) == pytest.approx(0.025)


class TestExponentialDecay:
    """Tests for exponential decay scheduler."""
    
    def test_exponential_decay_basic(self):
        """Test basic exponential decay."""
        params = ExponentialDecay(gamma=0.9)
        
        assert exponential_decay_fn(0, 0.1, params) == pytest.approx(0.1)
        assert exponential_decay_fn(1, 0.1, params) == pytest.approx(0.09)
        assert exponential_decay_fn(2, 0.1, params) == pytest.approx(0.081)


class TestCosineAnnealing:
    """Tests for cosine annealing scheduler."""
    
    def test_cosine_annealing_endpoints(self):
        """Test cosine annealing at start and end."""
        params = CosineAnnealing(T_max=100, min_lr=0.0001)
        
        # At epoch 0, should be close to base_lr
        lr_start = cosine_annealing_fn(0, 0.1, params)
        assert lr_start == pytest.approx(0.1, rel=0.01)
        
        # At epoch T_max, should be min_lr
        lr_end = cosine_annealing_fn(100, 0.1, params)
        assert lr_end == pytest.approx(0.0001)
    
    def test_cosine_annealing_midpoint(self):
        """Test cosine annealing at midpoint."""
        params = CosineAnnealing(T_max=100, min_lr=0.0)
        
        # At midpoint, should be ~half of base_lr
        lr_mid = cosine_annealing_fn(50, 0.1, params)
        assert lr_mid == pytest.approx(0.05, rel=0.1)


class TestEarlyStopping:
    """Tests for early stopping callback."""
    
    def test_no_stop_on_improvement(self):
        """Test that training continues when loss improves."""
        callback = EarlyStopping(patience=3, min_delta=0.001)
        
        assert not callback.on_epoch_end(1, 0.5)
        assert not callback.on_epoch_end(2, 0.4)
        assert not callback.on_epoch_end(3, 0.3)
    
    def test_stop_after_patience(self):
        """Test that training stops after patience epochs without improvement."""
        callback = EarlyStopping(patience=3, min_delta=0.001)
        
        callback.on_epoch_end(1, 0.5)
        callback.on_epoch_end(2, 0.5)  # No improvement
        callback.on_epoch_end(3, 0.5)  # No improvement
        result = callback.on_epoch_end(4, 0.5)  # No improvement
        
        assert result is True
    
    def test_reset(self):
        """Test reset clears state."""
        callback = EarlyStopping(patience=3)
        
        callback.on_epoch_end(1, 0.5)
        callback.reset()
        
        assert callback._best_loss is None
        assert callback._wait == 0
