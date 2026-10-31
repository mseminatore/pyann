"""Tests for loss functions."""

import pytest
from pyann import Sequential, Input, Dense
from pyann._bindings.ffi import lib
from pyann.losses import Loss, LossReduction, parse_loss, parse_loss_reduction


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
        assert Loss.from_string("binary_crossentropy") == Loss.BINARY_CROSS_ENTROPY
        assert Loss.from_string("bce") == Loss.BCE
    
    def test_from_string_invalid(self):
        """Test invalid string raises ValueError."""
        with pytest.raises(ValueError):
            Loss.from_string("invalid")
    
    def test_aliases(self):
        """Test loss function aliases."""
        assert Loss.MEAN_SQUARED_ERROR == Loss.MSE
        assert Loss.CCE == Loss.CATEGORICAL_CROSS_ENTROPY
        assert Loss.CROSS_ENTROPY == Loss.CATEGORICAL_CROSS_ENTROPY
        assert Loss.BINARY_CROSS_ENTROPY == Loss.CATEGORICAL_CROSS_ENTROPY
        assert Loss.BCE == Loss.CATEGORICAL_CROSS_ENTROPY


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


class TestLossReduction:
    """Tests for compile(loss_reduction=...) and the LossReduction enum."""

    def test_enum_values(self):
        assert LossReduction.SUM == lib.LOSS_REDUCTION_SUM == 0
        assert LossReduction.MEAN == lib.LOSS_REDUCTION_MEAN == 1

    def test_parse(self):
        assert parse_loss_reduction("mean") == LossReduction.MEAN
        assert parse_loss_reduction("SUM") == LossReduction.SUM
        assert parse_loss_reduction(LossReduction.MEAN) == LossReduction.MEAN
        with pytest.raises(ValueError, match="mean, sum"):
            parse_loss_reduction("average")
        with pytest.raises(TypeError):
            parse_loss_reduction(1)

    def test_default_and_compile(self):
        model = Sequential([Input((2,)), Dense(3, activation="linear")])
        assert model.loss_reduction == LossReduction.SUM
        model.compile(loss_reduction="mean")
        assert model.loss_reduction == LossReduction.MEAN
        model.get_weights()                              # builds the network
        assert lib.ann_get_loss_reduction(model._ptr) == lib.LOSS_REDUCTION_MEAN
        model.compile(loss_reduction=LossReduction.SUM)  # a built model changes at once
        assert lib.ann_get_loss_reduction(model._ptr) == lib.LOSS_REDUCTION_SUM
        assert model.loss_reduction == LossReduction.SUM

    def test_invalid_reduction_changes_nothing(self):
        model = Sequential([Input((2,)), Dense(3)]).compile(loss_reduction="mean")
        with pytest.raises(ValueError):
            model.compile(loss_reduction="median")
        assert model.loss_reduction == LossReduction.MEAN

    @pytest.mark.parametrize("loss, outputs, activation, factor", [
        ("mse", 3, "linear", 4 * 3 / 2),       # batch * outputs / 2
        ("bce", 3, "sigmoid", 4),              # batch
    ])
    def test_sum_steps_are_larger_than_mean(self, loss, outputs, activation, factor):
        """One SGD step with the sum reduction moves every parameter factor
        times as far as with the mean reduction (the gradient of the reported
        loss), for a batch of 4."""
        np = pytest.importorskip("numpy")
        rng = np.random.default_rng(0)
        x = rng.normal(size=(4, 2)).astype(np.float32)
        y = rng.random((4, outputs)).astype(np.float32)
        steps = []
        start = None
        for reduction in ("sum", "mean"):
            model = Sequential([Input((2,)), Dense(outputs, activation=activation)], optimizer="sgd", loss=loss)
            model.compile(learning_rate=0.01, loss_reduction=reduction)
            if start is None:
                start = model.get_weights()
            model.set_weights(start)
            model.train_on_batch(x, y)
            steps.append([after - before for after, before in zip(model.get_weights(), start)])
        for sum_step, mean_step in zip(*steps):
            # steps are differences of float32 weights: about 1e-7 of rounding, times factor
            np.testing.assert_allclose(sum_step, factor * mean_step, rtol=1e-4, atol=1e-6)
