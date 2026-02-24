import pytest
import torch

from language_machine.transformer.linear import Linear


class TestLinearInit:
    def test_weight_shape(self):
        linear = Linear(64, 128)
        assert linear.W.shape == (64, 128)

    def test_weight_is_parameter(self):
        linear = Linear(64, 128)
        assert isinstance(linear.W, torch.nn.Parameter)

    def test_device_argument(self):
        linear = Linear(64, 128, device="cpu")
        assert linear.W.device == torch.device("cpu")

    def test_dtype_argument(self):
        linear = Linear(64, 128, dtype=torch.float64)
        assert linear.W.dtype == torch.float64

    def test_default_dtype_is_float32(self):
        linear = Linear(64, 128)
        assert linear.W.dtype == torch.float32


class TestLinearForward:
    def test_output_shape_2d(self):
        linear = Linear(64, 128)
        x = torch.randn(10, 64)  # (batch, in_features)
        out = linear(x)
        assert out.shape == (10, 128)

    def test_output_shape_3d(self):
        linear = Linear(64, 128)
        x = torch.randn(32, 10, 64)  # (batch, seq, in_features)
        out = linear(x)
        assert out.shape == (32, 10, 128)

    def test_output_shape_1d(self):
        linear = Linear(64, 128)
        x = torch.randn(64)  # (in_features,)
        out = linear(x)
        assert out.shape == (128,)

    def test_forward_is_matmul(self):
        linear = Linear(64, 128)
        x = torch.randn(10, 64)
        out = linear(x)
        expected = x @ linear.W
        assert torch.allclose(out, expected)


class TestLinearInitialization:
    def test_weights_not_zero(self):
        linear = Linear(64, 128)
        assert not torch.allclose(linear.W, torch.zeros_like(linear.W))

    def test_weights_mean_near_zero(self):
        linear = Linear(256, 256)
        mean = linear.W.mean().item()
        assert abs(mean) < 0.1  # should be close to 0

    def test_weights_std_follows_xavier(self):
        in_f, out_f = 256, 256
        linear = Linear(in_f, out_f)
        expected_std = (2 / (in_f + out_f)) ** 0.5
        actual_std = linear.W.std().item()
        # Allow some tolerance since it's truncated
        assert abs(actual_std - expected_std) < 0.05

    def test_weights_are_truncated(self):
        in_f, out_f = 256, 256
        linear = Linear(in_f, out_f)
        std = (2 / (in_f + out_f)) ** 0.5
        # All values should be within [-3*std, 3*std]
        assert linear.W.min() >= -3 * std
        assert linear.W.max() <= 3 * std


class TestLinearGradient:
    def test_weights_require_grad(self):
        linear = Linear(64, 128)
        assert linear.W.requires_grad

    def test_backward_pass(self):
        linear = Linear(64, 128)
        x = torch.randn(10, 64)
        out = linear(x)
        loss = out.sum()
        loss.backward()
        assert linear.W.grad is not None
        assert linear.W.grad.shape == (64, 128)
