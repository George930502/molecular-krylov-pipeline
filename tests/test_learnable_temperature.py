"""Tests for PR 2.2: Learnable temperature in SigmoidTopK.

ADR-001 requires SigmoidTopK.temperature to be an nn.Parameter (via log_temperature)
with min-temperature clamping. This enables the optimizer to learn the temperature
during NF training, while ensuring it never collapses below min_temperature.

TDD: RED phase — these tests should FAIL against the current plain-float implementation.
"""

import math
import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestLearnableTemperature:
    """Core tests for learnable temperature in SigmoidTopK."""

    def test_temperature_is_parameter(self):
        """SigmoidTopK.log_temperature must be an nn.Parameter."""
        from flows.particle_conserving_flow import SigmoidTopK

        topk = SigmoidTopK(temperature=1.0)
        assert hasattr(topk, 'log_temperature'), (
            "SigmoidTopK must have a 'log_temperature' attribute"
        )
        assert isinstance(topk.log_temperature, nn.Parameter), (
            "log_temperature must be nn.Parameter, "
            f"got {type(topk.log_temperature)}"
        )

    def test_temperature_property_returns_positive(self):
        """The .temperature property must always return a value > min_temperature."""
        from flows.particle_conserving_flow import SigmoidTopK

        topk = SigmoidTopK(temperature=1.0, min_temperature=0.1)
        temp = topk.temperature
        assert isinstance(temp, torch.Tensor), (
            f"temperature property should return Tensor, got {type(temp)}"
        )
        assert temp.item() > topk.min_temperature, (
            f"temperature {temp.item()} should be > min_temperature {topk.min_temperature}"
        )

    def test_temperature_setter_roundtrip(self):
        """Setting temperature=0.5 and reading back should give approximately 0.5."""
        from flows.particle_conserving_flow import SigmoidTopK

        topk = SigmoidTopK(temperature=1.0, min_temperature=0.1)
        topk.temperature = 0.5
        readback = topk.temperature.item()
        assert abs(readback - 0.5) < 1e-4, (
            f"Roundtrip failed: set 0.5, got {readback}"
        )

    def test_temperature_gradients_flow(self):
        """Gradient must flow through temperature to log_temperature."""
        from flows.particle_conserving_flow import SigmoidTopK

        topk = SigmoidTopK(temperature=1.0, min_temperature=0.1)
        logits = torch.randn(8, 6)

        # Forward pass with soft mode to get smooth gradients
        selection = topk(logits, 2, hard=False)
        weights = torch.arange(1, 7, dtype=torch.float32)
        loss = (selection * weights).sum()
        loss.backward()

        assert topk.log_temperature.grad is not None, (
            "No gradient flowed to log_temperature"
        )
        assert topk.log_temperature.grad.abs().item() > 1e-12, (
            f"log_temperature gradient is effectively zero: "
            f"{topk.log_temperature.grad.item()}"
        )

    def test_temperature_min_clamp(self):
        """Temperature must never go below min_temperature, even with extreme log_temperature."""
        from flows.particle_conserving_flow import SigmoidTopK

        topk = SigmoidTopK(temperature=1.0, min_temperature=0.1)
        # Force log_temperature to a very negative value
        with torch.no_grad():
            topk.log_temperature.fill_(-100.0)
        temp = topk.temperature.item()
        assert temp >= topk.min_temperature, (
            f"Temperature {temp} fell below min_temperature {topk.min_temperature}"
        )

    def test_flow_temperature_setter_compatibility(self):
        """ParticleConservingFlow.set_temperature(0.3) must work without error."""
        from flows.particle_conserving_flow import ParticleConservingFlow

        flow = ParticleConservingFlow(
            n_orbitals=6, n_alpha=2, n_beta=2, hidden_dims=[64, 64],
        )
        # This calls self.topk_selector.temperature = temperature
        flow.set_temperature(0.3)
        # Verify the temperature was actually set
        temp = flow.topk_selector.temperature
        if isinstance(temp, torch.Tensor):
            temp = temp.item()
        assert abs(temp - 0.3) < 1e-4, (
            f"set_temperature(0.3) failed: topk_selector.temperature = {temp}"
        )

    def test_learnable_temp_in_optimizer(self):
        """log_temperature must appear in flow.parameters() so the optimizer can update it."""
        from flows.particle_conserving_flow import ParticleConservingFlow

        flow = ParticleConservingFlow(
            n_orbitals=6, n_alpha=2, n_beta=2, hidden_dims=[64, 64],
        )
        param_names = [name for name, _ in flow.named_parameters()]
        assert any('log_temperature' in name for name in param_names), (
            f"log_temperature not found in flow.parameters(). "
            f"Parameter names: {param_names}"
        )


class TestLearnableTemperatureEdgeCases:
    """Additional edge case tests."""

    def test_default_min_temperature(self):
        """Default min_temperature should be 0.1."""
        from flows.particle_conserving_flow import SigmoidTopK

        topk = SigmoidTopK(temperature=1.0)
        assert topk.min_temperature == 0.1, (
            f"Default min_temperature should be 0.1, got {topk.min_temperature}"
        )

    def test_custom_min_temperature(self):
        """Custom min_temperature should be respected."""
        from flows.particle_conserving_flow import SigmoidTopK

        topk = SigmoidTopK(temperature=2.0, min_temperature=0.5)
        assert topk.min_temperature == 0.5
        temp = topk.temperature.item()
        assert abs(temp - 2.0) < 1e-3, (
            f"Initial temperature should be ~2.0, got {temp}"
        )

    def test_temperature_initial_value(self):
        """Initial temperature property should match the constructor argument."""
        from flows.particle_conserving_flow import SigmoidTopK

        for init_temp in [0.5, 1.0, 2.0, 5.0]:
            topk = SigmoidTopK(temperature=init_temp, min_temperature=0.1)
            readback = topk.temperature.item()
            assert abs(readback - init_temp) < 1e-3, (
                f"Init temp={init_temp}, readback={readback}"
            )

    def test_setter_with_value_near_min(self):
        """Setting temperature to a value just above min_temperature should work."""
        from flows.particle_conserving_flow import SigmoidTopK

        topk = SigmoidTopK(temperature=1.0, min_temperature=0.1)
        topk.temperature = 0.15  # just above min
        readback = topk.temperature.item()
        assert abs(readback - 0.15) < 1e-3, (
            f"Set temp=0.15 near min, got {readback}"
        )

    def test_forward_uses_learnable_temperature(self):
        """SigmoidTopK forward must use the learnable temperature, not a stale float."""
        from flows.particle_conserving_flow import SigmoidTopK

        topk = SigmoidTopK(temperature=1.0, min_temperature=0.1)
        logits = torch.randn(4, 6)

        # Get output at temp=1.0
        out1 = topk(logits, 2, hard=False).detach().clone()

        # Change temperature
        topk.temperature = 3.0
        out2 = topk(logits, 2, hard=False).detach().clone()

        # Outputs should differ (higher temp = more uniform)
        assert not torch.allclose(out1, out2, atol=1e-4), (
            "Changing temperature had no effect on forward output — "
            "forward may be using a stale float instead of the property"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
