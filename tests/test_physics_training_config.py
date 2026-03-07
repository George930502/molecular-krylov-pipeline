"""Tests for PR 2.2b + 2.2d: PhysicsGuidedConfig defaults and factory reconciliation.

TDD RED phase: these tests validate that:
1. physics_weight default is non-zero (enables REINFORCE energy gradient)
2. convergence_threshold is raised to 0.35 (stricter mode-collapse detection)
3. Factory function defaults match PhysicsGuidedConfig defaults
4. entropy_weight default is 0.05 (non-zero entropy regularization)
5. Temperature annealing defaults: final_temperature=0.3, decay_epochs=400
"""

import pytest
import sys
import os
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flows.physics_guided_training import PhysicsGuidedConfig, create_physics_guided_trainer


class TestPhysicsWeightDefault:
    """physics_weight must be non-zero to enable the REINFORCE energy gradient."""

    def test_physics_weight_default_nonzero(self):
        """PhysicsGuidedConfig().physics_weight should be > 0.

        Without physics_weight, only the teacher loss (KL divergence from NQS)
        guides the flow. The physics loss provides a direct "low energy is better"
        signal via REINFORCE. A non-zero default enables this by default.
        """
        config = PhysicsGuidedConfig()
        assert config.physics_weight > 0, (
            f"physics_weight should be non-zero to enable REINFORCE energy gradient, "
            f"got {config.physics_weight}"
        )

    def test_physics_weight_default_value(self):
        """physics_weight default should be exactly 0.1."""
        config = PhysicsGuidedConfig()
        assert config.physics_weight == 0.1, (
            f"physics_weight should default to 0.1, got {config.physics_weight}"
        )


class TestConvergenceThreshold:
    """convergence_threshold controls mode-collapse detection sensitivity."""

    def test_convergence_threshold_raised(self):
        """convergence_threshold should be >= 0.35 for stricter mode-collapse detection.

        The convergence check is: unique_ratio < convergence_threshold.
        When unique_ratio drops below threshold, training stops (declares convergence).
        Raising the threshold means we stop earlier when diversity drops — i.e.,
        mode collapse is detected sooner. This prevents wasted epochs on a collapsed NF.
        """
        config = PhysicsGuidedConfig()
        assert config.convergence_threshold >= 0.35, (
            f"convergence_threshold should be >= 0.35 for stricter mode-collapse detection, "
            f"got {config.convergence_threshold}"
        )

    def test_convergence_threshold_exact_value(self):
        """convergence_threshold default should be exactly 0.35."""
        config = PhysicsGuidedConfig()
        assert config.convergence_threshold == 0.35, (
            f"convergence_threshold should default to 0.35, got {config.convergence_threshold}"
        )


class TestFactoryMatchesConfig:
    """Factory function defaults must match PhysicsGuidedConfig defaults."""

    def test_factory_teacher_weight_matches_config(self):
        """create_physics_guided_trainer teacher_weight default == PhysicsGuidedConfig default."""
        config_default = PhysicsGuidedConfig().teacher_weight
        sig = inspect.signature(create_physics_guided_trainer)
        factory_default = sig.parameters["teacher_weight"].default
        assert factory_default == config_default, (
            f"Factory teacher_weight={factory_default} != Config teacher_weight={config_default}"
        )

    def test_factory_physics_weight_matches_config(self):
        """create_physics_guided_trainer physics_weight default == PhysicsGuidedConfig default."""
        config_default = PhysicsGuidedConfig().physics_weight
        sig = inspect.signature(create_physics_guided_trainer)
        factory_default = sig.parameters["physics_weight"].default
        assert factory_default == config_default, (
            f"Factory physics_weight={factory_default} != Config physics_weight={config_default}"
        )

    def test_factory_entropy_weight_matches_config(self):
        """create_physics_guided_trainer entropy_weight default == PhysicsGuidedConfig default."""
        config_default = PhysicsGuidedConfig().entropy_weight
        sig = inspect.signature(create_physics_guided_trainer)
        factory_default = sig.parameters["entropy_weight"].default
        assert factory_default == config_default, (
            f"Factory entropy_weight={factory_default} != Config entropy_weight={config_default}"
        )


class TestEntropyWeightDefault:
    """entropy_weight should be non-zero for exploration regularization."""

    def test_entropy_weight_default(self):
        """PhysicsGuidedConfig().entropy_weight should be 0.05.

        Non-zero entropy weight prevents mode collapse by penalizing
        low-entropy flow distributions. 0.05 is a conservative default
        that provides regularization without dominating the loss.
        """
        config = PhysicsGuidedConfig()
        assert config.entropy_weight == 0.05, (
            f"entropy_weight should default to 0.05, got {config.entropy_weight}"
        )

    def test_entropy_weight_nonzero(self):
        """entropy_weight must be positive for exploration."""
        config = PhysicsGuidedConfig()
        assert config.entropy_weight > 0, (
            f"entropy_weight should be > 0 for exploration, got {config.entropy_weight}"
        )


class TestTemperatureAnnealingDefaults:
    """Temperature annealing should use slower schedule with higher floor."""

    def test_final_temperature_raised(self):
        """final_temperature should be 0.3 (higher floor prevents premature sharpening).

        With final_temperature=0.1, the NF sharpens too aggressively,
        losing diversity before training has converged. 0.3 maintains
        enough stochasticity for continued exploration.
        """
        config = PhysicsGuidedConfig()
        assert config.final_temperature == 0.3, (
            f"final_temperature should default to 0.3, got {config.final_temperature}"
        )

    def test_temperature_decay_epochs_raised(self):
        """temperature_decay_epochs should be 400 (slower decay schedule).

        With decay_epochs=200, temperature drops to final value halfway through
        training (500 epochs). With 400, the annealing covers 80% of training,
        giving more time for exploration before committing to sharpened sampling.
        """
        config = PhysicsGuidedConfig()
        assert config.temperature_decay_epochs == 400, (
            f"temperature_decay_epochs should default to 400, got {config.temperature_decay_epochs}"
        )

    def test_initial_temperature_unchanged(self):
        """initial_temperature should remain 1.0 (maximum exploration at start)."""
        config = PhysicsGuidedConfig()
        assert config.initial_temperature == 1.0, (
            f"initial_temperature should remain 1.0, got {config.initial_temperature}"
        )
