"""Tests for PR 2.1: Conditional NF Training.

ADR-001 spec: Make NF training conditional on system size.
- Systems with <= 20K valid configs: Direct-CI (skip_nf_training=True)
- Systems with > 20K valid configs: NF training enabled (skip_nf_training=False)
- User explicit override must be preserved.

TDD RED phase: these tests should FAIL before the implementation change.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import PipelineConfig


class TestConditionalNFSmallSystem:
    """Small systems (<= 1000 configs) should skip NF training."""

    def test_small_system_skips_nf(self):
        """LiH has 225 configs -- well below 20K threshold, should skip NF."""
        config = PipelineConfig()
        config.adapt_to_system_size(225, verbose=False)
        assert config.skip_nf_training is True, (
            f"Small system (225 configs) should skip NF training, "
            f"but skip_nf_training={config.skip_nf_training}"
        )


class TestConditionalNFMediumSystem:
    """Medium systems (1000-5000 configs) should skip NF training."""

    def test_medium_system_skips_nf(self):
        """NH3 has 3136 configs -- below 20K threshold, should skip NF."""
        config = PipelineConfig()
        config.adapt_to_system_size(3136, verbose=False)
        assert config.skip_nf_training is True, (
            f"Medium system (3136 configs) should skip NF training, "
            f"but skip_nf_training={config.skip_nf_training}"
        )


class TestConditionalNFLargeSystem:
    """Large systems (5000-20000 configs) should skip NF training."""

    def test_large_system_skips_nf(self):
        """N2/STO-3G has 14400 configs -- below 20K threshold, should skip NF."""
        config = PipelineConfig()
        config.adapt_to_system_size(14400, verbose=False)
        assert config.skip_nf_training is True, (
            f"Large system (14400 configs) should skip NF training, "
            f"but skip_nf_training={config.skip_nf_training}"
        )


class TestConditionalNFVeryLargeSystem:
    """Very large systems (> 20K configs) should enable NF training."""

    def test_very_large_system_enables_nf(self):
        """63504 configs exceeds 20K threshold -- NF training should be enabled."""
        config = PipelineConfig()
        config.adapt_to_system_size(63504, verbose=False)
        assert config.skip_nf_training is False, (
            f"Very large system (63504 configs) should enable NF training, "
            f"but skip_nf_training={config.skip_nf_training}"
        )


class TestConditionalNFUserOverride:
    """User explicit override must be preserved even for very large systems."""

    def test_user_override_preserved(self):
        """User explicitly sets skip_nf_training=True; even for >20K, should stay True."""
        config = PipelineConfig(skip_nf_training=True)
        # Mark that the user explicitly set this
        config._user_set_skip_nf = True
        config.adapt_to_system_size(63504, verbose=False)
        assert config.skip_nf_training is True, (
            f"User explicitly set skip_nf_training=True for 63504 configs, "
            f"but it was overridden to {config.skip_nf_training}"
        )

    def test_user_override_false_preserved(self):
        """User explicitly sets skip_nf_training=False for small system; should stay False."""
        config = PipelineConfig(skip_nf_training=False)
        config._user_set_skip_nf = True
        config.adapt_to_system_size(225, verbose=False)
        assert config.skip_nf_training is False, (
            f"User explicitly set skip_nf_training=False for 225 configs, "
            f"but it was overridden to {config.skip_nf_training}"
        )


class TestConditionalNFBoundary:
    """Boundary tests at exactly 20000 and 20001 configs."""

    def test_threshold_boundary_at_20000(self):
        """Exactly 20000 configs: at boundary, should skip NF (<=20000)."""
        config = PipelineConfig()
        config.adapt_to_system_size(20000, verbose=False)
        assert config.skip_nf_training is True, (
            f"Boundary system (20000 configs) should skip NF training, "
            f"but skip_nf_training={config.skip_nf_training}"
        )

    def test_threshold_boundary_at_20001(self):
        """20001 configs: just past boundary, should enable NF."""
        config = PipelineConfig()
        config.adapt_to_system_size(20001, verbose=False)
        assert config.skip_nf_training is False, (
            f"System with 20001 configs should enable NF training, "
            f"but skip_nf_training={config.skip_nf_training}"
        )


class TestConditionalNFVerboseOutput:
    """Verify verbose output messages for conditional NF."""

    def test_verbose_direct_ci_message(self, capsys):
        """Small system verbose output should mention Direct-CI."""
        config = PipelineConfig()
        config.adapt_to_system_size(225, verbose=True)
        captured = capsys.readouterr()
        assert "Direct-CI" in captured.out or "Direct-CI mode" in captured.out

    def test_verbose_nf_enabled_message(self, capsys):
        """Very large system verbose output should mention NF training enabled."""
        config = PipelineConfig()
        config.adapt_to_system_size(63504, verbose=True)
        captured = capsys.readouterr()
        assert (
            "NF training enabled" in captured.out
        ), f"Expected 'NF training enabled' in output, got: {captured.out}"


class TestConditionalNFIdempotent:
    """Verify adapt_to_system_size is idempotent (re-calling with same size is no-op)."""

    def test_idempotent_small(self):
        """Calling adapt_to_system_size twice with same size should be idempotent."""
        config = PipelineConfig()
        config.adapt_to_system_size(225, verbose=False)
        assert config.skip_nf_training is True
        config.adapt_to_system_size(225, verbose=False)
        assert config.skip_nf_training is True

    def test_idempotent_very_large(self):
        """Calling adapt_to_system_size twice with same size should be idempotent."""
        config = PipelineConfig()
        config.adapt_to_system_size(63504, verbose=False)
        assert config.skip_nf_training is False
        config.adapt_to_system_size(63504, verbose=False)
        assert config.skip_nf_training is False


class TestConditionalNFDefaultBehavior:
    """Verify PipelineConfig default skip_nf_training is False (not pre-set)."""

    def test_default_skip_nf_training_is_false(self):
        """Default PipelineConfig should have skip_nf_training=False."""
        config = PipelineConfig()
        assert config.skip_nf_training is False

    def test_no_user_set_flag_by_default(self):
        """Default PipelineConfig should NOT have _user_set_skip_nf attribute."""
        config = PipelineConfig()
        assert not hasattr(config, "_user_set_skip_nf")
