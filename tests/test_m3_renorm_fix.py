"""Tests for M3: Remove redundant re-normalization for AR flow in PhysicsGuidedFlowTrainer.

Bug description:
    _compute_flow_loss() re-normalizes log probs over the current batch:
        log_Z = torch.logsumexp(log_flow_probs_raw, dim=0)
        log_flow_probs = log_flow_probs_raw - log_Z
    For AutoregressiveFlowSampler, log_prob() already returns exact, fully-normalized
    log probabilities (autoregressive factorization guarantees sum-to-1 over ALL configs).
    The batch re-normalization inflates every config's probability by log(1/sum_batch),
    introducing a systematic upward bias.

Fix:
    When self.flow is an AutoregressiveFlowSampler, skip the logsumexp subtraction and
    use log_flow_probs_raw directly.  ParticleConservingFlowSampler retains the old path
    for backward compatibility.

Test structure:
    TestARFlowLogProbNormalized  — AR flow log_probs sum exactly to 1.0 over all configs
    TestFlowLossNoRenormForAR    — _compute_flow_loss does NOT re-normalize for AR flow
    TestFlowLossStillRenormsForPCF — _compute_flow_loss STILL re-normalizes for old flow
"""

import math
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Module-level imports — skip gracefully when AR flow not available
# ---------------------------------------------------------------------------

try:
    from flows.autoregressive_flow import AutoregressiveFlowSampler, AutoregressiveConfig

    _HAS_AR_FLOW = True
except ImportError:
    _HAS_AR_FLOW = False

try:
    from flows.particle_conserving_flow import ParticleConservingFlowSampler

    _HAS_PCF = True
except ImportError:
    _HAS_PCF = False

try:
    from flows.physics_guided_training import PhysicsGuidedConfig, PhysicsGuidedFlowTrainer

    _HAS_TRAINER = True
except ImportError:
    _HAS_TRAINER = False

pytestmark = pytest.mark.skipif(
    not (_HAS_AR_FLOW and _HAS_TRAINER),
    reason="autoregressive_flow or physics_guided_training not available",
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

_LIH_N_SITES = 12   # 6 spatial orbitals x 2 spin-orbitals
_LIH_N_ALPHA = 2
_LIH_N_BETA = 2


def _make_ar_flow(device: str = "cpu") -> "AutoregressiveFlowSampler":
    """Small AR flow for LiH (12-site, 2α+2β)."""
    cfg = AutoregressiveConfig(
        n_layers=2,
        n_heads=2,
        d_model=16,
        d_ff=32,
        dropout=0.0,
    )
    return AutoregressiveFlowSampler(
        num_sites=_LIH_N_SITES,
        n_alpha=_LIH_N_ALPHA,
        n_beta=_LIH_N_BETA,
        transformer_config=cfg,
    ).to(device)


def _make_pcf(device: str = "cpu") -> "ParticleConservingFlowSampler":
    """Small PCF for LiH."""
    return ParticleConservingFlowSampler(
        num_sites=_LIH_N_SITES,
        n_alpha=_LIH_N_ALPHA,
        n_beta=_LIH_N_BETA,
        hidden_dims=[32, 32],
    ).to(device)


def _make_dummy_nqs(device: str = "cpu") -> nn.Module:
    """Trivial NQS stub: returns log|ψ| ≈ 0 for any config.

    Must have at least one parameter so that AdamW does not raise
    "optimizer got an empty parameter list".
    """

    class _DummyNQS(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Tiny linear layer: exists only to satisfy the optimizer requirement.
            # Bias is set to zero so the output is always near-zero.
            self._dummy = nn.Linear(1, 1, bias=False)
            nn.init.zeros_(self._dummy.weight)

        def forward(self, configs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return torch.zeros(configs.shape[0], device=configs.device)

        def log_prob(self, configs: torch.Tensor) -> torch.Tensor:
            return torch.zeros(configs.shape[0], device=configs.device)

    return _DummyNQS().to(device)


def _make_dummy_hamiltonian() -> Any:
    """Minimal Hamiltonian stub sufficient for PhysicsGuidedFlowTrainer init."""
    ham = MagicMock()
    ham.n_sites = _LIH_N_SITES
    ham.n_alpha = _LIH_N_ALPHA
    ham.n_beta = _LIH_N_BETA
    # diagonal_element / get_connections not called in unit-tests below
    return ham


def _make_trainer(flow: nn.Module) -> "PhysicsGuidedFlowTrainer":
    """Instantiate a trainer with a trivial NQS and dummy Hamiltonian.

    Disable connection caching and essential-config injection so that the
    MagicMock Hamiltonian does not need to implement Hamiltonian methods.
    """
    config = PhysicsGuidedConfig(
        samples_per_batch=32,
        teacher_weight=1.0,
        physics_weight=0.1,
        entropy_weight=0.05,
        use_energy_baseline=False,
        use_connection_cache=False,       # avoids ConnectionCache(hamiltonian.num_sites)
        inject_essential_configs=False,   # avoids _generate_essential_configs()
    )
    return PhysicsGuidedFlowTrainer(
        flow=flow,
        nqs=_make_dummy_nqs(),
        hamiltonian=_make_dummy_hamiltonian(),
        config=config,
        device="cpu",
    )


def _enum_all_lih_configs() -> torch.Tensor:
    """Enumerate all C(6,2)*C(6,2) = 225 valid LiH spin-orbital configurations.

    Each row is a 12-bit binary vector (alpha | beta occupations).
    n_alpha = 2, n_beta = 2, n_orbitals = 6.
    """
    from itertools import combinations

    n_orb = _LIH_N_SITES // 2
    alpha_slots = list(combinations(range(n_orb), _LIH_N_ALPHA))
    beta_slots = list(combinations(range(n_orb), _LIH_N_BETA))

    configs = []
    for a in alpha_slots:
        for b in beta_slots:
            row = [0] * _LIH_N_SITES
            for i in a:
                row[i] = 1
            for j in b:
                row[n_orb + j] = 1
            configs.append(row)

    return torch.tensor(configs, dtype=torch.long)


# ---------------------------------------------------------------------------
# Test class 1: AR flow log_probs sum to 1
# ---------------------------------------------------------------------------


class TestARFlowLogProbNormalized:
    """AutoregressiveFlowSampler.log_prob() must already be fully normalized."""

    def test_ar_flow_log_prob_already_normalized(self):
        """exp(log_prob) summed over all valid configs should equal 1.0 within 1e-4.

        The autoregressive factorization:
            P(config) = prod_i P(o_i | o_1, ..., o_{i-1})
        Each conditional is a proper softmax, so the joint is a proper distribution
        over the finite config space.  Summing over ALL 225 LiH configs must give 1.
        """
        flow = _make_ar_flow()
        flow.eval()

        all_configs = _enum_all_lih_configs()  # (225, 12)

        with torch.no_grad():
            log_probs = flow.log_prob(all_configs)  # (225,)

        # Check shapes and finiteness
        assert log_probs.shape == (225,), f"Expected shape (225,), got {log_probs.shape}"
        assert torch.isfinite(log_probs).all(), "log_probs contain non-finite values"

        # All log_probs must be non-positive (log of a probability <= 1)
        assert (log_probs <= 0).all(), "log_prob > 0 implies probability > 1 — impossible"

        # Sum of probabilities must be 1 (normalized distribution)
        total_prob = log_probs.exp().sum().item()
        assert abs(total_prob - 1.0) < 1e-4, (
            f"AR flow probabilities do not sum to 1.0: sum={total_prob:.8f}. "
            "If this fails, the autoregressive factorization has a normalization bug."
        )

    def test_ar_flow_log_prob_non_uniform_after_gradient_step(self):
        """After one backward step on a random loss, probs should still sum to 1.

        This verifies that the normalization holds not just at init (where weights
        might cancel) but after parameter updates change the logit distribution.
        """
        flow = _make_ar_flow()

        all_configs = _enum_all_lih_configs()

        # One gradient step on a random objective
        optimizer = torch.optim.SGD(flow.parameters(), lr=1e-3)
        log_probs = flow.log_prob(all_configs)
        loss = -(log_probs * torch.rand_like(log_probs)).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Re-evaluate after the step
        flow.eval()
        with torch.no_grad():
            log_probs_after = flow.log_prob(all_configs)

        total_prob = log_probs_after.exp().sum().item()
        assert abs(total_prob - 1.0) < 1e-4, (
            f"AR flow probabilities no longer sum to 1.0 after gradient step: "
            f"sum={total_prob:.8f}"
        )


# ---------------------------------------------------------------------------
# Test class 2: _compute_flow_loss does NOT re-normalize for AR flow
# ---------------------------------------------------------------------------


class TestFlowLossNoRenormForAR:
    """_compute_flow_loss must skip the logsumexp subtraction for AR flows."""

    def _call_compute_flow_loss(
        self,
        trainer: "PhysicsGuidedFlowTrainer",
        unique_configs: torch.Tensor,
    ):
        """Helper: call _compute_flow_loss with synthetic tensors."""
        n = len(unique_configs)
        # nqs_probs: uniform distribution that sums to 1
        nqs_probs = torch.ones(n) / n
        # local_energies: constant (avoids physics-loss instability in unit test)
        local_energies = torch.full((n,), -7.5)
        # energy: scalar
        energy = torch.tensor(-7.5)
        # all_configs: same as unique for simplicity (batch_size = n)
        all_configs = unique_configs

        return trainer._compute_flow_loss(
            all_configs, unique_configs, nqs_probs, local_energies, energy
        )

    def test_flow_loss_no_renorm_for_ar(self):
        """When flow is AR, _compute_flow_loss must use raw log_probs (no logsumexp).

        Verification strategy:
        1. Record the raw log_probs returned by flow.log_prob().
        2. Call _compute_flow_loss() and capture what log_probs it actually uses
           by monkey-patching torch.logsumexp to raise if called on a (n,) tensor
           (the signature of the batch re-normalization call).
        3. If logsumexp is NOT called — the fix is in place.
        """
        flow = _make_ar_flow()
        trainer = _make_trainer(flow)

        all_configs = _enum_all_lih_configs()
        # Use a small subset to keep the test fast
        unique_configs = all_configs[:20]

        _logsumexp_called_on_batch = []

        _original_logsumexp = torch.logsumexp

        def _spy_logsumexp(input: torch.Tensor, dim: Any, **kwargs) -> torch.Tensor:
            # Only flag 1-D calls of size matching unique_configs (the re-norm call)
            if input.dim() == 1 and input.shape[0] == len(unique_configs):
                _logsumexp_called_on_batch.append(True)
            return _original_logsumexp(input, dim, **kwargs)

        with patch("torch.logsumexp", side_effect=_spy_logsumexp):
            loss, components = self._call_compute_flow_loss(trainer, unique_configs)

        assert not _logsumexp_called_on_batch, (
            "_compute_flow_loss called torch.logsumexp on a batch-sized tensor for the "
            "AR flow, which means re-normalization was NOT skipped. "
            "The fix should bypass logsumexp for AutoregressiveFlowSampler."
        )

        # Sanity: loss should be a finite scalar
        assert loss.dim() == 0, "flow loss should be a scalar"
        assert torch.isfinite(loss), f"flow loss is not finite: {loss.item()}"

    def test_flow_loss_ar_uses_raw_log_probs(self):
        """Teacher loss for AR flow must equal -sum(nqs_probs * log_flow_probs_raw).

        If re-normalization were applied, teacher_loss would be:
            -sum(nqs_probs * (log_flow_probs_raw - log_Z))
          = -sum(nqs_probs * log_flow_probs_raw) + log_Z  (since nqs_probs sums to 1)
        i.e. it would be shifted upward by log_Z = log(sum_batch_probs).

        For a properly normalized AR flow with batch < full space, log_Z < 0,
        so re-normalization would artificially increase teacher_loss.

        We verify: teacher_loss matches the raw computation exactly.
        """
        flow = _make_ar_flow()
        flow.eval()
        trainer = _make_trainer(flow)

        all_configs = _enum_all_lih_configs()
        unique_configs = all_configs[:20]
        n = len(unique_configs)

        nqs_probs = torch.ones(n) / n
        local_energies = torch.full((n,), -7.5)
        energy = torch.tensor(-7.5)

        with torch.no_grad():
            raw_log_probs = flow.log_prob(unique_configs)

        # Expected teacher_loss using raw (not re-normalized) probs
        expected_teacher = -(nqs_probs * raw_log_probs).sum().item()

        _, components = trainer._compute_flow_loss(
            unique_configs, unique_configs, nqs_probs, local_energies, energy
        )

        actual_teacher = components["teacher"].item()

        # The teacher loss must match the raw computation (no logsumexp bias)
        assert abs(actual_teacher - expected_teacher) < 1e-5, (
            f"Teacher loss mismatch for AR flow.\n"
            f"  Expected (raw):      {expected_teacher:.8f}\n"
            f"  Actual (from loss):  {actual_teacher:.8f}\n"
            f"  Difference:          {abs(actual_teacher - expected_teacher):.2e}\n"
            "If the difference equals |log_Z|, re-normalization was applied to AR flow."
        )

    def test_flow_loss_ar_bias_detection(self):
        """Directly show that re-normalization would introduce measurable bias.

        Math: for uniform nqs_probs (sum to 1) and batch subset of the full space,
            teacher_renorm - teacher_raw
              = -sum(p * (raw - log_Z)) + sum(p * raw)
              = sum(p) * log_Z
              = log_Z          (since sum(p) = 1)

        The batch probability mass is strictly < 1 (batch covers a strict subset of the
        full config space), so log_Z < 0 always, and teacher_renorm < teacher_raw.
        Re-normalization therefore artificially lowers teacher loss by |log_Z| nats —
        introducing a systematic downward bias.

        This test verifies:
        1. The algebraic identity holds numerically (bias = log_Z within 1e-5).
        2. log_Z < 0 for any proper batch (sanity-check on test geometry).
        """
        flow = _make_ar_flow()
        flow.eval()

        all_configs = _enum_all_lih_configs()
        # Use a strict subset: first 20 of 225 configs (any subset works)
        batch_configs = all_configs[:20]
        n = len(batch_configs)
        nqs_probs = torch.ones(n) / n

        with torch.no_grad():
            raw_log_probs = flow.log_prob(batch_configs)

        # What re-normalization would produce
        log_Z = torch.logsumexp(raw_log_probs, dim=0)
        renorm_log_probs = raw_log_probs - log_Z

        teacher_raw = -(nqs_probs * raw_log_probs).sum().item()
        teacher_renorm = -(nqs_probs * renorm_log_probs).sum().item()
        # Algebraic identity: bias = teacher_renorm - teacher_raw = log_Z
        bias = teacher_renorm - teacher_raw
        log_Z_val = log_Z.item()

        assert abs(bias - log_Z_val) < 1e-5, (
            f"Bias identity violated: bias={bias:.6f}, log_Z={log_Z_val:.6f}, "
            f"diff={abs(bias - log_Z_val):.2e}. "
            "Expected: teacher_renorm - teacher_raw == log_Z for uniform nqs_probs."
        )
        # log_Z < 0 always (batch sum of probs < 1 for any strict subset)
        assert log_Z_val < 0, (
            f"log_Z={log_Z_val:.4f} should be negative since batch covers only "
            f"{n}/225 of the config space.  If log_Z >= 0, the AR flow has a "
            "probability normalization bug."
        )


# ---------------------------------------------------------------------------
# Test class 3: _compute_flow_loss STILL re-normalizes for PCF (backward compat)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_PCF, reason="particle_conserving_flow not available")
class TestFlowLossStillRenormsForPCF:
    """For ParticleConservingFlowSampler, the batch re-normalization must be preserved."""

    def _call_compute_flow_loss(
        self,
        trainer: "PhysicsGuidedFlowTrainer",
        unique_configs: torch.Tensor,
    ):
        n = len(unique_configs)
        nqs_probs = torch.ones(n) / n
        local_energies = torch.full((n,), -7.5)
        energy = torch.tensor(-7.5)
        return trainer._compute_flow_loss(
            unique_configs, unique_configs, nqs_probs, local_energies, energy
        )

    def test_flow_loss_still_renorms_for_pcf(self):
        """When flow is PCF, _compute_flow_loss must still call logsumexp for re-norm.

        Strategy: spy on torch.logsumexp and verify it is called with a 1-D tensor
        of the same length as unique_configs — the re-normalization signature.
        """
        flow = _make_pcf()
        trainer = _make_trainer(flow)

        # Sample a small set of valid configs from the PCF itself
        with torch.no_grad():
            _, unique_configs = flow.sample(n_samples=16)
        unique_configs = unique_configs[:10]  # take at most 10

        _logsumexp_called = []

        _original_logsumexp = torch.logsumexp

        def _spy_logsumexp(input: torch.Tensor, dim: Any, **kwargs) -> torch.Tensor:
            if input.dim() == 1 and input.shape[0] == len(unique_configs):
                _logsumexp_called.append(True)
            return _original_logsumexp(input, dim, **kwargs)

        with patch("torch.logsumexp", side_effect=_spy_logsumexp):
            loss, _ = self._call_compute_flow_loss(trainer, unique_configs)

        assert _logsumexp_called, (
            "_compute_flow_loss did NOT call torch.logsumexp for PCF. "
            "The batch re-normalization was unexpectedly removed — "
            "this is a backward-compatibility regression."
        )

        assert torch.isfinite(loss), f"PCF flow loss is not finite: {loss.item()}"

    def test_pcf_teacher_loss_uses_renorm_probs(self):
        """Teacher loss for PCF must equal -sum(nqs_probs * renorm_log_probs).

        This confirms the re-normalization path is actually exercised (not just that
        logsumexp is called for some unrelated reason).
        """
        flow = _make_pcf()
        flow.eval()
        trainer = _make_trainer(flow)

        with torch.no_grad():
            _, unique_configs = flow.sample(n_samples=32)
        unique_configs = unique_configs[:10]
        n = len(unique_configs)
        nqs_probs = torch.ones(n) / n

        with torch.no_grad():
            raw_log_probs = flow.log_prob(unique_configs)
            log_Z = torch.logsumexp(raw_log_probs, dim=0)
            renorm_log_probs = raw_log_probs - log_Z

        expected_teacher = -(nqs_probs * renorm_log_probs).sum().item()

        local_energies = torch.full((n,), -7.5)
        energy = torch.tensor(-7.5)

        _, components = trainer._compute_flow_loss(
            unique_configs, unique_configs, nqs_probs, local_energies, energy
        )

        actual_teacher = components["teacher"].item()

        assert abs(actual_teacher - expected_teacher) < 1e-5, (
            f"Teacher loss for PCF does not match re-normalized computation.\n"
            f"  Expected (renorm): {expected_teacher:.8f}\n"
            f"  Actual:            {actual_teacher:.8f}\n"
            "The re-normalization path for PCF may have been accidentally removed."
        )
