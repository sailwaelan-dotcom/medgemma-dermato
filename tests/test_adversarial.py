"""Automated CI test wrapping the deterministic Adversarial Red-Teaming suite."""

from scripts.run_adversarial_stress_test import run_stress_suite


def test_adversarial_red_team_suite_passes_all_barriers() -> None:
    # Run without ONNX weights in CI (deterministic unit/governance checks must pass 100%)
    summary = run_stress_suite(onnx_model_path=None)
    assert summary["total_tests"] >= 12
    assert summary["all_passed"] is True, f"Failed adversarial checks: {summary['results']}"
