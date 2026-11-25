"""Unit tests for SQLite LocalCaseStore and longitudinal comparator."""

from pathlib import Path

from medgemma_dermato.agent import AgentConfig, DermoscopyTriageAgent
from medgemma_dermato.clinical_harness import ClinicalContext
from medgemma_dermato.storage import LocalCaseStore
from tests.test_agent import _MockSystem1, _SpyVLM, _textured_image


def test_sqlite_case_store_crud_and_longitudinal_comparison(tmp_path: Path) -> None:
    store = LocalCaseStore(tmp_path / "cases.db")
    img = _textured_image(512)

    agent_low = DermoscopyTriageAgent(
        system1=_MockSystem1(p_risk=0.03, p_mel=0.01),
        vlm_client=_SpyVLM(p_risk=0.04),
        agent_cfg=AgentConfig(adaptive_routing_enabled=True),
    )
    res0 = agent_low.run(img, case_id="CASE-J0")
    res0.assessment.audit.timestamp_utc = "2026-01-01T10:00:00+00:00"
    store.save_case(res0, patient_ref="PAT-01", lesion_label="Dos")

    agent_high = DermoscopyTriageAgent(
        system1=_MockSystem1(p_risk=0.55, p_mel=0.45),
        vlm_client=_SpyVLM(p_risk=0.72),
    )
    res1 = agent_high.run(
        img,
        clinical_context=ClinicalContext(lesion_evolving=True),
        case_id="CASE-J90",
    )
    res1.assessment.audit.timestamp_utc = "2026-04-01T10:00:00+00:00"
    store.save_case(res1, patient_ref="PAT-01", lesion_label="Dos")

    listed = store.list_cases(patient_ref="PAT-01")
    assert len(listed) == 2
    assert listed[0]["case_id"] == "CASE-J90"

    updated = store.update_notes("CASE-J90", "Biopsie programmee")
    assert updated is not None
    assert updated["clinician_notes"] == "Biopsie programmee"

    cmp_res = store.compare_cases("CASE-J0", "CASE-J90")
    assert cmp_res.days_apart == 90.0
    assert cmp_res.zone_escalated is True
    assert cmp_res.evolution_alert is True
    assert cmp_res.delta_risk > 0.25

    assert store.delete_case("CASE-J0") is True
    assert store.get_case("CASE-J0") is None
