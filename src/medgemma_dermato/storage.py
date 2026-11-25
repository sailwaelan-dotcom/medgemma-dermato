"""Local offline SQLite case history and longitudinal lesion comparator (Critère E)."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from medgemma_dermato.agent import AgentRunResult
from medgemma_dermato.db_model_bridge import (
    hydrate_and_validate_from_storage,
    validate_for_storage,
)

ZONE_ORDER: dict[str, int] = {
    "GREEN_LOW_RISK_SURVEILLANCE": 0,
    "AMBER_UNCERTAIN_TELEEXPERTISE": 1,
    "RED_PRIORITY_DERMATOLOGY": 2,
}


class CaseComparison(BaseModel):
    """Longitudinal comparison between baseline (J0) and follow-up (J+N) exams."""

    base_case_id: str
    followup_case_id: str
    patient_ref: str | None = None
    lesion_label: str | None = None
    days_apart: float
    base_risk: float
    followup_risk: float
    delta_risk: float
    delta_sys1: float
    delta_medgemma: float
    base_zone: str
    followup_zone: str
    zone_escalated: bool
    evolution_alert: bool
    alert_reason_fr: str


class LocalCaseStore:
    """Thread-safe local SQLite store for offline Edge dermatology cases."""

    def __init__(self, db_path: Path | str = Path("data/local_cases.db")) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    created_utc TEXT NOT NULL,
                    patient_ref TEXT,
                    lesion_label TEXT,
                    age_years INTEGER,
                    sex TEXT,
                    anatom_site TEXT,
                    triage_zone TEXT NOT NULL,
                    routing_decision TEXT NOT NULL,
                    calibrated_risk REAL NOT NULL,
                    p_sys1_cal REAL NOT NULL,
                    p_medgemma_cal REAL NOT NULL,
                    sys1_top_entity TEXT,
                    tta_risk_std REAL,
                    image_sha256 TEXT,
                    clinician_notes TEXT NOT NULL DEFAULT '',
                    original_b64 TEXT,
                    heatmap_b64 TEXT,
                    roi_b64 TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save_case(
        self,
        run_res: AgentRunResult,
        patient_ref: str | None = None,
        lesion_label: str | None = None,
        age_years: int | None = None,
        sex: str | None = None,
        anatom_site: str | None = None,
        original_b64: str | None = None,
        heatmap_b64: str | None = None,
        roi_b64: str | None = None,
        clinician_notes: str = "",
    ) -> dict[str, Any]:
        """Persist one AgentRunResult and visual artifacts into local SQLite."""
        a = run_res.assessment
        created_utc = a.audit.timestamp_utc or datetime.now(UTC).isoformat()
        payload = {
            "case_id": run_res.case_id,
            "created_utc": created_utc,
            "patient_ref": patient_ref,
            "lesion_label": lesion_label,
            "age_years": age_years,
            "sex": sex,
            "anatom_site": anatom_site,
            "routing_decision": run_res.routing_decision,
            "vlm_invoked": run_res.vlm_invoked,
            "roi_crop_applied": run_res.roi_crop_applied,
            "system2_logprobs_used": run_res.system2_logprobs_used,
            "system2_latency_ms": run_res.system2_latency_ms,
            "total_latency_ms": run_res.total_latency_ms,
            "preflight": run_res.preflight.model_dump(),
            "system1": run_res.system1.model_dump(),
            "assessment": a.model_dump(),
            "action_plan": run_res.action_plan.model_dump(),
            "trace": [s.model_dump() for s in run_res.trace],
        }
        envelope = validate_for_storage(payload)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cases (
                    case_id, created_utc, patient_ref, lesion_label, age_years, sex,
                    anatom_site, triage_zone, routing_decision, calibrated_risk,
                    p_sys1_cal, p_medgemma_cal, sys1_top_entity, tta_risk_std,
                    image_sha256, clinician_notes, original_b64, heatmap_b64, roi_b64,
                    payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope.case_id,
                    envelope.created_utc,
                    envelope.payload.get("patient_ref"),
                    envelope.payload.get("lesion_label"),
                    age_years,
                    sex,
                    envelope.payload.get("anatom_site"),
                    envelope.triage_zone,
                    envelope.routing_decision,
                    envelope.calibrated_risk_score,
                    float(a.p_sys1_cal),
                    float(a.p_medgemma_cal),
                    a.sys1_top_entity,
                    a.tta_risk_std,
                    a.audit.image_sha256,
                    clinician_notes,
                    original_b64,
                    heatmap_b64,
                    roi_b64,
                    envelope.model_dump_json(),
                ),
            )
            conn.commit()
        return self.get_case(run_res.case_id) or {}

    def list_cases(
        self,
        zone: str | None = None,
        patient_ref: str | None = None,
        query: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List stored cases ordered newest first with optional zone/patient/search filters."""
        zone_param = zone.strip() if zone and zone.strip() else None
        patient_like = (
            f"%{patient_ref.strip().lower()}%" if patient_ref and patient_ref.strip() else None
        )
        query_like = f"%{query.strip().lower()}%" if query and query.strip() else None
        sql = """
            SELECT case_id, created_utc, patient_ref, lesion_label, age_years, sex, anatom_site,
                   triage_zone, routing_decision, calibrated_risk, p_sys1_cal, p_medgemma_cal,
                   sys1_top_entity, tta_risk_std, image_sha256, clinician_notes, original_b64,
                   heatmap_b64, roi_b64
            FROM cases
            WHERE (? IS NULL OR triage_zone = ?)
              AND (? IS NULL OR LOWER(COALESCE(patient_ref, '')) LIKE ?)
              AND (? IS NULL OR (
                   LOWER(case_id) LIKE ?
                OR LOWER(COALESCE(patient_ref, '')) LIKE ?
                OR LOWER(COALESCE(lesion_label, '')) LIKE ?
                OR LOWER(COALESCE(anatom_site, '')) LIKE ?
                OR LOWER(COALESCE(sys1_top_entity, '')) LIKE ?
              ))
            ORDER BY created_utc DESC
            LIMIT ?
        """
        params = (
            zone_param,
            zone_param,
            patient_like,
            patient_like,
            query_like,
            query_like,
            query_like,
            query_like,
            query_like,
            query_like,
            int(limit),
        )
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_case(self, case_id: str) -> dict[str, Any] | None:
        """Retrieve a full case record and verify its SHA-256 storage integrity."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        payload = hydrate_and_validate_from_storage(d.pop("payload_json"))
        payload.update(
            {
                "created_utc": d["created_utc"],
                "patient_ref": d["patient_ref"],
                "lesion_label": d["lesion_label"],
                "age_years": d["age_years"],
                "sex": d["sex"],
                "anatom_site": d["anatom_site"],
                "clinician_notes": d["clinician_notes"],
                "original_b64": d["original_b64"],
                "heatmap_b64": d["heatmap_b64"],
                "roi_b64": d["roi_b64"],
            }
        )
        return payload

    def update_notes(self, case_id: str, notes: str) -> dict[str, Any] | None:
        """Update clinician notes on an existing case."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE cases SET clinician_notes = ? WHERE case_id = ?",
                (notes, case_id),
            )
            conn.commit()
            if cur.rowcount == 0:
                return None
        return self.get_case(case_id)

    def delete_case(self, case_id: str) -> bool:
        """Delete one case cleanly (GDPR / local cabinet cleanup)."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM cases WHERE case_id = ?", (case_id,))
            conn.commit()
            return cur.rowcount > 0

    def compare_cases(self, base_id: str, followup_id: str) -> CaseComparison:
        """Compare J0 baseline and J+N follow-up cases for longitudinal evolution (Critère E)."""
        base = self.get_case(base_id)
        follow = self.get_case(followup_id)
        if base is None:
            raise KeyError(f"Base case not found: {base_id}")
        if follow is None:
            raise KeyError(f"Follow-up case not found: {followup_id}")

        t0 = datetime.fromisoformat(base["created_utc"])
        t1 = datetime.fromisoformat(follow["created_utc"])
        days_apart = round((t1 - t0).total_seconds() / 86400.0, 2)

        ba = base["assessment"]
        fa = follow["assessment"]
        delta_risk = round(
            float(fa["calibrated_risk_score"]) - float(ba["calibrated_risk_score"]), 4
        )
        delta_s1 = round(float(fa["p_sys1_cal"]) - float(ba["p_sys1_cal"]), 4)
        delta_med = round(float(fa["p_medgemma_cal"]) - float(ba["p_medgemma_cal"]), 4)

        b_zone = str(ba["triage_zone"])
        f_zone = str(fa["triage_zone"])
        zone_escalated = ZONE_ORDER.get(f_zone, 0) > ZONE_ORDER.get(b_zone, 0)
        evolution_alert = zone_escalated or delta_risk >= 0.10

        if zone_escalated and delta_risk >= 0.10:
            reason = (
                f"Alerte Évolution (Critère E) : escalade {b_zone} → {f_zone} "
                f"avec hausse de risque +{delta_risk:.1%}."
            )
        elif zone_escalated:
            reason = f"Alerte Évolution (Critère E) : changement de zone {b_zone} → {f_zone}."
        elif delta_risk >= 0.10:
            reason = (
                f"Alerte Évolution (Critère E) : progression du score calibré de +{delta_risk:.1%}."
            )
        else:
            reason = (
                f"Stabilité longitudinale relative (Δ risque = {delta_risk:+.1%}, "
                f"zone inchangée/non aggravée)."
            )

        return CaseComparison(
            base_case_id=base_id,
            followup_case_id=followup_id,
            patient_ref=follow.get("patient_ref") or base.get("patient_ref"),
            lesion_label=follow.get("lesion_label") or base.get("lesion_label"),
            days_apart=days_apart,
            base_risk=float(ba["calibrated_risk_score"]),
            followup_risk=float(fa["calibrated_risk_score"]),
            delta_risk=delta_risk,
            delta_sys1=delta_s1,
            delta_medgemma=delta_med,
            base_zone=b_zone,
            followup_zone=f_zone,
            zone_escalated=zone_escalated,
            evolution_alert=evolution_alert,
            alert_reason_fr=reason,
        )
