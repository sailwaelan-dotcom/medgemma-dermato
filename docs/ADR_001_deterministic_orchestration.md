# ADR-001: Bounded Deterministic Clinical Orchestration vs. Unconstrained LLM Agent Loops

- **Status:** Accepted
- **Role Perspective:** Google Cloud Professional Agentic Architect (PAA)
- **Component:** `ClinicalTriageOrchestrator` (`src/medgemma_dermato/agent.py`) & `LastMileClinicalHarness` (`src/medgemma_dermato/clinical_harness.py`)

## Context

In clinical triage (dermoscopy pre-triage), an application receives a skin lesion image alongside structured/unstructured patient metadata and must assign an actionable pathway (`GREEN` / `AMBER` / `RED`) under strict patient-safety constraints (`>= 95%` malignant sensitivity, `>= 95%` melanoma sensitivity, and zero autonomous diagnostic claims under EU MDR).

When designing this workflow from an Agentic Architect perspective, two orchestration paradigms are possible:
1. **Autonomous LLM ReAct / Function-Calling Loop:** Give the VLM/LLM dynamic control over which tools to invoke, whether to run quality checks, and when to stop.
2. **Bounded Deterministic State Machine (`ClinicalTriageOrchestrator`):** Wrap probabilistic models (`ConvNeXt-Tiny ONNX` + `MedGemma 1.5 4B Q8_0`) inside a fixed, auditable 6-step pipeline where safety barriers and clinical red flags are enforced deterministically in code.

## Decision

We explicitly **reject autonomous LLM tool-calling loops on the critical clinical triage path** and enforce a **deterministic 6-step orchestrator** (`ClinicalTriageOrchestrator`) paired with an immutable 6-barrier safety harness (`LastMileClinicalHarness`).

### Core Architectural Guarantees

1. **Zero Skipped Safety Barriers:** An LLM planner can forget to invoke preflight QA, skip TTA instability checks, or bypass clinical red flags (ABCDE Evolution / bleeding) under token pressure or prompt perturbation. Here, steps 1 → 6 execute unconditionally, with conditional branching (`dermoscopic_roi_crop_recheck`, `adaptive_vlm_router`) governed strictly by calibrated numerical thresholds.
2. **Zero Anchoring Bias in Longitudinal Follow-Up:** As enforced by `src/medgemma_dermato/db_model_bridge.py`, prior SQLite verdicts (`LongitudinalPriorDTO`) are **never** injected into MedGemma's prompt. The VLM evaluates the current lesion blindly, and longitudinal escalation (`LONGITUDINAL_BRIDGE_EVOLUTION` if `Δ risk >= +10%`) is adjudicated deterministically post-inference.
3. **Contradiction & Regulatory Isolation:** If MedGemma emits reassuring text (`"pas d'argument majeur de malignité"`) alongside a high calibrated probability, `LastMileClinicalHarness` deterministically blocks `GREEN` and strips the misleading text (catching **57 contradictions including 15 real cancers** in our local RTX 4080 `Q8_0` benchmark).
4. **Reproducible Cryptographic Auditability:** Every run emits a fixed 6-step latency trace + SHA-256 `AuditTrail` without stochastic tool-order variance.

## Consequences

- **Positive:** 100% deterministic guardrail enforcement, `13/13` adversarial stress-test pass rate, zero prompt-loop latency blowup, and clear separation between ML inference and safety policy.
- **Where True Agentic Autonomy Belongs:** Multi-step LLM autonomy is intentionally reserved for **non-critical surrounding workflows** (e.g., drafting patient education summaries, pre-filling administrative insurance/referral letters, or triaging incoming clinic inbox messages), never for overriding clinical risk routing.
