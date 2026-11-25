"""Policy-as-Code CI/CD gate verification against versioned MSKCC reference snapshots."""

import json
import sys
from pathlib import Path

from medgemma_dermato.clinical_harness import evaluate_deployment_gate


def main() -> int:
    snap_path = Path("configs/reference_gate_snapshots.json")
    payload = json.loads(snap_path.read_text(encoding="utf-8"))
    failures = 0

    print(f"Verifying Barrier-6 DeploymentGate on: {payload['dataset']}\n")
    for item in payload["snapshots"]:
        name = item["name"]
        expected = item["expected_decision"]
        gate = evaluate_deployment_gate(item["metrics"])
        status = "OK" if gate.decision == expected else "DRIFT"
        if gate.decision != expected:
            failures += 1
        print(f"  [{status}] {name:36s} -> decision={gate.decision:12s} (expected={expected})")

    if failures:
        print(f"\nFAILED: {failures} policy gate snapshot(s) drifted from expected verdict.")
        return 1
    print("\nALL POLICY GATES VERIFIED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
