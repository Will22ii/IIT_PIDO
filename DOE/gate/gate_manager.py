# DOE/gate/gate_manager.py

from __future__ import annotations

from typing import Dict


class GateManager:
    """
    GateManager

    Responsibilities:
    - Observe Gate1 (Top-k stability) and Gate2 (Uncertainty)
    - Report pass/fail and gate-only stop condition

    This class is stateless and does not manage phase, ratios,
    or explorer-level termination.
    """

    # -------------------------------------------------
    # Main API
    # -------------------------------------------------

    def evaluate(
        self,
        *,
        gate1_result: Dict,
        gate2_result: Dict,
    ) -> Dict[str, bool]:
        """
        Returns gate-only decision dict consumed by AdditionalDOEOrchestrator.

        {
          "gate1_passed": bool,
          "gate2_passed": bool,
          "gate_stop": bool,
        }
        """

        g1_passed = bool(gate1_result.get("passed", False))
        g2_passed = bool(gate2_result.get("passed", False))

        return {
            "gate1_passed": g1_passed,
            "gate2_passed": g2_passed,
            "gate_stop": bool(g1_passed and g2_passed),
        }
