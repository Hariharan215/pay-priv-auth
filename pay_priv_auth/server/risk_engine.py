"""Toy risk engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskEngine:
    th_frictionless: float = 0.85
    th_otp: float = 0.75

    def decide_frictionless(self, score: float, rba_low_risk: bool) -> str:
        if score >= self.th_frictionless and rba_low_risk:
            return "approve_frictionless"
        return "step_up"

    def decide_stepup(self, score_otp: float, otp_valid: bool) -> str:
        if score_otp >= self.th_otp and otp_valid:
            return "approve"
        return "deny"
