"""Alert contract (FR-300).

Alert content is untrusted data, never instructions (FR-1101). Validation is
deterministic; parsing failures route to ``invalid_input`` (FR-801).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, model_validator

from .enums import Severity, SymptomType


class Alert(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_id: str
    dataset: str
    pipeline: str
    symptom_type: SymptomType
    observed_value: float | int | None = None
    expected_value: float | int | None = None
    observed_change_pct: float | None = None
    window_start: datetime
    window_end: datetime
    detected_at: datetime
    severity: Severity

    @model_validator(mode="after")
    def _validate_window(self) -> Alert:
        if self.window_end < self.window_start:
            raise ValueError("window_end must not precede window_start")
        return self
