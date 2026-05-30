"""Loss Given Default model.

Phase 0 stub: returns a fixed LGD. Phase 2 will implement beta regression
and/or two-stage classification on LendingClub recoveries data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LGDModel:
    version: str = "lgd-stub-0.1.0"
    _hardcoded_lgd: float = 0.45  # ~industry default for unsecured

    def predict(self, features: dict[str, Any]) -> float:
        return self._hardcoded_lgd
