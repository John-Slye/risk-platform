"""Market risk wrapper — stub for Phase 0.

Phase 4 ports in Project 1's var_methods.py, backtesting.py, stress.py,
decomposition.py, evt.py and wires them to this class interface.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MarketRisk:
    """Stub VaR/ES API. Real implementation in Phase 4."""
    version: str = "market-stub-0.1.0"

    def var(self, method: str = "historical", alpha: float = 0.05) -> dict:
        method = method.lower()
        if method not in {"historical", "parametric_normal", "parametric_t",
                          "monte_carlo_normal", "monte_carlo_t", "fhs"}:
            raise ValueError(f"unknown method {method!r}")
        # Stub returns the numbers from Project 1's empirical run.
        ref = {
            ("historical", 0.05): 0.0099,    ("historical", 0.01): 0.0180,
            ("parametric_normal", 0.05): 0.0108, ("parametric_normal", 0.01): 0.0154,
            ("parametric_t", 0.05): 0.0092,  ("parametric_t", 0.01): 0.0173,
            ("monte_carlo_normal", 0.05): 0.0109, ("monte_carlo_normal", 0.01): 0.0156,
            ("monte_carlo_t", 0.05): 0.0147, ("monte_carlo_t", 0.01): 0.0267,
            ("fhs", 0.05): 0.0074,           ("fhs", 0.01): 0.0123,
        }
        var = ref.get((method, alpha), 0.015)
        return {"VaR": var, "method": method, "alpha": alpha, "model": self.version}

    def es(self, method: str = "historical", alpha: float = 0.05) -> dict:
        # Stub: ES = VaR * 1.5 (fat-tail multiplier)
        v = self.var(method, alpha)
        return {"ES": v["VaR"] * 1.5, "method": method, "alpha": alpha,
                "model": self.version}

    def stress(self, scenario: str = "2020_covid") -> dict:
        # Numbers from Project 1's stress run.
        ref = {
            "2020_covid": {"cum_loss": 0.234, "worst_day": 0.0699, "ann_vol": 0.471},
            "2008_gfc":   {"cum_loss": 0.228, "worst_day": 0.0516, "ann_vol": 0.392},
            "2022_rates": {"cum_loss": 0.092, "worst_day": 0.0271, "ann_vol": 0.138},
        }
        if scenario not in ref:
            raise ValueError(f"unknown scenario {scenario!r}; one of {list(ref)}")
        return {"scenario": scenario, **ref[scenario], "model": self.version}
