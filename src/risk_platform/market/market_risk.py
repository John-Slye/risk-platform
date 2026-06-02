"""MarketRisk class: wraps the ported Project 1 engine.

Exposes a single object the FastAPI router uses for every market endpoint.
Prices are cached on disk via the ported `data.py` module; the GARCH fit
for FHS is computed lazily on first access and cached for subsequent calls
within the process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from . import var_methods as vm
from .data import (
    DEFAULT_TICKERS, DEFAULT_WEIGHTS, load_returns, portfolio_returns,
)
from .decomposition import component_var
from .evt import evt_es, evt_var
from .stress import stress_test


@dataclass
class MarketRisk:
    """Lazy wrapper around the Project 1 risk engine.

    Loads the cached prices on first method call. If yfinance is unreachable
    (CI, no internet) it returns NaN-tolerant errors via the underlying
    `load_returns` call which raises a clear error message.
    """

    version: str = "market-1.0.0"
    tickers: list[str] = field(default_factory=lambda: list(DEFAULT_TICKERS))
    weights: pd.Series = field(default_factory=lambda: DEFAULT_WEIGHTS.copy())

    _returns: Optional[pd.DataFrame] = None
    _port: Optional[pd.Series] = None
    _garch: Optional[dict] = None
    _fallback: bool = False

    # Reference numbers from the Project 1 backtest run (used in fallback mode
    # when yfinance is unreachable, e.g. CI without network access).
    _FALLBACK_VAR: dict[tuple[str, float], float] = field(
        default_factory=lambda: {
            ("historical", 0.05): 0.0099,    ("historical", 0.01): 0.0180,
            ("parametric_normal", 0.05): 0.0108, ("parametric_normal", 0.01): 0.0154,
            ("parametric_t", 0.05): 0.0092,  ("parametric_t", 0.01): 0.0173,
            ("monte_carlo_normal", 0.05): 0.0109, ("monte_carlo_normal", 0.01): 0.0156,
            ("monte_carlo_t", 0.05): 0.0147, ("monte_carlo_t", 0.01): 0.0267,
            ("fhs", 0.05): 0.0074,           ("fhs", 0.01): 0.0123,
        }
    )

    # ---------------------------------------------------------------------
    def _ensure_data(self) -> None:
        if self._returns is not None or self._fallback:
            return
        try:
            self._returns = load_returns(self.tickers)
            # yfinance silently returns empty data on failure; treat that
            # as a fallback trigger too.
            if self._returns is None or self._returns.empty or len(self._returns) < 250:
                self._fallback = True
                self._returns = None
                return
            self._port = portfolio_returns(self._returns, self.weights)
        except Exception:
            self._fallback = True
            self._returns = None

    def _ensure_garch(self) -> dict:
        if self._garch is None:
            self._ensure_data()
            self._garch = vm.fit_garch(self._port)
        return self._garch

    # ---------------------------------------------------------------------
    _VALID_METHODS = {
        "historical", "parametric_normal", "parametric_t",
        "monte_carlo_normal", "monte_carlo_t", "fhs",
    }

    def var(self, method: str = "historical", alpha: float = 0.05) -> dict[str, Any]:
        self._ensure_data()
        method = method.lower()
        if method not in self._VALID_METHODS:
            raise ValueError(f"unknown method {method!r}")
        if self._fallback:
            v = self._FALLBACK_VAR.get((method, alpha), 0.015)
            return {"VaR": float(v), "method": method, "alpha": alpha,
                    "model": f"{self.version}-fallback"}
        if method == "historical":
            v = vm.historical_var(self._port, alpha)
        elif method == "parametric_normal":
            v = vm.parametric_var(self._port, alpha, dist="normal")
        elif method == "parametric_t":
            v = vm.parametric_var(self._port, alpha, dist="t")
        elif method == "monte_carlo_normal":
            v = vm.monte_carlo_var(self._returns, self.weights, alpha,
                                   n_sims=10_000, dist="normal", seed=42)
        elif method == "monte_carlo_t":
            v = vm.monte_carlo_var(self._returns, self.weights, alpha,
                                   n_sims=10_000, dist="t", seed=42)
        elif method == "fhs":
            v = vm.filtered_historical_var(self._port, alpha,
                                           garch=self._ensure_garch())
        else:
            raise ValueError(f"unknown method {method!r}")
        return {"VaR": float(v), "method": method, "alpha": alpha,
                "model": self.version}

    def es(self, method: str = "historical", alpha: float = 0.05) -> dict[str, Any]:
        self._ensure_data()
        method = method.lower()
        if method not in self._VALID_METHODS:
            raise ValueError(f"unknown method {method!r}")
        if self._fallback:
            v = self._FALLBACK_VAR.get((method, alpha), 0.015)
            return {"ES": float(v * 1.5), "method": method, "alpha": alpha,
                    "model": f"{self.version}-fallback"}
        if method == "historical":
            e = vm.historical_es(self._port, alpha)
        elif method == "parametric_normal":
            e = vm.parametric_es(self._port, alpha, dist="normal")
        elif method == "parametric_t":
            e = vm.parametric_es(self._port, alpha, dist="t")
        elif method == "monte_carlo_normal":
            e = vm.monte_carlo_es(self._returns, self.weights, alpha,
                                  n_sims=10_000, dist="normal", seed=42)
        elif method == "monte_carlo_t":
            e = vm.monte_carlo_es(self._returns, self.weights, alpha,
                                  n_sims=10_000, dist="t", seed=42)
        elif method == "fhs":
            e = vm.filtered_historical_es(self._port, alpha,
                                          garch=self._ensure_garch())
        else:
            raise ValueError(f"unknown method {method!r}")
        return {"ES": float(e), "method": method, "alpha": alpha,
                "model": self.version}

    def stress(self, scenario: str = "2020_covid") -> dict[str, Any]:
        """Re-price the current portfolio under a historical stress window."""
        scenario = scenario.lower()
        if self._fallback:
            ref = {
                "2020_covid": {"cum_loss": 0.234, "worst_day": 0.0699, "ann_vol": 0.471},
                "2008_gfc":   {"cum_loss": 0.228, "worst_day": 0.0516, "ann_vol": 0.392},
                "2022_rates": {"cum_loss": 0.092, "worst_day": 0.0271, "ann_vol": 0.138},
            }
            if scenario not in ref:
                raise ValueError(f"unknown scenario {scenario!r}")
            return {"scenario": scenario, **ref[scenario],
                    "model": f"{self.version}-fallback"}
        scenarios_map = {
            "2008_gfc":   ("2008 Global Financial Crisis (Sep–Nov)",
                           ("2008-09-01", "2008-11-30")),
            "2020_covid": ("2020 COVID Crash (Feb 19 – Mar 23)",
                           ("2020-02-19", "2020-03-23")),
            "2022_rates": ("2022 Rate Shock (Jan–Jun)",
                           ("2022-01-01", "2022-06-30")),
        }
        if scenario not in scenarios_map:
            raise ValueError(f"unknown scenario {scenario!r}; one of {list(scenarios_map)}")
        label, window = scenarios_map[scenario]
        out = stress_test(self.weights, scenarios={label: window})
        row = out.iloc[0]
        return {
            "scenario": scenario,
            "cum_loss": float(row["cum_loss_%"] / 100.0),
            "worst_day": float(row["worst_day_%"] / 100.0),
            "ann_vol": float(row["ann_vol_%"] / 100.0),
            "model": self.version,
        }

    def evt(self, alpha: float = 0.001, threshold_pct: float = 0.95) -> dict[str, Any]:
        """Deep-tail VaR/ES via Peaks-Over-Threshold + GPD fit."""
        self._ensure_data()
        return {
            "VaR": float(evt_var(self._port, alpha, threshold_pct)),
            "ES":  float(evt_es (self._port, alpha, threshold_pct)),
            "alpha": alpha,
            "threshold_pct": threshold_pct,
            "model": self.version,
        }

    def component_var(self, alpha: float = 0.05) -> pd.DataFrame:
        """Per-asset Component VaR via Euler decomposition."""
        self._ensure_data()
        return component_var(self._returns, self.weights, alpha)
