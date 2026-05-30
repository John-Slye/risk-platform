"""Portfolio credit risk: Vasicek + copulas + Credit VaR.

Phase 0 ships stubs. Phase 3 replaces with real Vasicek ASRF, Gaussian +
Student-t copula Monte Carlo, Credit VaR, Economic Capital, concentration.
"""

from .vasicek import vasicek_loss_distribution
from .copula import simulate_portfolio_losses

__all__ = ["vasicek_loss_distribution", "simulate_portfolio_losses"]
