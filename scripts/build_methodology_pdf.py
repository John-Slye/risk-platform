"""Build the 2-page Model Methodology PDF.

Run from project root:
    uv run python scripts/build_methodology_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "Model_Methodology.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="HeroTitle", parent=styles["Title"], fontSize=18, leading=22, spaceAfter=4))
styles.add(ParagraphStyle(name="Sub",       parent=styles["Title"], fontSize=10, leading=13, textColor=colors.HexColor("#444"), spaceAfter=14))
styles.add(ParagraphStyle(name="H1",        parent=styles["Heading1"], fontSize=13, leading=16, spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle(name="H2",        parent=styles["Heading2"], fontSize=11, leading=14, spaceBefore=6,  spaceAfter=3))
styles.add(ParagraphStyle(name="Body",      parent=styles["Normal"],   fontSize=9.5, leading=12.5, alignment=TA_JUSTIFY, spaceAfter=5))


def H1(s): return Paragraph(s, styles["H1"])
def H2(s): return Paragraph(s, styles["H2"])
def P(s):  return Paragraph(s, styles["Body"])


def tbl(data, col_widths=None):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e8eef5")),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#999")),
        ("ALIGN", (1,0), (-1,-1), "RIGHT"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f7f7f7")]),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
    ]))
    return t


story = []

# Title
story += [
    Paragraph("Integrated Credit &amp; Market Risk Analytics Platform", styles["HeroTitle"]),
    Paragraph("Model Methodology - October 2026", styles["Sub"]),
]

# Page 1: Data + PD
story += [
    H1("1. Data"),
    P("Credit risk: LendingClub Accepted Loans 2007-2018 (~2.26M loans). "
      "Filtered to terminal-status loans only (Fully Paid + Charged Off + Default) "
      "and 2014-2018 vintages, yielding 1,117,393 modeling observations. "
      "Time-based split: train 2014-2016 (891K), validate 2017 (169K), test 2018 (56K). "
      "Target = 1 if loan_status in {Charged Off, Default}, else 0."),
    P("Market risk: 9 ETFs (SPY, QQQ, IWM, EFA, TLT, HYG, GLD, USO, UUP) over 2014-2026 "
      "(3,118 daily observations) from yfinance. Extended history to 2007 for the 2008 GFC stress scenario."),

    H1("2. PD models"),
    H2("Scorecard (logistic regression with WOE/IV)"),
    P("Features binned via <i>optbinning.BinningProcess</i> with monotonic-trend enforcement "
      "and IV selection (min IV = 0.02). 12 of 21 candidate features survived. "
      "Logistic regression fit via <i>statsmodels.Logit</i> on WOE-transformed features "
      "(p-values required for adverse-action reporting). PDO scaling: base score 600 at "
      "50:1 odds, PDO = 20 (industry standard)."),

    H2("XGBoost benchmark"),
    P("<i>XGBClassifier</i> with native categorical support, early stopping on validation set. "
      "Hyperparameters: max_depth=6, learning_rate=0.05, subsample=0.85, min_child_weight=50, "
      "reg_lambda=1.0. Frozen train-time categorical vocabulary at inference to prevent "
      "unseen-value errors."),

    H1("3. PD validation results (out-of-time, 2018 vintage)"),
    tbl([
        ["Metric", "Scorecard", "XGBoost", "Delta"],
        ["AUC",  "0.6932", "0.7130", "+0.020"],
        ["KS",   "0.2839", "0.3139", "+0.030"],
        ["Gini", "0.3864", "0.4260", "+0.040"],
        ["Brier","0.1283", "0.1258", "-0.003"],
    ], col_widths=[1.4*inch, 1.0*inch, 1.0*inch, 0.9*inch]),
    Spacer(1, 4),
    P("PSI train -&gt; test = 0.030 (stable). Top decile predicted PD overshoots realized by "
      "~14 percentage points, attributable to vintage maturity bias (2018 loans not fully matured)."),

    H1("4. LGD model"),
    P("XGBoost regressor on realized_lgd = 1 - recoveries / funded_amnt for defaulted loans. "
      "Predictions clipped to [0.05, 0.95]. Out-of-time MAE = 0.07. R-squared near zero "
      "out-of-time, attributable to the structural difficulty of unsecured-credit LGD "
      "(recoveries are driven by post-default events not observable at origination)."),

    PageBreak(),
]

# Page 2: Portfolio credit + market + limitations
story += [
    H1("5. Expected Loss and Basel IRB"),
    P("EL = PD * LGD * EAD per loan. Basel III IRB capital charge per dollar of EAD:"),
    P("K = LGD * [Phi((Phi^-1(PD) + sqrt(rho) * Phi^-1(0.999)) / sqrt(1 - rho)) - PD] * M_adj"),
    P("Asset correlation rho uses the Basel corporate formula. RWA = K * 12.5 * EAD."),

    H1("6. Portfolio credit risk"),
    H2("Vasicek ASRF (analytical, Basel IRB foundation)"),
    P("Each obligor's latent asset value A_i = sqrt(rho) * M + sqrt(1 - rho) * eps_i; "
      "default if A_i &lt; Phi^-1(PD). In the large-portfolio limit, conditional PD given an "
      "alpha-quantile factor draw has a closed form, which IS the IRB formula."),

    H2("Copula Monte Carlo"),
    P("Same one-factor structure as Vasicek, simulated. Gaussian copula uses N(0,1) marginals; "
      "Student-t copula scales the Gaussian latent by sqrt(chi-squared_df / df), preserving "
      "joint tail dependence. Default threshold is Phi^-1(PD) or T_df^-1(PD) respectively."),

    H2("Validation"),
    P("On a 1,000-obligor portfolio (PD 0.20, LGD 0.92, rho 0.10, EAD $15,000): "
      "Vasicek ASRF VaR(99.9%) = $7.68M, Gaussian MC = $7.81M (1.6% sampling noise). "
      "Student-t copula (df=5) VaR(99.9%) = $9.20M, Economic Capital = $6.44M, "
      "28% above Gaussian under identical inputs. Headline 2008-modeling-story result."),

    H1("7. Market risk (ported from standalone VaR project)"),
    P("Five families: Historical Simulation, Variance-Covariance (Normal &amp; Student-t), "
      "Monte Carlo (MV-Normal &amp; MV-t), Filtered Historical Simulation with GARCH(1,1) residuals, "
      "and Extreme Value Theory via Peaks-Over-Threshold."),
    P("Out-of-sample backtest results (rolling 500-day window, GARCH refit every 60 days):"),
    tbl([
        ["Method", "99% exc rate", "Kupiec p", "Christof-ind p", "Verdict"],
        ["Historical",         "1.30%", "0.142", "0.001", "Right rate, clustered"],
        ["Parametric Normal",  "2.41%", "0.000", "0.000", "Over-exceeds 2.4x"],
        ["MC Normal",          "2.29%", "0.000", "0.000", "Same"],
        ["FHS GARCH(1,1)",     "1.13%", "0.547", "0.029", "Passes both"],
    ], col_widths=[1.4*inch, 0.95*inch, 0.85*inch, 1.10*inch, 1.55*inch]),
    P("EVT-GPD on 5% worst losses gives shape parameter xi = 0.31 (heavy tail). "
      "EVT-extrapolated 99.9% VaR = 4.14%, more than 2x the Parametric-Normal estimate "
      "at the same confidence."),

    H1("8. Known limitations"),
    P("1) <b>Vintage maturity bias.</b> 2018 LendingClub loans not fully matured at the "
      "snapshot date; realized default rates are artificially low. The rigorous fix is a "
      "fixed observation window (e.g., default-within-24-months)."),
    P("2) <b>LGD has limited signal.</b> Unsecured consumer LGD is dominated by post-default "
      "events; R-squared near zero on out-of-time test is structural. Mean is well-calibrated."),
    P("3) <b>Homogeneous portfolio assumption</b> in the /portfolio/credit_var endpoint. "
      "Heterogeneous portfolios are handled via the /credit/portfolio_el endpoint instead."),
    P("4) <b>Docker image ships with stub models.</b> Trained PD/LGD pickles must be generated "
      "locally; API gracefully falls back to calibrated stubs when pickles are absent."),
    P("5) <b>No counterparty / collateral framework.</b> Pure unsecured PD/LGD/EAD - real "
      "books need collateral haircuts and guarantor netting."),
]

doc = SimpleDocTemplate(
    str(OUT), pagesize=LETTER,
    leftMargin=0.7*inch, rightMargin=0.7*inch,
    topMargin=0.6*inch, bottomMargin=0.6*inch,
    title="Risk Platform - Model Methodology", author="John Slye",
)
doc.build(story)
print(f"Wrote {OUT}")
