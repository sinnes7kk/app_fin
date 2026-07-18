# Meta-labeling — shadow report — 2026-07-18 19:21

Rows used: **255**  |  OOS folds: **2**  |  walk-forward 5 splits, 15d label horizon.

Features: `bullish_premium_share`, `sector_relative_pct`, `dollar_delta_weighted_flow`, `aggressor_bull_share`, `aggressor_net_prem_bps`, `ask_side_ratio`, `directional_sweep_share`, `far_otm_call_share`, `realized_vol_regime`, `momentum_composite`

---

## Out-of-sample metrics

- **ROC-AUC:** +0.662  _(0.50 = coin flip; >0.55 = usable)_
- **Brier score:** +0.232  _(lower = better calibrated)_
- **Base win rate:** +0.571

## Realized-R lift by P(win)

- Top⅓ P(win) mean realized R: **+0.646**
- Bottom⅓ P(win) mean realized R: **-0.109**
- **Lift (top − bottom): +0.755**

## Decile sizing curve (mean realized R, low→high P(win))

| D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| -0.13 | -0.47 | +0.06 | +0.26 | +0.45 | +0.09 | +0.61 | +0.10 | +0.96 | +0.78 |

**Verdict: ✅ meta-labeling shows OOS edge — candidate for sizing**

_Meta-labeling never changes trade direction — it only scales conviction/size. Promote to live sizing only after the OOS edge holds across 3-4 weeks of fresh folds._
