# Meta-labeling — shadow report — 2026-08-23 22:22

Rows used: **0**  |  OOS folds: **0**  |  walk-forward 5 splits, 15d label horizon.

Features: —

---

## Out-of-sample metrics

- **ROC-AUC:** —  _(0.50 = coin flip; >0.55 = usable)_
- **Brier score:** —  _(lower = better calibrated)_
- **Base win rate:** —

## Realized-R lift by P(win)

- Top⅓ P(win) mean realized R: **—**
- Bottom⅓ P(win) mean realized R: **—**
- **Lift (top − bottom): —**

**Verdict: ⛔ no reliable OOS edge yet — keep shadow, keep collecting**

_Note: scikit-learn unavailable_

_Meta-labeling never changes trade direction — it only scales conviction/size. Promote to live sizing only after the OOS edge holds across 3-4 weeks of fresh folds._
