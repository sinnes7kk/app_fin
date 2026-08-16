# Momentum score — shadow head-to-head — 2026-08-16 22:21

Panel: **1015 rows** joined on (as_of, ticker, direction) with a populated replay `realized_r`.

Walk-forward: 5 purged folds, label horizon 15d (López de Prado purge + embargo).

---

## 1. Overall rank IC (Spearman vs realized_r)

| Score | n | Pooled Spearman | OOS mean-fold | OOS pooled | folds |
| --- | --- | --- | --- | --- | --- |
| `momentum_score` | 848 | +0.039 | +0.060 | +0.043 | 3 |
| `conviction_score` | 881 | +0.073 | +0.022 | +0.044 | 3 |

## 2. Per-DTE-bucket rank IC

| Score | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `momentum_score` | -0.89 (n=5) | -0.21 (n=31) | +0.10 (n=460) | -0.08 (n=219) | +0.09 (n=133) |
| `conviction_score` | +0.35 (n=5) | -0.03 (n=32) | +0.07 (n=475) | +0.02 (n=219) | +0.18 (n=150) |

## 3. Tercile lift (mean realized_r: top third − bottom third)

| Score | n | Top⅓ mean R | Bottom⅓ mean R | Lift |
| --- | --- | --- | --- | --- |
| `momentum_score` | 848 | +0.073 | +0.035 | +0.038 |
| `conviction_score` | 881 | +0.278 | +0.018 | +0.260 |

## 4. Promotion gate

- Gate A — momentum OOS mean-fold ≥ **+0.10**: +0.060 → fail
- Gate B — edge over conviction ≥ **+0.05**: +0.037 (momentum +0.060 vs conviction +0.022) → fail

**Verdict: ⛔ HOLD (shadow)**

_Note: with a single in-sample market regime and a still-small fold count, treat a passing verdict as necessary-not-sufficient; re-confirm across 3-4 weeks of fresh folds before any cutover._
