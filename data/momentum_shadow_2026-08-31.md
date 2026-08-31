# Momentum score — shadow head-to-head — 2026-08-31 00:14

Panel: **1165 rows** joined on (as_of, ticker, direction) with a populated replay `realized_r`.

Walk-forward: 5 purged folds, label horizon 15d (López de Prado purge + embargo).

---

## 1. Overall rank IC (Spearman vs realized_r)

| Score | n | Pooled Spearman | OOS mean-fold | OOS pooled | folds |
| --- | --- | --- | --- | --- | --- |
| `momentum_score` | 1052 | +0.015 | +0.007 | +0.011 | 3 |
| `conviction_score` | 1085 | +0.071 | +0.040 | +0.061 | 3 |

## 2. Per-DTE-bucket rank IC

| Score | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `momentum_score` | -0.22 (n=7) | -0.26 (n=47) | +0.09 (n=576) | -0.09 (n=264) | +0.04 (n=158) |
| `conviction_score` | +0.59 (n=7) | -0.02 (n=48) | +0.06 (n=591) | +0.01 (n=264) | +0.20 (n=175) |

## 3. Tercile lift (mean realized_r: top third − bottom third)

| Score | n | Top⅓ mean R | Bottom⅓ mean R | Lift |
| --- | --- | --- | --- | --- |
| `momentum_score` | 1052 | +0.058 | +0.065 | -0.007 |
| `conviction_score` | 1085 | +0.252 | +0.017 | +0.235 |

## 4. Promotion gate

- Gate A — momentum OOS mean-fold ≥ **+0.10**: +0.007 → fail
- Gate B — edge over conviction ≥ **+0.05**: -0.033 (momentum +0.007 vs conviction +0.040) → fail

**Verdict: ⛔ HOLD (shadow)**

_Note: with a single in-sample market regime and a still-small fold count, treat a passing verdict as necessary-not-sufficient; re-confirm across 3-4 weeks of fresh folds before any cutover._
