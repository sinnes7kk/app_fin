# Momentum score — shadow head-to-head — 2026-09-06 23:33

Panel: **1244 rows** joined on (as_of, ticker, direction) with a populated replay `realized_r`.

Walk-forward: 5 purged folds, label horizon 15d (López de Prado purge + embargo).

---

## 1. Overall rank IC (Spearman vs realized_r)

| Score | n | Pooled Spearman | OOS mean-fold | OOS pooled | folds |
| --- | --- | --- | --- | --- | --- |
| `momentum_score` | 1065 | +0.013 | -0.002 | +0.016 | 3 |
| `conviction_score` | 1098 | +0.072 | +0.063 | +0.065 | 3 |

## 2. Per-DTE-bucket rank IC

| Score | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `momentum_score` | -0.22 (n=7) | -0.27 (n=51) | +0.08 (n=580) | -0.09 (n=264) | +0.04 (n=163) |
| `conviction_score` | +0.59 (n=7) | +0.02 (n=52) | +0.06 (n=595) | +0.01 (n=264) | +0.20 (n=180) |

## 3. Tercile lift (mean realized_r: top third − bottom third)

| Score | n | Top⅓ mean R | Bottom⅓ mean R | Lift |
| --- | --- | --- | --- | --- |
| `momentum_score` | 1065 | +0.059 | +0.064 | -0.005 |
| `conviction_score` | 1098 | +0.248 | +0.010 | +0.238 |

## 4. Promotion gate

- Gate A — momentum OOS mean-fold ≥ **+0.10**: -0.002 → fail
- Gate B — edge over conviction ≥ **+0.05**: -0.065 (momentum -0.002 vs conviction +0.063) → fail

**Verdict: ⛔ HOLD (shadow)**

_Note: with a single in-sample market regime and a still-small fold count, treat a passing verdict as necessary-not-sufficient; re-confirm across 3-4 weeks of fresh folds before any cutover._
