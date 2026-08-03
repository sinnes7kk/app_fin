# Momentum score — shadow head-to-head — 2026-08-03 07:50

Panel: **881 rows** joined on (as_of, ticker, direction) with a populated replay `realized_r`.

Walk-forward: 5 purged folds, label horizon 15d (López de Prado purge + embargo).

---

## 1. Overall rank IC (Spearman vs realized_r)

| Score | n | Pooled Spearman | OOS mean-fold | OOS pooled | folds |
| --- | --- | --- | --- | --- | --- |
| `momentum_score` | 826 | +0.039 | +0.041 | +0.036 | 3 |
| `conviction_score` | 859 | +0.076 | +0.040 | +0.059 | 3 |

## 2. Per-DTE-bucket rank IC

| Score | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `momentum_score` | — | -0.25 (n=28) | +0.09 (n=455) | -0.09 (n=218) | +0.10 (n=121) |
| `conviction_score` | — | +0.03 (n=29) | +0.07 (n=470) | +0.02 (n=218) | +0.17 (n=138) |

## 3. Tercile lift (mean realized_r: top third − bottom third)

| Score | n | Top⅓ mean R | Bottom⅓ mean R | Lift |
| --- | --- | --- | --- | --- |
| `momentum_score` | 826 | +0.054 | +0.032 | +0.022 |
| `conviction_score` | 859 | +0.280 | +0.012 | +0.268 |

## 4. Promotion gate

- Gate A — momentum OOS mean-fold ≥ **+0.10**: +0.041 → fail
- Gate B — edge over conviction ≥ **+0.05**: +0.001 (momentum +0.041 vs conviction +0.040) → fail

**Verdict: ⛔ HOLD (shadow)**

_Note: with a single in-sample market regime and a still-small fold count, treat a passing verdict as necessary-not-sufficient; re-confirm across 3-4 weeks of fresh folds before any cutover._
