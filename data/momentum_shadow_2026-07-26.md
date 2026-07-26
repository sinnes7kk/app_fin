# Momentum score — shadow head-to-head — 2026-07-26 22:59

Panel: **789 rows** joined on (as_of, ticker, direction) with a populated replay `realized_r`.

Walk-forward: 5 purged folds, label horizon 15d (López de Prado purge + embargo).

---

## 1. Overall rank IC (Spearman vs realized_r)

| Score | n | Pooled Spearman | OOS mean-fold | OOS pooled | folds |
| --- | --- | --- | --- | --- | --- |
| `momentum_score` | 287 | -0.008 | -0.115 | -0.114 | 3 |
| `conviction_score` | 307 | +0.035 | -0.183 | -0.081 | 3 |

## 2. Per-DTE-bucket rank IC

| Score | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `momentum_score` | — | -0.08 (n=14) | -0.03 (n=107) | +0.06 (n=78) | -0.02 (n=85) |
| `conviction_score` | — | -0.39 (n=15) | -0.03 (n=111) | +0.09 (n=78) | +0.10 (n=100) |

## 3. Tercile lift (mean realized_r: top third − bottom third)

| Score | n | Top⅓ mean R | Bottom⅓ mean R | Lift |
| --- | --- | --- | --- | --- |
| `momentum_score` | 287 | +0.113 | +0.238 | -0.126 |
| `conviction_score` | 307 | +0.239 | +0.203 | +0.037 |

## 4. Promotion gate

- Gate A — momentum OOS mean-fold ≥ **+0.10**: -0.115 → fail
- Gate B — edge over conviction ≥ **+0.05**: +0.068 (momentum -0.115 vs conviction -0.183) → PASS

**Verdict: ⛔ HOLD (shadow)**

_Note: with a single in-sample market regime and a still-small fold count, treat a passing verdict as necessary-not-sufficient; re-confirm across 3-4 weeks of fresh folds before any cutover._
