# Momentum score — shadow head-to-head — 2026-08-02 22:57

Panel: **869 rows** joined on (as_of, ticker, direction) with a populated replay `realized_r`.

Walk-forward: 5 purged folds, label horizon 15d (López de Prado purge + embargo).

---

## 1. Overall rank IC (Spearman vs realized_r)

| Score | n | Pooled Spearman | OOS mean-fold | OOS pooled | folds |
| --- | --- | --- | --- | --- | --- |
| `momentum_score` | 301 | +0.002 | -0.012 | -0.051 | 3 |
| `conviction_score` | 321 | +0.031 | -0.091 | -0.090 | 3 |

## 2. Per-DTE-bucket rank IC

| Score | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `momentum_score` | — | -0.01 (n=15) | -0.03 (n=110) | +0.04 (n=83) | +0.02 (n=90) |
| `conviction_score` | — | -0.43 (n=16) | -0.04 (n=114) | +0.11 (n=83) | +0.10 (n=105) |

## 3. Tercile lift (mean realized_r: top third − bottom third)

| Score | n | Top⅓ mean R | Bottom⅓ mean R | Lift |
| --- | --- | --- | --- | --- |
| `momentum_score` | 301 | +0.176 | +0.246 | -0.070 |
| `conviction_score` | 321 | +0.241 | +0.192 | +0.049 |

## 4. Promotion gate

- Gate A — momentum OOS mean-fold ≥ **+0.10**: -0.012 → fail
- Gate B — edge over conviction ≥ **+0.05**: +0.079 (momentum -0.012 vs conviction -0.091) → PASS

**Verdict: ⛔ HOLD (shadow)**

_Note: with a single in-sample market regime and a still-small fold count, treat a passing verdict as necessary-not-sufficient; re-confirm across 3-4 weeks of fresh folds before any cutover._
