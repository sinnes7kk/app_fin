# Momentum score — shadow head-to-head — 2026-11-02 01:19

Panel: **1286 rows** joined on (as_of, ticker, direction) with a populated replay `realized_r`.

Walk-forward: 5 purged folds, label horizon 15d (López de Prado purge + embargo).

---

## 1. Overall rank IC (Spearman vs realized_r)

| Score | n | Pooled Spearman | OOS mean-fold | OOS pooled | folds |
| --- | --- | --- | --- | --- | --- |
| `momentum_score` | 1250 | +0.017 | +0.018 | +0.020 | 3 |
| `conviction_score` | 1284 | +0.045 | -0.007 | -0.003 | 3 |

## 2. Per-DTE-bucket rank IC

| Score | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `momentum_score` | -0.40 (n=9) | -0.10 (n=68) | +0.05 (n=672) | -0.05 (n=315) | +0.06 (n=186) |
| `conviction_score` | +0.50 (n=9) | +0.07 (n=70) | +0.03 (n=687) | +0.00 (n=315) | +0.15 (n=203) |

## 3. Tercile lift (mean realized_r: top third − bottom third)

| Score | n | Top⅓ mean R | Bottom⅓ mean R | Lift |
| --- | --- | --- | --- | --- |
| `momentum_score` | 1250 | +0.046 | +0.048 | -0.002 |
| `conviction_score` | 1284 | +0.201 | +0.033 | +0.168 |

## 4. Promotion gate

- Gate A — momentum OOS mean-fold ≥ **+0.10**: +0.018 → fail
- Gate B — edge over conviction ≥ **+0.05**: +0.025 (momentum +0.018 vs conviction -0.007) → fail

**Verdict: ⛔ HOLD (shadow)**

_Note: with a single in-sample market regime and a still-small fold count, treat a passing verdict as necessary-not-sufficient; re-confirm across 3-4 weeks of fresh folds before any cutover._
