# Momentum score — shadow head-to-head — 2026-09-28 00:20

Panel: **1286 rows** joined on (as_of, ticker, direction) with a populated replay `realized_r`.

Walk-forward: 5 purged folds, label horizon 15d (López de Prado purge + embargo).

---

## 1. Overall rank IC (Spearman vs realized_r)

| Score | n | Pooled Spearman | OOS mean-fold | OOS pooled | folds |
| --- | --- | --- | --- | --- | --- |
| `momentum_score` | 1246 | +0.020 | +0.023 | +0.025 | 3 |
| `conviction_score` | 1280 | +0.046 | -0.005 | -0.002 | 3 |

## 2. Per-DTE-bucket rank IC

| Score | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `momentum_score` | -0.20 (n=7) | -0.10 (n=68) | +0.05 (n=671) | -0.05 (n=314) | +0.06 (n=186) |
| `conviction_score` | +0.64 (n=7) | +0.07 (n=70) | +0.04 (n=686) | -0.00 (n=314) | +0.15 (n=203) |

## 3. Tercile lift (mean realized_r: top third − bottom third)

| Score | n | Top⅓ mean R | Bottom⅓ mean R | Lift |
| --- | --- | --- | --- | --- |
| `momentum_score` | 1246 | +0.034 | +0.030 | +0.004 |
| `conviction_score` | 1280 | +0.197 | +0.023 | +0.174 |

## 4. Promotion gate

- Gate A — momentum OOS mean-fold ≥ **+0.10**: +0.023 → fail
- Gate B — edge over conviction ≥ **+0.05**: +0.029 (momentum +0.023 vs conviction -0.005) → fail

**Verdict: ⛔ HOLD (shadow)**

_Note: with a single in-sample market regime and a still-small fold count, treat a passing verdict as necessary-not-sufficient; re-confirm across 3-4 weeks of fresh folds before any cutover._
