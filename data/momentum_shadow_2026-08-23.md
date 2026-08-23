# Momentum score — shadow head-to-head — 2026-08-23 22:22

Panel: **1093 rows** joined on (as_of, ticker, direction) with a populated replay `realized_r`.

Walk-forward: 5 purged folds, label horizon 15d (López de Prado purge + embargo).

---

## 1. Overall rank IC (Spearman vs realized_r)

| Score | n | Pooled Spearman | OOS mean-fold | OOS pooled | folds |
| --- | --- | --- | --- | --- | --- |
| `momentum_score` | 980 | +0.027 | +0.030 | +0.032 | 3 |
| `conviction_score` | 1013 | +0.084 | +0.059 | +0.083 | 3 |

## 2. Per-DTE-bucket rank IC

| Score | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `momentum_score` | -0.19 (n=6) | -0.21 (n=42) | +0.09 (n=539) | -0.09 (n=244) | +0.06 (n=149) |
| `conviction_score` | +0.57 (n=6) | -0.03 (n=43) | +0.08 (n=554) | +0.03 (n=244) | +0.20 (n=166) |

## 3. Tercile lift (mean realized_r: top third − bottom third)

| Score | n | Top⅓ mean R | Bottom⅓ mean R | Lift |
| --- | --- | --- | --- | --- |
| `momentum_score` | 980 | +0.093 | +0.070 | +0.023 |
| `conviction_score` | 1013 | +0.278 | -0.016 | +0.294 |

## 4. Promotion gate

- Gate A — momentum OOS mean-fold ≥ **+0.10**: +0.030 → fail
- Gate B — edge over conviction ≥ **+0.05**: -0.029 (momentum +0.030 vs conviction +0.059) → fail

**Verdict: ⛔ HOLD (shadow)**

_Note: with a single in-sample market regime and a still-small fold count, treat a passing verdict as necessary-not-sufficient; re-confirm across 3-4 weeks of fresh folds before any cutover._
