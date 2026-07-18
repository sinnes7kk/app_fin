# Momentum score — shadow head-to-head — 2026-07-18 19:16

Panel: **539 rows** joined on (as_of, ticker, direction) with a populated replay `realized_r`.

Walk-forward: 5 purged folds, label horizon 15d (López de Prado purge + embargo).

---

## 1. Overall rank IC (Spearman vs realized_r)

| Score | n | Pooled Spearman | OOS mean-fold | OOS pooled | folds |
| --- | --- | --- | --- | --- | --- |
| `momentum_score` | 275 | +0.030 | -0.021 | -0.034 | 3 |
| `conviction_score` | 295 | +0.042 | -0.117 | -0.076 | 3 |

## 2. Per-DTE-bucket rank IC

| Score | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `momentum_score` | — | -0.07 (n=14) | +0.02 (n=100) | +0.09 (n=76) | +0.03 (n=82) |
| `conviction_score` | — | -0.38 (n=15) | -0.01 (n=104) | +0.13 (n=76) | +0.08 (n=97) |

## 3. Tercile lift (mean realized_r: top third − bottom third)

| Score | n | Top⅓ mean R | Bottom⅓ mean R | Lift |
| --- | --- | --- | --- | --- |
| `momentum_score` | 275 | +0.271 | +0.272 | -0.001 |
| `conviction_score` | 295 | +0.396 | +0.242 | +0.154 |

## 4. Promotion gate

- Gate A — momentum OOS mean-fold ≥ **+0.10**: -0.021 → fail
- Gate B — edge over conviction ≥ **+0.05**: +0.095 (momentum -0.021 vs conviction -0.117) → PASS

**Verdict: ⛔ HOLD (shadow)**

_Note: with a single in-sample market regime and a still-small fold count, treat a passing verdict as necessary-not-sufficient; re-confirm across 3-4 weeks of fresh folds before any cutover._
