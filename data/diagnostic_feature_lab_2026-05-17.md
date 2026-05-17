# Feature lab — Spearman ranking — 2026-05-17 22:55

Joined `feature_lab.csv` × `grade_history_with_replay.csv` on (as_of, ticker, direction).  Panel size: **57 rows** (after dropping rows without realized_r).

Spearman is a rank correlation between each candidate feature and the bar-by-bar replay `realized_r`. Features with consistent |Spearman| ≥ 0.10 across multiple DTE buckets and a positive walk-forward OOS Spearman are promotion candidates. Features with consistently *negative* Spearman are candidates for sign inversion.

**Caveat:** until the panel reaches ~250 closed-and-replayed rows any single ranking is dominated by sampling noise. Treat this as a watchlist of hypotheses, not a hit list of fixes.

---

## 1. Overall ranking

| Feature | n | Spearman | OOS Spearman | n_val |
| --- | --- | --- | --- | --- |
| `prem_momentum_z3d` | 7 | +0.536 | — | 0 |
| `dollar_delta_weighted_flow` | 19 | -0.526 | -0.383 | 8 |
| `gex_total` | 19 | -0.488 | -0.515 | 8 |
| `far_otm_put_share` | 19 | +0.342 | +0.467 | 8 |
| `bullish_premium_share` | 19 | -0.319 | -0.299 | 8 |
| `unusual_premium_share` | 7 | -0.318 | — | 0 |
| `term_slope_30_90` | 19 | -0.214 | -0.683 | 8 |
| `sector_relative_pct` | 18 | +0.200 | +0.323 | 8 |
| `vanna_total` | 19 | +0.188 | +0.347 | 8 |
| `realized_vol_regime` | 31 | +0.183 | +0.228 | 13 |
| `far_otm_call_share` | 19 | -0.161 | -0.240 | 8 |
| `expiry_concentration_top1` | 19 | -0.158 | -0.084 | 8 |
| `max_pain_dist_pct` | 31 | -0.149 | -0.735 | 13 |
| `vrp_proxy` | 31 | -0.115 | +0.008 | 13 |
| `atm_iv_90d` | 19 | -0.087 | -0.108 | 8 |
| `charm_total` | 19 | -0.084 | -0.228 | 8 |
| `atm_iv_60d` | 19 | -0.019 | -0.108 | 8 |
| `atm_iv_30d` | 19 | -0.017 | -0.024 | 8 |
| `iv_skew_25d` | 0 | — | — | 0 |
| `dealer_net_delta_at_spot` | 3 | — | — | 0 |
| `dealer_net_gamma_at_spot` | 3 | — | — | 0 |

## 2. Per-DTE-bucket breakdown

| Feature | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `prem_momentum_z3d` | — | — | — | — | — |
| `dollar_delta_weighted_flow` | — | — | — | -0.33 (n=10) | -0.44 (n=8) |
| `gex_total` | — | — | — | -0.48 (n=10) | -0.53 (n=9) |
| `far_otm_put_share` | — | — | — | +0.28 (n=10) | +0.60 (n=8) |
| `bullish_premium_share` | — | — | — | +0.04 (n=10) | -0.76 (n=9) |
| `unusual_premium_share` | — | — | — | -0.32 (n=7) | — |
| `term_slope_30_90` | — | — | — | -0.02 (n=10) | -0.50 (n=9) |
| `sector_relative_pct` | — | — | — | -0.16 (n=7) | +0.33 (n=10) |
| `vanna_total` | — | — | — | +0.13 (n=10) | +0.35 (n=9) |
| `realized_vol_regime` | — | — | — | +0.29 (n=13) | -0.24 (n=16) |
| `far_otm_call_share` | — | — | — | -0.04 (n=10) | -0.02 (n=8) |
| `expiry_concentration_top1` | — | — | — | -0.50 (n=10) | +0.43 (n=9) |
| `max_pain_dist_pct` | — | — | — | -0.48 (n=13) | +0.17 (n=16) |
| `vrp_proxy` | — | — | — | -0.01 (n=13) | -0.08 (n=16) |
| `atm_iv_90d` | — | — | — | -0.13 (n=10) | +0.02 (n=9) |
| `charm_total` | — | — | — | +0.04 (n=10) | -0.43 (n=9) |
| `atm_iv_60d` | — | — | — | -0.09 (n=10) | +0.02 (n=9) |
| `atm_iv_30d` | — | — | — | -0.12 (n=10) | +0.08 (n=9) |
| `iv_skew_25d` | — | — | — | — | — |
| `dealer_net_delta_at_spot` | — | — | — | — | — |
| `dealer_net_gamma_at_spot` | — | — | — | — | — |

## 3. Promotion candidates

| Feature | n | Spearman | OOS Spearman | Action |
| --- | --- | --- | --- | --- |
| `realized_vol_regime` | 31 | +0.183 | +0.228 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `vrp_proxy` | 31 | -0.115 | +0.008 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
