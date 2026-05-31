# Feature lab — Spearman ranking — 2026-05-31 23:01

Joined `feature_lab.csv` × `grade_history_with_replay.csv` on (as_of, ticker, direction).  Panel size: **204 rows** (after dropping rows without realized_r).

Spearman is a rank correlation between each candidate feature and the bar-by-bar replay `realized_r`. Features with consistent |Spearman| ≥ 0.10 across multiple DTE buckets and a positive walk-forward OOS Spearman are promotion candidates. Features with consistently *negative* Spearman are candidates for sign inversion.

**Caveat:** until the panel reaches ~250 closed-and-replayed rows any single ranking is dominated by sampling noise. Treat this as a watchlist of hypotheses, not a hit list of fixes.

---

## 1. Overall ranking

| Feature | n | Spearman | OOS Spearman | n_val |
| --- | --- | --- | --- | --- |
| `prem_momentum_z3d` | 17 | +0.540 | +0.643 | 7 |
| `far_otm_call_share` | 79 | -0.252 | -0.339 | 32 |
| `dealer_net_delta_at_spot` | 33 | -0.244 | -0.236 | 14 |
| `far_otm_put_share` | 79 | +0.211 | +0.028 | 32 |
| `gex_total` | 108 | -0.201 | +0.052 | 44 |
| `sector_relative_pct` | 79 | +0.154 | +0.195 | 32 |
| `dealer_net_gamma_at_spot` | 33 | -0.120 | -0.035 | 14 |
| `charm_total` | 108 | +0.114 | +0.222 | 44 |
| `bullish_premium_share` | 108 | -0.094 | +0.090 | 44 |
| `unusual_premium_share` | 46 | -0.077 | -0.190 | 19 |
| `atm_iv_60d` | 108 | -0.055 | +0.001 | 44 |
| `dollar_delta_weighted_flow` | 79 | +0.053 | +0.278 | 32 |
| `atm_iv_90d` | 108 | -0.042 | +0.036 | 44 |
| `atm_iv_30d` | 108 | -0.040 | +0.025 | 44 |
| `expiry_concentration_top1` | 108 | +0.038 | +0.107 | 44 |
| `vanna_total` | 108 | +0.035 | +0.009 | 44 |
| `term_slope_30_90` | 108 | -0.021 | -0.026 | 44 |
| `vrp_proxy` | 116 | +0.017 | +0.089 | 47 |
| `realized_vol_regime` | 116 | -0.016 | -0.017 | 47 |
| `max_pain_dist_pct` | 120 | +0.016 | +0.090 | 48 |
| `iv_skew_25d` | 18 | -0.006 | -0.311 | 8 |

## 2. Per-DTE-bucket breakdown

| Feature | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `prem_momentum_z3d` | — | — | -0.07 (n=7) | +0.83 (n=6) | — |
| `far_otm_call_share` | — | — | -0.27 (n=33) | -0.28 (n=29) | -0.17 (n=15) |
| `dealer_net_delta_at_spot` | — | — | -0.24 (n=21) | -0.17 (n=11) | — |
| `far_otm_put_share` | — | — | +0.10 (n=33) | +0.14 (n=29) | +0.65 (n=15) |
| `gex_total` | — | — | -0.11 (n=38) | -0.36 (n=35) | -0.22 (n=31) |
| `sector_relative_pct` | — | — | +0.21 (n=29) | +0.16 (n=26) | +0.30 (n=22) |
| `dealer_net_gamma_at_spot` | — | — | -0.16 (n=21) | +0.16 (n=11) | — |
| `charm_total` | — | — | +0.07 (n=38) | +0.30 (n=35) | -0.17 (n=31) |
| `bullish_premium_share` | — | — | -0.08 (n=38) | +0.06 (n=35) | -0.32 (n=31) |
| `unusual_premium_share` | — | — | +0.18 (n=30) | -0.64 (n=16) | — |
| `atm_iv_60d` | — | — | +0.02 (n=38) | +0.04 (n=35) | -0.14 (n=31) |
| `dollar_delta_weighted_flow` | — | — | +0.13 (n=33) | +0.01 (n=29) | -0.01 (n=15) |
| `atm_iv_90d` | — | — | +0.03 (n=38) | +0.09 (n=35) | -0.09 (n=31) |
| `atm_iv_30d` | — | — | +0.03 (n=38) | -0.00 (n=35) | -0.13 (n=31) |
| `expiry_concentration_top1` | — | — | +0.05 (n=38) | +0.22 (n=35) | -0.12 (n=31) |
| `vanna_total` | — | — | +0.02 (n=38) | -0.01 (n=35) | +0.04 (n=31) |
| `term_slope_30_90` | — | — | -0.27 (n=38) | +0.25 (n=35) | -0.02 (n=31) |
| `vrp_proxy` | — | +0.29 (n=5) | +0.05 (n=37) | +0.09 (n=37) | -0.07 (n=37) |
| `realized_vol_regime` | — | +0.36 (n=5) | -0.28 (n=37) | -0.14 (n=37) | +0.02 (n=37) |
| `max_pain_dist_pct` | — | -0.05 (n=5) | +0.00 (n=39) | -0.13 (n=38) | +0.17 (n=38) |
| `iv_skew_25d` | — | — | +0.03 (n=9) | -0.15 (n=5) | — |

## 3. Promotion candidates

| Feature | n | Spearman | OOS Spearman | Action |
| --- | --- | --- | --- | --- |
| `far_otm_put_share` | 79 | +0.211 | +0.028 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `gex_total` | 108 | -0.201 | +0.052 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `sector_relative_pct` | 79 | +0.154 | +0.195 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `charm_total` | 108 | +0.114 | +0.222 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
