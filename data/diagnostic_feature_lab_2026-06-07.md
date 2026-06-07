# Feature lab — Spearman ranking — 2026-06-07 23:06

Joined `feature_lab.csv` × `grade_history_with_replay.csv` on (as_of, ticker, direction).  Panel size: **292 rows** (after dropping rows without realized_r).

Spearman is a rank correlation between each candidate feature and the bar-by-bar replay `realized_r`. Features with consistent |Spearman| ≥ 0.10 across multiple DTE buckets and a positive walk-forward OOS Spearman are promotion candidates. Features with consistently *negative* Spearman are candidates for sign inversion.

**Caveat:** until the panel reaches ~250 closed-and-replayed rows any single ranking is dominated by sampling noise. Treat this as a watchlist of hypotheses, not a hit list of fixes.

---

## 1. Overall ranking

| Feature | n | Spearman | OOS Spearman | n_val |
| --- | --- | --- | --- | --- |
| `prem_momentum_z3d` | 18 | +0.507 | +0.548 | 8 |
| `dealer_net_delta_at_spot` | 50 | -0.204 | -0.248 | 20 |
| `far_otm_put_share` | 106 | +0.203 | +0.071 | 43 |
| `far_otm_call_share` | 106 | -0.195 | -0.167 | 43 |
| `sector_relative_pct` | 100 | +0.161 | +0.199 | 40 |
| `charm_total` | 142 | +0.158 | +0.335 | 57 |
| `realized_vol_regime` | 150 | -0.137 | -0.255 | 60 |
| `max_pain_dist_pct` | 154 | +0.108 | +0.203 | 62 |
| `term_slope_30_90` | 142 | +0.094 | +0.207 | 57 |
| `dollar_delta_weighted_flow` | 106 | +0.072 | +0.265 | 43 |
| `iv_skew_25d` | 24 | -0.061 | -0.050 | 10 |
| `gex_total` | 142 | -0.055 | +0.138 | 57 |
| `vanna_total` | 142 | +0.052 | -0.021 | 57 |
| `expiry_concentration_top1` | 142 | +0.048 | +0.172 | 57 |
| `atm_iv_30d` | 142 | -0.047 | -0.053 | 57 |
| `atm_iv_60d` | 142 | -0.033 | -0.007 | 57 |
| `bullish_premium_share` | 142 | +0.030 | +0.253 | 57 |
| `atm_iv_90d` | 142 | -0.023 | -0.002 | 57 |
| `dealer_net_gamma_at_spot` | 50 | +0.020 | +0.022 | 20 |
| `vrp_proxy` | 150 | +0.017 | -0.044 | 60 |
| `unusual_premium_share` | 62 | -0.006 | +0.051 | 25 |

## 2. Per-DTE-bucket breakdown

| Feature | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `prem_momentum_z3d` | — | — | -0.07 (n=7) | +0.32 (n=7) | — |
| `dealer_net_delta_at_spot` | — | — | -0.10 (n=31) | -0.08 (n=16) | — |
| `far_otm_put_share` | — | — | +0.16 (n=46) | +0.12 (n=35) | +0.46 (n=23) |
| `far_otm_call_share` | — | — | -0.24 (n=46) | -0.18 (n=35) | -0.27 (n=23) |
| `sector_relative_pct` | — | — | +0.25 (n=41) | +0.13 (n=30) | +0.12 (n=27) |
| `charm_total` | — | -0.16 (n=5) | +0.19 (n=54) | +0.32 (n=43) | -0.17 (n=40) |
| `realized_vol_regime` | — | +0.24 (n=6) | -0.31 (n=53) | -0.16 (n=45) | -0.23 (n=46) |
| `max_pain_dist_pct` | — | +0.18 (n=6) | +0.13 (n=55) | -0.04 (n=46) | +0.20 (n=47) |
| `term_slope_30_90` | — | -0.47 (n=5) | +0.00 (n=54) | +0.18 (n=43) | +0.16 (n=40) |
| `dollar_delta_weighted_flow` | — | — | +0.18 (n=46) | +0.09 (n=35) | -0.05 (n=23) |
| `iv_skew_25d` | — | — | -0.03 (n=13) | -0.02 (n=7) | — |
| `gex_total` | — | +0.00 (n=5) | -0.01 (n=54) | -0.33 (n=43) | +0.06 (n=40) |
| `vanna_total` | — | +0.00 (n=5) | +0.00 (n=54) | -0.11 (n=43) | +0.22 (n=40) |
| `expiry_concentration_top1` | — | -0.63 (n=5) | +0.12 (n=54) | +0.24 (n=43) | -0.06 (n=40) |
| `atm_iv_30d` | — | +0.00 (n=5) | -0.07 (n=54) | -0.00 (n=43) | -0.08 (n=40) |
| `atm_iv_60d` | — | -0.32 (n=5) | -0.03 (n=54) | +0.02 (n=43) | -0.10 (n=40) |
| `bullish_premium_share` | — | -0.08 (n=5) | +0.02 (n=54) | +0.11 (n=43) | +0.02 (n=40) |
| `atm_iv_90d` | — | -0.32 (n=5) | -0.01 (n=54) | +0.05 (n=43) | -0.07 (n=40) |
| `dealer_net_gamma_at_spot` | — | — | +0.09 (n=31) | +0.10 (n=16) | — |
| `vrp_proxy` | — | -0.04 (n=6) | +0.08 (n=53) | +0.00 (n=45) | +0.02 (n=46) |
| `unusual_premium_share` | — | — | +0.13 (n=41) | -0.32 (n=21) | — |

## 3. Promotion candidates

| Feature | n | Spearman | OOS Spearman | Action |
| --- | --- | --- | --- | --- |
| `far_otm_put_share` | 106 | +0.203 | +0.071 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `sector_relative_pct` | 100 | +0.161 | +0.199 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `charm_total` | 142 | +0.158 | +0.335 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `max_pain_dist_pct` | 154 | +0.108 | +0.203 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
