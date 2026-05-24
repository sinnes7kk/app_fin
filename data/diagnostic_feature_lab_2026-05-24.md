# Feature lab — Spearman ranking — 2026-05-24 22:59

Joined `feature_lab.csv` × `grade_history_with_replay.csv` on (as_of, ticker, direction).  Panel size: **140 rows** (after dropping rows without realized_r).

Spearman is a rank correlation between each candidate feature and the bar-by-bar replay `realized_r`. Features with consistent |Spearman| ≥ 0.10 across multiple DTE buckets and a positive walk-forward OOS Spearman are promotion candidates. Features with consistently *negative* Spearman are candidates for sign inversion.

**Caveat:** until the panel reaches ~250 closed-and-replayed rows any single ranking is dominated by sampling noise. Treat this as a watchlist of hypotheses, not a hit list of fixes.

---

## 1. Overall ranking

| Feature | n | Spearman | OOS Spearman | n_val |
| --- | --- | --- | --- | --- |
| `prem_momentum_z3d` | 13 | +0.616 | — | 0 |
| `far_otm_put_share` | 52 | +0.352 | +0.418 | 21 |
| `iv_skew_25d` | 10 | +0.339 | — | 0 |
| `dealer_net_delta_at_spot` | 17 | -0.291 | -0.750 | 7 |
| `gex_total` | 72 | -0.280 | -0.146 | 29 |
| `dealer_net_gamma_at_spot` | 17 | -0.253 | -0.607 | 7 |
| `far_otm_call_share` | 52 | -0.195 | +0.057 | 21 |
| `bullish_premium_share` | 72 | -0.194 | -0.125 | 29 |
| `dollar_delta_weighted_flow` | 52 | -0.150 | +0.261 | 21 |
| `sector_relative_pct` | 55 | +0.128 | +0.325 | 22 |
| `unusual_premium_share` | 22 | -0.109 | -0.136 | 9 |
| `realized_vol_regime` | 84 | +0.086 | +0.109 | 34 |
| `atm_iv_60d` | 72 | -0.080 | -0.198 | 29 |
| `atm_iv_90d` | 72 | -0.068 | -0.153 | 29 |
| `expiry_concentration_top1` | 72 | +0.065 | +0.057 | 29 |
| `vrp_proxy` | 84 | -0.065 | -0.085 | 34 |
| `atm_iv_30d` | 72 | -0.063 | -0.230 | 29 |
| `charm_total` | 72 | +0.057 | +0.203 | 29 |
| `term_slope_30_90` | 72 | +0.041 | +0.277 | 29 |
| `max_pain_dist_pct` | 84 | +0.025 | +0.254 | 34 |
| `vanna_total` | 72 | -0.020 | +0.020 | 29 |

## 2. Per-DTE-bucket breakdown

| Feature | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `prem_momentum_z3d` | — | — | — | +0.80 (n=5) | — |
| `far_otm_put_share` | — | — | +0.47 (n=17) | +0.30 (n=19) | +0.62 (n=14) |
| `iv_skew_25d` | — | — | — | — | — |
| `dealer_net_delta_at_spot` | — | — | -0.24 (n=10) | -0.64 (n=6) | — |
| `gex_total` | — | — | +0.02 (n=20) | -0.46 (n=23) | -0.15 (n=26) |
| `dealer_net_gamma_at_spot` | — | — | -0.21 (n=10) | +0.00 (n=6) | — |
| `far_otm_call_share` | — | — | -0.13 (n=17) | -0.43 (n=19) | -0.13 (n=14) |
| `bullish_premium_share` | — | — | -0.35 (n=20) | +0.06 (n=23) | -0.30 (n=26) |
| `dollar_delta_weighted_flow` | — | — | -0.20 (n=17) | +0.02 (n=19) | -0.07 (n=14) |
| `sector_relative_pct` | — | — | +0.29 (n=15) | -0.01 (n=18) | +0.34 (n=20) |
| `unusual_premium_share` | — | — | +0.08 (n=14) | -0.48 (n=8) | — |
| `realized_vol_regime` | — | — | -0.26 (n=21) | -0.02 (n=26) | -0.02 (n=33) |
| `atm_iv_60d` | — | — | +0.29 (n=20) | -0.11 (n=23) | -0.09 (n=26) |
| `atm_iv_90d` | — | — | +0.31 (n=20) | -0.13 (n=23) | -0.04 (n=26) |
| `expiry_concentration_top1` | — | — | -0.02 (n=20) | +0.12 (n=23) | -0.10 (n=26) |
| `vrp_proxy` | — | — | +0.28 (n=21) | -0.06 (n=26) | +0.03 (n=33) |
| `atm_iv_30d` | — | — | +0.44 (n=20) | -0.19 (n=23) | -0.07 (n=26) |
| `charm_total` | — | — | -0.01 (n=20) | +0.28 (n=23) | -0.20 (n=26) |
| `term_slope_30_90` | — | — | -0.36 (n=20) | +0.25 (n=23) | -0.05 (n=26) |
| `max_pain_dist_pct` | — | — | +0.03 (n=21) | -0.22 (n=26) | +0.25 (n=33) |
| `vanna_total` | — | — | +0.07 (n=20) | +0.08 (n=23) | +0.08 (n=26) |

## 3. Promotion candidates

| Feature | n | Spearman | OOS Spearman | Action |
| --- | --- | --- | --- | --- |
| `far_otm_put_share` | 52 | +0.352 | +0.418 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `far_otm_call_share` | 52 | -0.195 | +0.057 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `dollar_delta_weighted_flow` | 52 | -0.150 | +0.261 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `sector_relative_pct` | 55 | +0.128 | +0.325 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
