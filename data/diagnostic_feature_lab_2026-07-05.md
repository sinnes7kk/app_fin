# Feature lab — Spearman ranking — 2026-07-05 23:02

Joined `feature_lab.csv` × `grade_history_with_replay.csv` on (as_of, ticker, direction).  Panel size: **572 rows** (after dropping rows without realized_r).

Spearman is a rank correlation between each candidate feature and the bar-by-bar replay `realized_r`. Features with consistent |Spearman| ≥ 0.10 across multiple DTE buckets and a positive walk-forward OOS Spearman are promotion candidates. Features with consistently *negative* Spearman are candidates for sign inversion.

**Caveat:** until the panel reaches ~250 closed-and-replayed rows any single ranking is dominated by sampling noise. Treat this as a watchlist of hypotheses, not a hit list of fixes.

---

## 1. Overall ranking

| Feature | n | Spearman | OOS Spearman | n_val |
| --- | --- | --- | --- | --- |
| `prem_momentum_z3d` | 26 | +0.588 | +0.826 | 11 |
| `bullish_premium_share` | 255 | +0.166 | +0.348 | 102 |
| `sector_relative_pct` | 158 | +0.149 | +0.117 | 64 |
| `far_otm_call_share` | 175 | -0.132 | -0.086 | 70 |
| `iv_skew_25d` | 35 | -0.119 | -0.227 | 14 |
| `charm_total` | 255 | +0.117 | -0.014 | 102 |
| `dollar_delta_weighted_flow` | 175 | +0.110 | +0.239 | 70 |
| `realized_vol_regime` | 252 | -0.099 | -0.045 | 101 |
| `dealer_net_gamma_at_spot` | 88 | +0.095 | +0.194 | 36 |
| `vrp_proxy` | 252 | +0.070 | +0.157 | 101 |
| `expiry_concentration_top1` | 254 | +0.067 | +0.091 | 102 |
| `dealer_net_delta_at_spot` | 88 | -0.066 | +0.150 | 36 |
| `far_otm_put_share` | 175 | +0.062 | -0.162 | 70 |
| `unusual_premium_share` | 112 | +0.057 | +0.086 | 45 |
| `term_slope_30_90` | 255 | +0.030 | -0.027 | 102 |
| `gex_total` | 255 | -0.025 | +0.083 | 102 |
| `atm_iv_60d` | 255 | -0.012 | -0.052 | 102 |
| `atm_iv_30d` | 255 | -0.009 | -0.029 | 102 |
| `atm_iv_90d` | 255 | -0.007 | -0.057 | 102 |
| `vanna_total` | 255 | -0.002 | -0.003 | 102 |
| `max_pain_dist_pct` | 267 | +0.002 | -0.144 | 107 |

## 2. Per-DTE-bucket breakdown

| Feature | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `prem_momentum_z3d` | — | — | +0.00 (n=8) | +0.28 (n=9) | +0.84 (n=6) |
| `bullish_premium_share` | — | +0.35 (n=11) | -0.05 (n=92) | +0.24 (n=68) | +0.26 (n=81) |
| `sector_relative_pct` | — | +0.50 (n=5) | +0.22 (n=61) | +0.12 (n=47) | +0.07 (n=43) |
| `far_otm_call_share` | — | -0.60 (n=6) | -0.12 (n=71) | -0.15 (n=52) | -0.17 (n=44) |
| `iv_skew_25d` | — | — | -0.04 (n=18) | +0.16 (n=10) | +0.26 (n=6) |
| `charm_total` | — | -0.17 (n=11) | +0.23 (n=92) | +0.20 (n=68) | -0.09 (n=81) |
| `dollar_delta_weighted_flow` | — | +0.37 (n=6) | +0.09 (n=71) | +0.21 (n=52) | +0.01 (n=44) |
| `realized_vol_regime` | — | +0.35 (n=12) | -0.17 (n=84) | -0.07 (n=69) | -0.12 (n=84) |
| `dealer_net_gamma_at_spot` | — | — | +0.22 (n=54) | +0.21 (n=25) | — |
| `vrp_proxy` | — | +0.59 (n=12) | +0.14 (n=84) | +0.06 (n=69) | -0.03 (n=84) |
| `expiry_concentration_top1` | — | -0.42 (n=11) | +0.08 (n=92) | +0.33 (n=68) | -0.08 (n=80) |
| `dealer_net_delta_at_spot` | — | — | +0.07 (n=54) | -0.05 (n=25) | — |
| `far_otm_put_share` | — | +0.17 (n=6) | +0.16 (n=71) | -0.11 (n=52) | +0.22 (n=44) |
| `unusual_premium_share` | — | — | -0.01 (n=71) | +0.10 (n=37) | — |
| `term_slope_30_90` | — | +0.00 (n=11) | -0.17 (n=92) | +0.19 (n=68) | +0.08 (n=81) |
| `gex_total` | — | +0.08 (n=11) | -0.09 (n=92) | -0.03 (n=68) | +0.04 (n=81) |
| `atm_iv_60d` | — | +0.10 (n=11) | -0.01 (n=92) | -0.17 (n=68) | +0.06 (n=81) |
| `atm_iv_30d` | — | +0.06 (n=11) | -0.00 (n=92) | -0.18 (n=68) | +0.09 (n=81) |
| `atm_iv_90d` | — | +0.03 (n=11) | -0.01 (n=92) | -0.14 (n=68) | +0.07 (n=81) |
| `vanna_total` | — | +0.37 (n=11) | -0.06 (n=92) | -0.07 (n=68) | +0.03 (n=81) |
| `max_pain_dist_pct` | — | +0.30 (n=12) | +0.08 (n=93) | -0.02 (n=71) | -0.05 (n=88) |

## 3. Promotion candidates

| Feature | n | Spearman | OOS Spearman | Action |
| --- | --- | --- | --- | --- |
| `bullish_premium_share` | 255 | +0.166 | +0.348 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `sector_relative_pct` | 158 | +0.149 | +0.117 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
| `dollar_delta_weighted_flow` | 175 | +0.110 | +0.239 | **candidate** — review for inclusion in conviction_score via NNLS recalibration |
