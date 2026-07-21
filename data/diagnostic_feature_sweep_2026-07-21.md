# Unified feature sweep — 2026-07-21 10:06

Panel: **338 rows** with a bar-by-bar `replay_realized_r`, joined `grade_history_with_replay.csv` × `feature_lab.csv` on (as_of, ticker, direction).

`spearman` = pooled rank IC vs realized R (in-sample). `oos` = chronological 60/40 walk-forward rank IC. `r_spread` = mean realized R of top tercile − bottom tercile (the $ edge, in R). Sorted by sign-agreeing OOS IC so in-sample-only flukes sink.

**Caveat:** one bull-market regime, small OOS slices. Treat this as a hypothesis watchlist, not a hit list. A feature needs |IC| that holds OOS across fresh weeks before it earns a place in a live score.

---

## Full ranking (all scorers, one target)

| Feature | Family | Live? | n | Spearman | p | OOS | n_val | R-spread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bullish_premium_share` | feature-lab | shadow | 283 | +0.160 | 0.007 | +0.309 | 114 | +0.41 |
| `aggressor_bull_share` | aggressor | shadow | 202 | +0.025 | 0.718 | +0.246 | 81 | +0.08 |
| `multileg_share` | flow-tracker (unused) | — | 338 | +0.015 | 0.789 | +0.180 | 136 | -0.01 |
| `dollar_delta_weighted_flow` | feature-lab | shadow | 195 | +0.088 | 0.222 | +0.159 | 78 | +0.28 |
| `expiry_concentration_top1` | UW options | shadow | 282 | +0.096 | 0.107 | +0.158 | 113 | +0.33 |
| `unusual_premium_share` | feature-lab | shadow | 127 | +0.031 | 0.727 | +0.133 | 51 | +0.06 |
| `dealer_net_gamma_at_spot` | UW options | shadow | 95 | +0.075 | 0.465 | +0.087 | 38 | +0.36 |
| `term_slope_30_90` | UW options | shadow | 283 | +0.057 | 0.335 | +0.074 | 114 | +0.14 |
| `vanna_total` | UW options | shadow | 283 | +0.003 | 0.959 | +0.073 | 114 | +0.09 |
| `perc_3_day_total_latest` | flow-tracker (unused) | — | 338 | +0.155 | 0.004 | +0.073 | 136 | +0.53 |
| `flow_intensity` | conviction_score + final_score | LIVE | 338 | +0.047 | 0.390 | +0.072 | 136 | +0.10 |
| `prem_mcap_bps` | conviction_score component | LIVE | 338 | +0.047 | 0.390 | +0.072 | 136 | +0.10 |
| `sector_relative_pct` | feature-lab | shadow | 167 | +0.140 | 0.072 | +0.071 | 67 | +0.46 |
| `ask_side_ratio` | aggressor | shadow | 202 | +0.003 | 0.965 | +0.058 | 81 | +0.01 |
| `vrp_proxy` | feature-lab | shadow | 276 | +0.045 | 0.457 | +0.040 | 111 | +0.05 |
| `atm_iv_60d` | UW options | shadow | 283 | +0.003 | 0.963 | +0.040 | 114 | +0.11 |
| `atm_iv_90d` | UW options | shadow | 283 | +0.007 | 0.907 | +0.037 | 114 | +0.12 |
| `accumulation_score` | flow-tracker (unused) | — | 338 | +0.064 | 0.241 | +0.033 | 136 | +0.10 |
| `atm_iv_30d` | UW options | shadow | 283 | -0.004 | 0.944 | +0.047 | 114 | +0.10 |
| `dealer_net_delta_at_spot` | UW options | shadow | 95 | -0.028 | 0.787 | +0.188 | 38 | -0.30 |
| `aggressor_net_prem_bps` | aggressor | shadow | 204 | -0.019 | 0.789 | +0.075 | 82 | +0.06 |
| `directional_sweep_share` | aggressor | shadow | 204 | -0.010 | 0.883 | +0.044 | 82 | -0.06 |
| `gex_total` | UW options | shadow | 283 | -0.028 | 0.636 | +0.086 | 114 | -0.05 |
| `cumulative_premium` | conviction_score component | LIVE | 338 | -0.018 | 0.744 | -0.009 | 136 | -0.08 |
| `window_return_pct` | flow-tracker (unused) | — | 338 | +0.075 | 0.168 | -0.010 | 136 | +0.27 |
| `accel_ratio_today` | flow-tracker (unused) | — | 338 | +0.022 | 0.682 | -0.013 | 136 | +0.00 |
| `sweep_share` | flow-tracker (unused) | — | 338 | +0.045 | 0.405 | -0.021 | 136 | +0.02 |
| `prem_momentum_z3d` | feature-lab | shadow | 225 | -0.001 | 0.985 | -0.021 | 90 | -0.09 |
| `momentum_score` | composite | shadow | 275 | +0.030 | 0.618 | -0.025 | 110 | -0.00 |
| `momentum_composite` | composite | shadow | 275 | +0.030 | 0.618 | -0.025 | 110 | -0.00 |
| `persistence_ratio` | conviction_score component | LIVE | 338 | +0.023 | 0.674 | -0.029 | 136 | +0.01 |
| `latest_oi_change` | conviction_score component | LIVE | 338 | +0.042 | 0.436 | -0.031 | 136 | +0.03 |
| `far_otm_call_share` | feature-lab | shadow | 195 | -0.104 | 0.148 | -0.044 | 78 | -0.33 |
| `latest_iv_rank` | flow-tracker (unused) | — | 338 | -0.033 | 0.540 | -0.044 | 136 | +0.04 |
| `realized_vol_regime` | feature-lab | shadow | 276 | -0.097 | 0.109 | -0.061 | 111 | -0.27 |
| `conviction_score` | composite | LIVE grade | 338 | +0.055 | 0.313 | -0.063 | 136 | +0.17 |
| `latest_put_call_ratio` | flow-tracker (unused) | — | 338 | -0.042 | 0.439 | -0.085 | 136 | -0.23 |
| `far_otm_put_share` | feature-lab | shadow | 195 | +0.071 | 0.324 | -0.090 | 78 | +0.13 |
| `charm_total` | UW options | shadow | 283 | +0.103 | 0.083 | -0.099 | 114 | +0.29 |
| `max_pain_dist_pct` | UW options | shadow | 295 | -0.002 | 0.967 | -0.178 | 118 | +0.19 |
| `iv_skew_25d` | UW options | shadow | 36 | -0.119 | 0.482 | -0.191 | 15 | -0.32 |

## Passes a minimal bar (n≥40, |Spearman|≥0.10, OOS same sign)

- `bullish_premium_share` (feature-lab, shadow): Spearman +0.160, OOS +0.309, R-spread +0.41, n=283
- `perc_3_day_total_latest` (flow-tracker (unused), —): Spearman +0.155, OOS +0.073, R-spread +0.53, n=338
- `sector_relative_pct` (feature-lab, shadow): Spearman +0.140, OOS +0.071, R-spread +0.46, n=167
- `far_otm_call_share` (feature-lab, shadow): Spearman -0.104, OOS -0.044, R-spread -0.33, n=195
