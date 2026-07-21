# Unified feature sweep — 2026-07-21 10:58

Panel: **161 rows** with a bar-by-bar `replay_realized_r`, joined `grade_history_with_replay.csv` × `feature_lab.csv` on (as_of, ticker, direction).

`spearman` = pooled rank IC vs realized R (in-sample). `oos` = chronological 60/40 walk-forward rank IC. `r_spread` = mean realized R of top tercile − bottom tercile (the $ edge, in R). Sorted by sign-agreeing OOS IC so in-sample-only flukes sink.

**Caveat:** one bull-market regime, small OOS slices. Treat this as a hypothesis watchlist, not a hit list. A feature needs |IC| that holds OOS across fresh weeks before it earns a place in a live score.

---

## Full ranking (all scorers, one target)

| Feature | Family | Live? | n | Spearman | p | OOS | n_val | R-spread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `perc_3_day_total_latest` | flow-tracker (unused) | — | 161 | +0.221 | 0.005 | +0.247 | 65 | +0.88 |
| `bullish_premium_share` | feature-lab | shadow | 131 | +0.144 | 0.102 | +0.243 | 53 | +0.33 |
| `momentum_score` | composite | shadow | 131 | +0.117 | 0.184 | +0.226 | 53 | +0.15 |
| `momentum_composite` | composite | shadow | 131 | +0.117 | 0.184 | +0.226 | 53 | +0.15 |
| `aggressor_bull_share` | aggressor | shadow | 93 | +0.025 | 0.810 | +0.186 | 38 | +0.01 |
| `vrp_proxy` | feature-lab | shadow | 134 | +0.069 | 0.428 | +0.170 | 54 | +0.05 |
| `aggressor_net_prem_bps` | aggressor | shadow | 95 | +0.020 | 0.848 | +0.164 | 38 | +0.00 |
| `dollar_delta_weighted_flow` | feature-lab | shadow | 98 | +0.180 | 0.077 | +0.162 | 40 | +0.55 |
| `gex_total` | UW options | shadow | 131 | +0.037 | 0.675 | +0.131 | 53 | +0.08 |
| `expiry_concentration_top1` | UW options | shadow | 130 | +0.097 | 0.270 | +0.122 | 52 | +0.48 |
| `charm_total` | UW options | shadow | 131 | +0.102 | 0.243 | +0.113 | 53 | +0.18 |
| `ask_side_ratio` | aggressor | shadow | 93 | +0.062 | 0.551 | +0.090 | 38 | +0.08 |
| `prem_mcap_bps` | conviction_score component | LIVE | 161 | +0.053 | 0.506 | +0.081 | 65 | +0.15 |
| `flow_intensity` | conviction_score + final_score | LIVE | 161 | +0.053 | 0.506 | +0.081 | 65 | +0.15 |
| `sweep_share` | flow-tracker (unused) | — | 161 | +0.052 | 0.511 | +0.066 | 65 | +0.03 |
| `cumulative_premium` | conviction_score component | LIVE | 161 | +0.042 | 0.596 | +0.064 | 65 | +0.12 |
| `directional_sweep_share` | aggressor | shadow | 95 | +0.051 | 0.622 | +0.048 | 38 | +0.13 |
| `accumulation_score` | flow-tracker (unused) | — | 161 | +0.025 | 0.753 | +0.046 | 65 | +0.05 |
| `latest_iv_rank` | flow-tracker (unused) | — | 161 | -0.055 | 0.489 | +0.066 | 65 | +0.03 |
| `dealer_net_delta_at_spot` | UW options | shadow | 43 | -0.035 | 0.820 | +0.149 | 18 | -0.83 |
| `realized_vol_regime` | feature-lab | shadow | 134 | -0.100 | 0.250 | +0.041 | 54 | -0.31 |
| `multileg_share` | flow-tracker (unused) | — | 161 | -0.030 | 0.702 | +0.087 | 65 | -0.03 |
| `accel_ratio_today` | flow-tracker (unused) | — | 161 | -0.006 | 0.942 | -0.010 | 65 | -0.01 |
| `sector_relative_pct` | feature-lab | shadow | 89 | +0.094 | 0.376 | -0.010 | 36 | +0.32 |
| `window_return_pct` | flow-tracker (unused) | — | 161 | +0.082 | 0.301 | -0.028 | 65 | +0.29 |
| `term_slope_30_90` | UW options | shadow | 131 | -0.008 | 0.927 | -0.041 | 53 | -0.33 |
| `prem_momentum_z3d` | feature-lab | shadow | 109 | +0.043 | 0.655 | -0.050 | 44 | +0.25 |
| `dealer_net_gamma_at_spot` | UW options | shadow | 43 | +0.061 | 0.691 | -0.089 | 18 | -0.01 |
| `latest_oi_change` | conviction_score component | LIVE | 161 | +0.057 | 0.474 | -0.094 | 65 | +0.07 |
| `conviction_score` | composite | LIVE grade | 161 | -0.016 | 0.843 | -0.100 | 65 | -0.01 |
| `latest_put_call_ratio` | flow-tracker (unused) | — | 161 | -0.074 | 0.348 | -0.116 | 65 | -0.49 |
| `vanna_total` | UW options | shadow | 131 | +0.024 | 0.786 | -0.120 | 53 | +0.22 |
| `atm_iv_30d` | UW options | shadow | 131 | -0.012 | 0.890 | -0.126 | 53 | +0.03 |
| `persistence_ratio` | conviction_score component | LIVE | 161 | -0.030 | 0.702 | -0.142 | 65 | -0.18 |
| `max_pain_dist_pct` | UW options | shadow | 137 | -0.004 | 0.964 | -0.144 | 55 | +0.37 |
| `atm_iv_60d` | UW options | shadow | 131 | -0.027 | 0.757 | -0.147 | 53 | -0.07 |
| `far_otm_call_share` | feature-lab | shadow | 98 | -0.186 | 0.067 | -0.155 | 40 | -0.61 |
| `atm_iv_90d` | UW options | shadow | 131 | -0.030 | 0.736 | -0.167 | 53 | -0.02 |
| `unusual_premium_share` | feature-lab | shadow | 60 | +0.006 | 0.966 | -0.195 | 24 | +0.24 |
| `far_otm_put_share` | feature-lab | shadow | 98 | +0.042 | 0.676 | -0.277 | 40 | +0.17 |

## Passes a minimal bar (n≥40, |Spearman|≥0.10, OOS same sign)

- `perc_3_day_total_latest` (flow-tracker (unused), —): Spearman +0.221, OOS +0.247, R-spread +0.88, n=161
- `bullish_premium_share` (feature-lab, shadow): Spearman +0.144, OOS +0.243, R-spread +0.33, n=131
- `momentum_score` (composite, shadow): Spearman +0.117, OOS +0.226, R-spread +0.15, n=131
- `momentum_composite` (composite, shadow): Spearman +0.117, OOS +0.226, R-spread +0.15, n=131
- `dollar_delta_weighted_flow` (feature-lab, shadow): Spearman +0.180, OOS +0.162, R-spread +0.55, n=98
- `charm_total` (UW options, shadow): Spearman +0.102, OOS +0.113, R-spread +0.18, n=131
- `far_otm_call_share` (feature-lab, shadow): Spearman -0.186, OOS -0.155, R-spread -0.61, n=98
