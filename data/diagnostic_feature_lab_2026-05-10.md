# Feature lab — Spearman ranking — 2026-05-10 22:51

Joined `feature_lab.csv` × `grade_history_with_replay.csv` on (as_of, ticker, direction).  Panel size: **6 rows** (after dropping rows without realized_r).

Spearman is a rank correlation between each candidate feature and the bar-by-bar replay `realized_r`. Features with consistent |Spearman| ≥ 0.10 across multiple DTE buckets and a positive walk-forward OOS Spearman are promotion candidates. Features with consistently *negative* Spearman are candidates for sign inversion.

**Caveat:** until the panel reaches ~250 closed-and-replayed rows any single ranking is dominated by sampling noise. Treat this as a watchlist of hypotheses, not a hit list of fixes.

---

## 1. Overall ranking

| Feature | n | Spearman | OOS Spearman | n_val |
| --- | --- | --- | --- | --- |
| `bullish_premium_share` | 0 | — | — | 0 |
| `unusual_premium_share` | 0 | — | — | 0 |
| `vrp_proxy` | 0 | — | — | 0 |
| `far_otm_call_share` | 0 | — | — | 0 |
| `far_otm_put_share` | 0 | — | — | 0 |
| `dollar_delta_weighted_flow` | 0 | — | — | 0 |
| `sector_relative_pct` | 0 | — | — | 0 |
| `prem_momentum_z3d` | 0 | — | — | 0 |
| `realized_vol_regime` | 0 | — | — | 0 |
| `gex_total` | 0 | — | — | 0 |
| `vanna_total` | 0 | — | — | 0 |
| `charm_total` | 0 | — | — | 0 |
| `iv_skew_25d` | 0 | — | — | 0 |
| `atm_iv_30d` | 0 | — | — | 0 |
| `atm_iv_60d` | 0 | — | — | 0 |
| `atm_iv_90d` | 0 | — | — | 0 |
| `term_slope_30_90` | 0 | — | — | 0 |
| `expiry_concentration_top1` | 0 | — | — | 0 |
| `max_pain_dist_pct` | 0 | — | — | 0 |
| `dealer_net_delta_at_spot` | 0 | — | — | 0 |
| `dealer_net_gamma_at_spot` | 0 | — | — | 0 |

## 2. Per-DTE-bucket breakdown

| Feature | lottery | swing | position | leap | unknown |
| --- | --- | --- | --- | --- | --- |
| `bullish_premium_share` | — | — | — | — | — |
| `unusual_premium_share` | — | — | — | — | — |
| `vrp_proxy` | — | — | — | — | — |
| `far_otm_call_share` | — | — | — | — | — |
| `far_otm_put_share` | — | — | — | — | — |
| `dollar_delta_weighted_flow` | — | — | — | — | — |
| `sector_relative_pct` | — | — | — | — | — |
| `prem_momentum_z3d` | — | — | — | — | — |
| `realized_vol_regime` | — | — | — | — | — |
| `gex_total` | — | — | — | — | — |
| `vanna_total` | — | — | — | — | — |
| `charm_total` | — | — | — | — | — |
| `iv_skew_25d` | — | — | — | — | — |
| `atm_iv_30d` | — | — | — | — | — |
| `atm_iv_60d` | — | — | — | — | — |
| `atm_iv_90d` | — | — | — | — | — |
| `term_slope_30_90` | — | — | — | — | — |
| `expiry_concentration_top1` | — | — | — | — | — |
| `max_pain_dist_pct` | — | — | — | — | — |
| `dealer_net_delta_at_spot` | — | — | — | — | — |
| `dealer_net_gamma_at_spot` | — | — | — | — | — |

## 3. Promotion candidates

_None yet — none of the candidate features clear the bar (n≥30, |Spearman|≥0.10, OOS≥0). Keep collecting._
