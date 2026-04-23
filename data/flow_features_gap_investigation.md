# `data/flow_features/` gap investigation — 2026-03-28 through 2026-04-13

## Summary

Between 2026-03-28 and 2026-04-13 (inclusive), no new `flow_features_*.csv`
snapshots were committed to the repo. The gap is 17 calendar days / ~12
trading days.

## Root cause (already fixed)

Commit `91c34ca` (Apr 14 16:23 CEST) — *"fix: scan data not committing since
Mar 28"* — identified and patched the bug:

> `git add` with all paths in one command fails fatally when any path is
> missing (e.g. `trade_log_agent.csv` on first run). The `|| true` swallowed
> the error but nothing got staged, so every scan since the agent portfolio
> commit silently skipped the data commit step.

The fix switched the workflow to a per-path for-loop with a pre-existence
check (`[ -e "$f" ] && git add "$f" || true`). This is what
`.github/workflows/hourly_scan.yml` does today.

In other words: the hourly scans between Mar 28 and Apr 13 **did** run, and
the pipeline **did** produce `flow_features_*.csv` rows in the ephemeral
GitHub Actions runner. None of those files were ever staged, committed, or
pushed — the runners were torn down with the data still in them.

## Is the data recoverable?

**No** for the flagged-flow CSVs. UW's public endpoints don't expose the
individual sweep-level alert prints retroactively at the granularity we'd
need to rebuild those snapshots.

**Yes, implicitly** for the z-score baselines that matter for the
widen-unusual-zscore-baseline plan:

- `flow_intensity`, `vol_oi`, and `unusual_premium_share` baselines are now
  hydrated by `load_uw_baselines` from `/stock/{ticker}/options-volume`,
  which returns a rolling 30-day window of daily per-ticker aggregates.
  That endpoint is independent of our internal CSV store, so the gap is
  effectively invisible for those three components.

## Components still affected

The internal `flow_stats.py` tier 1/2/3 ladder uses `data/flow_features/*.csv`
for components **not** covered by UW options-volume:

- `premium_per_trade` (`*_ppt_bps`)
- `repeat` (`*_repeat_count`)
- `sweep` (`*_sweep_count`)
- `breadth` (`*_breadth`)
- `dte` (`dte_score`)

These will show elevated Tier 4 (absolute fallback) counts until the
internal store accumulates enough post-Apr-14 history. Expect natural
tier-1 coverage for these components to climb as ~12 more trading days
pass post-fix.

## Actions

1. **None for the extended plan** — the new z-score components route around
   the gap by design.
2. `data/zscore_coverage_history.json` has been added to the CI commit
   loop so future gaps are immediately visible in per-run tier distribution
   tracking.
3. No one-off backfill script is recommended: the missing data genuinely
   cannot be reconstructed at the same fidelity.
