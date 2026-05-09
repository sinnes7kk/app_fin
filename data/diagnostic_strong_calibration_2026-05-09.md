# Flow Tracker — Strong + Early Calibration Sweep (2026-05-09)

Sweep window: **2026-04-17 → 2026-05-07** (15 trading days).
Source: `data/snapshots_archive.csv.gz` (append-only screener history).

Each row evaluates `compute_multi_day_flow` with the candidate's lookback / day-skew floor patched in, then counts how many tickers passed `passes_strong` on each historical `as_of`.

## Strong cohort

| label | LB | skew | min_act | cum$M | bps | cons | dayP | accel_t | grade | days≥1 | avg/d | max/d | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current | 5 | 0.20 | 5 | 25 | 5.0 | 0.30 | 0.60 | 0.5 | 4 | 0/15 | 0.0 | 0 | 0 |
| wide-window-7 | 7 | 0.15 | 5 | 25 | 5.0 | 0.20 | 0.60 | 0.0 | 4 | 0/15 | 0.0 | 0 | 0 |
| wide-window-10 | 10 | 0.10 | 5 | 25 | 5.0 | 0.20 | 0.60 | 0.0 | 4 | 0/15 | 0.0 | 0 | 0 |
| wide-10-low-skew | 10 | 0.08 | 5 | 25 | 5.0 | 0.10 | 0.60 | 0.0 | 4 | 1/15 | 0.1 | 1 | 1 |
| wide-10-min4 | 10 | 0.08 | 4 | 25 | 5.0 | 0.10 | 0.60 | 0.0 | 4 | 8/15 | 0.7 | 2 | 10 |
| wide-10-min4-A_minus | 10 | 0.08 | 4 | 25 | 5.0 | 0.10 | 0.60 | 0.0 | 4 | 8/15 | 0.7 | 2 | 10 |
| wide-10-min3 | 10 | 0.08 | 3 | 25 | 5.0 | 0.10 | 0.60 | 0.0 | 4 | 10/15 | 1.3 | 4 | 20 |
| tight-but-fixed | 10 | 0.08 | 5 | 25 | 5.0 | 0.20 | 0.70 | 0.0 | 4 | 0/15 | 0.0 | 0 | 0 |
| loose-grade | 10 | 0.08 | 4 | 15 | 3.0 | 0.10 | 0.60 | 0.0 | 3 | 8/15 | 0.7 | 2 | 10 |

### Strong — per-day pass counts

| date | current | wide-window-7 | wide-window-10 | wide-10-low-skew | wide-10-min4 | wide-10-min4-A_minus | wide-10-min3 | tight-but-fixed | loose-grade |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-04-17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2026-04-20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2026-04-21 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| 2026-04-22 | 0 | 0 | 0 | 0 | 1 | 1 | 4 | 0 | 1 |
| 2026-04-23 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2026-04-24 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 1 |
| 2026-04-27 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2026-04-28 | 0 | 0 | 0 | 0 | 1 | 1 | 2 | 0 | 1 |
| 2026-04-29 | 0 | 0 | 0 | 0 | 2 | 2 | 3 | 0 | 2 |
| 2026-04-30 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 1 |
| 2026-05-01 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 1 |
| 2026-05-04 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 1 |
| 2026-05-05 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2026-05-06 | 0 | 0 | 0 | 0 | 2 | 2 | 3 | 0 | 2 |
| 2026-05-07 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |

### Strong — top survivors per candidate

**wide-10-low-skew** (1 hits across window):
- POET (BULLISH): 1 day(s)

**wide-10-min4** (10 hits across window):
- WULF (BEARISH): 2 day(s)
- MUSA (BULLISH): 2 day(s)
- SATS (BEARISH): 1 day(s)
- POET (BULLISH): 1 day(s)
- FSLR (BULLISH): 1 day(s)
- AXSM (BULLISH): 1 day(s)
- PTCT (BEARISH): 1 day(s)
- QCOM (BULLISH): 1 day(s)

**wide-10-min4-A_minus** (10 hits across window):
- WULF (BEARISH): 2 day(s)
- MUSA (BULLISH): 2 day(s)
- SATS (BEARISH): 1 day(s)
- POET (BULLISH): 1 day(s)
- FSLR (BULLISH): 1 day(s)
- AXSM (BULLISH): 1 day(s)
- PTCT (BEARISH): 1 day(s)
- QCOM (BULLISH): 1 day(s)

**wide-10-min3** (20 hits across window):
- BLD (BEARISH): 2 day(s)
- GPC (BEARISH): 2 day(s)
- WULF (BEARISH): 2 day(s)
- FSLR (BULLISH): 2 day(s)
- MUSA (BULLISH): 2 day(s)
- RDDT (BEARISH): 1 day(s)
- MRVL (BULLISH): 1 day(s)
- SATS (BEARISH): 1 day(s)

**loose-grade** (10 hits across window):
- WULF (BEARISH): 2 day(s)
- MUSA (BULLISH): 2 day(s)
- SATS (BEARISH): 1 day(s)
- POET (BULLISH): 1 day(s)
- FSLR (BULLISH): 1 day(s)
- AXSM (BULLISH): 1 day(s)
- PTCT (BEARISH): 1 day(s)
- QCOM (BULLISH): 1 day(s)

## Early cohort (2-day flow)

Same gate, but `min_active_days = 2` and (for most candidates) `min_day_persistence = 1.00` + `require_no_flips = True` — i.e. **both** active days lean the same direction. The lookback is 4 calendar days so the window holds ~2-3 trading days plus today, keeping the 2-day confirmation tight.

| label | LB | skew | min_act | cum$M | bps | cons | dayP | accel_t | grade | days≥1 | avg/d | max/d | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| early-strict | 4 | 0.20 | 2 | 10 | 3.0 | 0.30 | 1.00 | -99.0 | 3 | 14/15 | 3.8 | 8 | 57 |
| early-mid | 4 | 0.10 | 2 | 5 | 2.0 | 0.10 | 1.00 | -99.0 | 3 | 14/15 | 6.4 | 9 | 96 |
| early-loose | 4 | 0.08 | 2 | 5 | 2.0 | 0.05 | 1.00 | -99.0 | 2 | 14/15 | 6.6 | 9 | 99 |
| early-permissive | 4 | 0.08 | 2 | 3 | 1.5 | 0.05 | 1.00 | -99.0 | 0 | 14/15 | 7.9 | 11 | 119 |
| early-no-flip | 4 | 0.10 | 2 | 5 | 2.0 | 0.05 | 0.50 | -99.0 | 2 | 14/15 | 8.4 | 11 | 126 |

### Early — per-day pass counts

| date | early-strict | early-mid | early-loose | early-permissive | early-no-flip |
|---|---:|---:|---:|---:|---:|
| 2026-04-17 | 0 | 0 | 0 | 0 | 0 |
| 2026-04-20 | 1 | 3 | 4 | 4 | 9 |
| 2026-04-21 | 4 | 5 | 6 | 6 | 8 |
| 2026-04-22 | 2 | 3 | 4 | 5 | 8 |
| 2026-04-23 | 4 | 7 | 7 | 9 | 8 |
| 2026-04-24 | 3 | 6 | 6 | 8 | 8 |
| 2026-04-27 | 3 | 7 | 7 | 11 | 8 |
| 2026-04-28 | 3 | 8 | 8 | 11 | 11 |
| 2026-04-29 | 4 | 9 | 9 | 11 | 11 |
| 2026-04-30 | 5 | 9 | 9 | 10 | 9 |
| 2026-05-01 | 3 | 6 | 6 | 7 | 7 |
| 2026-05-04 | 4 | 9 | 9 | 10 | 10 |
| 2026-05-05 | 6 | 9 | 9 | 10 | 9 |
| 2026-05-06 | 8 | 8 | 8 | 9 | 11 |
| 2026-05-07 | 7 | 7 | 7 | 8 | 9 |

### Early — top survivors per candidate

**early-strict** (57 hits across window):
- RRC (BULLISH): 4 day(s)
- PAGS (BEARISH): 4 day(s)
- GPC (BEARISH): 3 day(s)
- PCG (BEARISH): 3 day(s)
- DRVN (BULLISH): 3 day(s)
- MUSA (BULLISH): 3 day(s)
- EFX (BEARISH): 3 day(s)
- IMVT (BEARISH): 3 day(s)

**early-mid** (96 hits across window):
- RRC (BULLISH): 4 day(s)
- PAGS (BEARISH): 4 day(s)
- DOCS (BULLISH): 3 day(s)
- GPC (BEARISH): 3 day(s)
- AUPH (BEARISH): 3 day(s)
- SNDX (BEARISH): 3 day(s)
- PCG (BEARISH): 3 day(s)
- DRVN (BULLISH): 3 day(s)

**early-loose** (99 hits across window):
- BLD (BEARISH): 5 day(s)
- RRC (BULLISH): 4 day(s)
- PAGS (BEARISH): 4 day(s)
- DOCS (BULLISH): 3 day(s)
- GPC (BEARISH): 3 day(s)
- AUPH (BEARISH): 3 day(s)
- SNDX (BEARISH): 3 day(s)
- PCG (BEARISH): 3 day(s)

**early-permissive** (119 hits across window):
- BLD (BEARISH): 5 day(s)
- RRC (BULLISH): 4 day(s)
- GHM (BULLISH): 4 day(s)
- PAGS (BEARISH): 4 day(s)
- IMVT (BEARISH): 4 day(s)
- DOCS (BULLISH): 3 day(s)
- GPC (BEARISH): 3 day(s)
- AUPH (BEARISH): 3 day(s)

**early-no-flip** (126 hits across window):
- BLD (BEARISH): 5 day(s)
- RRC (BULLISH): 4 day(s)
- PAGS (BEARISH): 4 day(s)
- DOCS (BULLISH): 3 day(s)
- GPC (BEARISH): 3 day(s)
- AUPH (BEARISH): 3 day(s)
- SNDX (BEARISH): 3 day(s)
- PCG (BEARISH): 3 day(s)
