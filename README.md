# app_fin

Financial flow and signals application.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Layout

- `data/` — raw flow, aggregates, and signal outputs
- `app/` — application code (vendors, features, ranking, rules, signals, jobs)
- `tests/` — tests


#V1 Technical Spec
Project

Swing Continuation Flow Scanner

Objective

Build a daily system that:

ingests new options flow

filters for higher-signal institutional positioning

aggregates/ranks tickers

checks daily price-based continuation rules

outputs a daily set of trade candidates

This is not an execution algo yet.
V1 is a scanner + signal generator.

1. Scope
Included in V1

daily options-flow ingestion

local storage of raw flow

ticker-level daily feature generation

candidate ranking

daily candle-based rule engine

signal output

backtest-friendly signal/history logging

Excluded from V1

intraday logic

real-time streaming

brokerage integration

auto-order execution

reversal/bottoming setups

advanced ML

portfolio optimization beyond simple limits

options pricing engine / Greeks-based execution engine

2. Core Functional Requirements
FR1 — Flow ingestion

The system must ingest all new options flow events since the last successful run.

FR2 — Flow normalization

The system must map vendor-specific fields into an internal canonical schema.

FR3 — Raw storage

The system must store raw normalized flow events for reprocessing and auditability.

FR4 — Daily feature generation

The system must compute ticker-level features over rolling windows like:

1 day

3 day

optionally 5 day

FR5 — Candidate ranking

The system must produce a ranked bullish and bearish candidate list.

FR6 — Price-rule evaluation

The system must pull daily OHLCV data and evaluate:

trend

pullback location

support/resistance hold

confirmation candle

FR7 — Signal generation

The system must output a final signal object for each qualified setup.

FR8 — Persistence

The system must persist:

raw flow

daily aggregates

ranked candidates

final signals

run logs/checkpoints

3. High-Level Architecture
                   ┌─────────────────────┐
                   │ Unusual Whales API  │
                   └─────────┬───────────┘
                             │
                             ▼
                   ┌─────────────────────┐
                   │ Flow Ingestion Job  │
                   └─────────┬───────────┘
                             │
                             ▼
                   ┌─────────────────────┐
                   │ Normalize + Dedupe  │
                   └─────────┬───────────┘
                             │
                             ▼
                   ┌─────────────────────┐
                   │ Raw Flow Storage    │
                   └─────────┬───────────┘
                             │
                             ▼
                   ┌─────────────────────┐
                   │ Feature Builder     │
                   │ (1d / 3d windows)   │
                   └─────────┬───────────┘
                             │
                             ▼
                   ┌─────────────────────┐
                   │ Candidate Ranker    │
                   └─────────┬───────────┘
                             │
                             ▼
                   ┌─────────────────────┐
                   │ Yahoo Finance OHLCV │
                   └─────────┬───────────┘
                             │
                             ▼
                   ┌─────────────────────┐
                   │ Rules Engine        │
                   └─────────┬───────────┘
                             │
                             ▼
                   ┌─────────────────────┐
                   │ Final Signals       │
                   └─────────────────────┘
4. Data Sources
4.1 Options Flow Source

Primary vendor: Unusual Whales

Required vendor fields

At minimum:

ticker

timestamp

option type (call/put)

strike

expiration date

premium

contracts/size

execution side (bid/ask/mid if available)

execution type (sweep/block/split)

volume

open interest

unique event/trade ID if available

4.2 Price Source

V1 source: Yahoo Finance

Required price fields

Daily OHLCV:

timestamp/date

open

high

low

close

volume

5. Canonical Internal Data Model
5.1 FlowEvent

Normalized representation of one options-flow event.

class FlowEvent:
    event_id: str
    vendor: str
    ticker: str
    event_ts: datetime
    option_type: str           # CALL / PUT
    strike: float
    expiration_date: date
    dte: int
    premium: float
    contracts: int | None
    execution_side: str | None # ASK / BID / MID / UNKNOWN
    execution_type: str | None # SWEEP / BLOCK / SPLIT / OTHER
    volume: int | None
    open_interest: int | None
    iv: float | None
    raw_payload: dict
5.2 TickerFlowAggregate

Per ticker, per trade date, per direction/window.

class TickerFlowAggregate:
    trade_date: date
    ticker: str
    direction: str             # LONG / SHORT
    window: str                # 1D / 3D / 5D
    total_premium: float
    event_count: int
    repeat_flow_count: int
    avg_dte_weighted: float | None
    avg_volume_oi_ratio: float | None
    dominant_execution_type: str | None
    call_premium: float
    put_premium: float
    flow_bias: float
    score_raw: float
5.3 PriceFeatureSet

Derived from daily OHLCV.

class PriceFeatureSet:
    trade_date: date
    ticker: str
    close: float
    ema20: float
    ema50: float
    atr14: float
    trend_direction: str       # LONG / SHORT / NEUTRAL
    structure_state: str       # UPTREND / DOWNTREND / MIXED
    support_level: float | None
    resistance_level: float | None
    pullback_valid: bool
    support_hold_valid: bool
    confirmation_valid: bool
5.4 TradeSignal

Final output object.

class TradeSignal:
    signal_id: str
    trade_date: date
    ticker: str
    direction: str             # LONG / SHORT
    candidate_rank: int
    conviction_score: float
    entry_type: str            # SWING_CONTINUATION
    entry_reference: str       # daily_close / next_open / watchlist
    entry_zone_low: float | None
    entry_zone_high: float | None
    stop_price: float | None
    target_1: float | None
    target_2: float | None
    time_stop_days: int | None
    reasons: list[str]
    flow_snapshot: dict
    price_snapshot: dict
6. Storage Design
Recommended V1 approach

Raw flow: Parquet files partitioned by date

Aggregates / signals / checkpoints: SQLite or Postgres

6.1 Raw flow storage

Path example:

data/raw_flow/vendor=uw/date=YYYY-MM-DD/*.parquet
6.2 Database tables
ingestion_checkpoints

Tracks last successful ingestion.

Fields:

source_name

last_successful_ts

last_run_started_at

last_run_completed_at

status

rows_ingested

notes

ticker_flow_aggregates

Stores daily aggregate features.

Fields:

trade_date

ticker

direction

window

total_premium

event_count

repeat_flow_count

avg_dte_weighted

avg_volume_oi_ratio

flow_bias

score_raw

candidate_rankings

Stores ranked daily watchlist.

Fields:

trade_date

ticker

direction

rank

aggregate_score

included_reason

price_features

Stores daily derived price features.

Fields:

trade_date

ticker

close

ema20

ema50

atr14

trend_direction

structure_state

support_level

resistance_level

pullback_valid

support_hold_valid

confirmation_valid

trade_signals

Stores final outputs.

Fields:

signal_id

trade_date

ticker

direction

candidate_rank

conviction_score

entry_zone_low

entry_zone_high

stop_price

target_1

target_2

time_stop_days

reasons_json

flow_snapshot_json

price_snapshot_json

7. Processing Pipeline
7.1 Daily job schedule

Run once after market close.

Job sequence

ingest new flow

normalize and dedupe

persist raw flow

build rolling ticker aggregates

rank candidates

pull daily OHLCV for ranked candidates

compute price features

evaluate rule engine

write final signals

write run logs/checkpoints

8. Rule Engine Specification
8.1 Flow qualification rules

A flow event qualifies for swing continuation if:

premium >= 500,000

DTE >= 30

DTE <= 120

execution_side == ASK if available

ticker is valid equity ticker

option_type maps cleanly to directional bias

Direction mapping

CALL @ ASK → bullish

PUT @ ASK → bearish

For V1, skip ambiguous cases where side is not usable.

8.2 Aggregate-level candidate rules

A ticker becomes a candidate if its 1D/3D aggregate meets minimum standards.

Example V1 bullish criteria

bullish total premium over 1D or 3D above threshold

repeat_flow_count >= 1

flow_bias > 0

Example V1 bearish criteria

bearish total premium over 1D or 3D above threshold

repeat_flow_count >= 1

flow_bias < 0

8.3 Trend rules
Long

close > ema20

optionally ema20 > ema50

recent structure suggests higher highs / higher lows

Short

close < ema20

optionally ema20 < ema50

recent structure suggests lower highs / lower lows

8.4 Pullback rules
Long

price has pulled back from recent highs

current location is near one of:

daily ema20

prior breakout level

support zone

Short

price has bounced into:

daily ema20

prior breakdown level

resistance zone

8.5 Support/resistance hold rules
Long

recent bars touched support zone

latest close remains above lower support band

no decisive breakdown through support

Short

recent bars touched resistance zone

latest close remains below upper resistance band

no decisive breakout above resistance

8.6 Confirmation candle rules

Bullish confirmation can be any of:

bullish engulfing

strong close near highs

bullish rejection wick

Bearish confirmation is the opposite version.

V1 recommendation:
Require support/resistance hold plus at least one confirmation pattern.

9. Indicator Definitions
9.1 EMA

EMA20 on daily closes

EMA50 on daily closes

9.2 ATR

ATR14 on daily candles

9.3 Support/resistance zone width

default width = 0.25 * ATR14

10. Conviction Scoring
Score range

0 to 10

Suggested weights
Flow score (0–4)

premium >= 500k: +1

premium >= 1M: +1

repeat flow present: +1

avg volume / OI > 1: +1

Trend/structure score (0–3)

trend aligned: +2

clean pullback: +1

Location score (0–2)

near support/resistance level: +1

not overextended: +1

Confirmation score (0–1)

valid confirmation candle: +1

Signal thresholds

< 6: reject

6–7: medium conviction

8–10: high conviction

11. Exit Logic Spec

V1 signals should include suggested management logic even if execution is manual.

Long exits

stop below pullback low / support zone

target 1 at 1.5R

move stop to breakeven after partial

remaining position trails or exits on trend break

time stop after 3–5 days if no progress

Short exits

Mirror long logic.

12. Ranking Engine
Purpose

Reduce all tickers into a small daily watchlist.

Ranking inputs

total premium

repeat flow count

flow bias

weighted average DTE

execution quality

Example raw rank formula
rank_score = (
    0.45 * normalized_total_premium +
    0.25 * normalized_repeat_flow_count +
    0.20 * normalized_flow_bias +
    0.10 * normalized_avg_dte_quality
)

Output:

top N bullish candidates

top N bearish candidates

Suggested V1:

top 10 bullish

top 10 bearish

then price rules decide final signals

13. Module Breakdown
config.py

Holds all tunable parameters:

thresholds

file paths

scheduling config

ranking weights

vendors/unusual_whales.py

Vendor adapter:

API fetch methods

response parsing

vendor-to-canonical mapping

data/raw_store.py

Raw storage utilities:

write parquet

read by date range

dedupe helpers

features/flow_features.py

Builds ticker-level aggregate features.

features/price_features.py

Downloads OHLCV and computes:

EMA

ATR

structure labels

support/resistance zones

confirmation flags

ranking/candidate_ranker.py

Scores and ranks tickers from flow aggregates.

rules/continuation_rules.py

Evaluates whether a candidate qualifies as a final trade setup.

signals/generator.py

Builds final TradeSignal objects.

jobs/daily_run.py

Orchestrates the full daily pipeline.

db/models.py

Database schemas / ORM models.

utils/logging.py

Run logs, exceptions, diagnostics.

14. Orchestration Logic
Daily run pseudocode
def run_daily_pipeline(run_date):
    checkpoint = load_checkpoint("uw_flow")

    raw_events = ingest_new_flow(since=checkpoint.last_successful_ts)
    normalized = normalize_flow_events(raw_events)
    deduped = dedupe_flow_events(normalized)

    write_raw_flow(deduped)

    flow_aggregates = build_flow_aggregates(run_date)
    save_flow_aggregates(flow_aggregates)

    candidates = rank_candidates(flow_aggregates)
    save_candidate_rankings(candidates)

    tickers = [c.ticker for c in candidates]
    ohlcv = fetch_daily_ohlcv(tickers)
    price_features = build_price_features(ohlcv)
    save_price_features(price_features)

    signals = generate_signals(candidates, price_features)
    save_trade_signals(signals)

    update_checkpoint("uw_flow", max_event_ts(deduped))
15. Error Handling Requirements
Must handle

API timeout/failure

duplicate event windows

partial data for some tickers

missing OHLCV data

malformed vendor rows

empty candidate day

Required behavior

log failure

preserve partial outputs where valid

do not corrupt checkpoints on failed runs

allow rerun for a date range

16. Logging and Observability

Each run should log:

start/end time

rows fetched

rows deduped

rows written

number of candidate tickers

number of final signals

failures/warnings

Useful metrics:

average daily raw flow count

average filtered flow count

average candidate count

average signal count

17. Configuration Defaults
MIN_PREMIUM = 500_000
STRONG_PREMIUM = 1_000_000
MIN_DTE = 30
MAX_DTE = 120

TOP_BULLISH_CANDIDATES = 10
TOP_BEARISH_CANDIDATES = 10

EMA_FAST = 20
EMA_SLOW = 50
ATR_PERIOD = 14
ZONE_ATR_MULTIPLIER = 0.25

MIN_CONVICTION = 6
TIME_STOP_DAYS = 5
TARGET_1_R = 1.5
TARGET_2_R = 3.0
18. Backtest Compatibility Requirements

The system should preserve enough daily history so you can later test:

signal frequency

win rate

average move after signal

expectancy by conviction bucket

performance by market regime

Therefore every signal row should snapshot:

flow features used

price features used

thresholds/version used

19. V1 Deliverables
Required deliverables

working daily ingestion job

normalized raw flow storage

aggregate feature pipeline

candidate ranking output

price-rule engine

final signal output

config-driven thresholds

run logs/checkpointing

20. Suggested Build Order

define schemas and config

build UW normalization layer

build raw storage + checkpointing

build daily flow aggregate builder

build candidate ranker

build Yahoo Finance OHLCV fetcher

build price-feature functions

build rule engine

build signal generator

add logs/tests/replay scripts

---

# Future Version Notes

## V3 — Agent-validated scoring

Replace brittle rules with specialized LLM/vision agents for the hardest scoring
aspects. The rules engine continues to do the heavy lifting (filtering 500+ tickers
down to ~30-50 candidates cheaply), but each surviving candidate gets validated by
agents before a final signal decision.

**Architecture:** Rules pipeline → top ~50 candidates → parallel agent calls → agent
score aggregation → final signal.

Candidate agents:
- **S/R quality agent** — receives chart image + algorithmically identified levels;
  answers "is this genuine structural support/resistance that a human trader would
  respect?" Solves the problem of rules identifying noise levels (the BA issue).
- **Pattern quality agent** — evaluates multi-day price sequences holistically;
  catches complex patterns that are hard to enumerate as if/else rules.
- **Context agent** — checks earnings calendar, sector rotation, news catalysts;
  provides awareness that pure price/flow data cannot capture.

Agent output: a confidence modifier (e.g., 0.5x–1.5x) applied to the rules-based
price score, or an independent 0-10 agent score blended into the final formula.

Cost is manageable: ~30-50 LLM calls per pipeline run, a few cents total.

## V4 — Specialized parallel agents

Split the single validation agent into domain-specific agents that run in parallel:
S/R quality, pattern quality, macro context, sector momentum. Each contributes an
independent score channel to the final blend, similar to how flow/price/options
currently work.

## V5 — Agent-in-the-loop position management

Extend agents to open-position review: evaluate whether the thesis is still intact,
whether to tighten stops, or whether a better opportunity warrants rotation. Agents
receive the current chart, entry context, and position health metrics.

## Flow scoring — historical distribution (all future versions)

Replace absolute-threshold flow scoring with Z-scores against a rolling historical
distribution, or percentile ranks against a lookback window. Requires accumulated
history (weeks/months of pipeline runs). This gives a statistically grounded answer
to "is this flow genuinely unusual?" rather than relying on hand-tuned thresholds.

## V3+ pattern additions

- **Reclaim / V-recovery / capitulation bottom** — price breaks below key support,
  then reclaims it with volume. Currently excluded from the rules engine because
  it's a reversal (not continuation) setup and harder to code reliably.
