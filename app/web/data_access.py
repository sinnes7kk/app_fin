"""Resolve latest CSV/JSON artifacts under data/."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def latest_glob(subdir: str, pattern: str) -> Path | None:
    folder = DATA_ROOT / subdir
    if not folder.is_dir():
        return None
    matches = sorted(folder.glob(pattern), key=lambda p: p.name, reverse=True)
    return matches[0] if matches else None


def read_csv_safe(path: Path | None) -> pd.DataFrame:
    if path is None or not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json_list(path: Path) -> list:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def load_final_signals() -> tuple[pd.DataFrame, str]:
    p = latest_glob("final_signals", "final_signals_*.csv")
    df = read_csv_safe(p)
    return df, p.name if p else ""


def load_rejected() -> tuple[pd.DataFrame, str]:
    p = latest_glob("final_signals", "rejected_*.csv")
    df = read_csv_safe(p)
    return df, p.name if p else ""


def load_ranked_bullish() -> tuple[pd.DataFrame, str]:
    p = latest_glob("ranked_candidates", "ranked_bullish_*.csv")
    df = read_csv_safe(p)
    return df, p.name if p else ""


def load_ranked_bearish() -> tuple[pd.DataFrame, str]:
    p = latest_glob("ranked_candidates", "ranked_bearish_*.csv")
    df = read_csv_safe(p)
    return df, p.name if p else ""


def load_flow_features() -> tuple[pd.DataFrame, str]:
    p = latest_glob("flow_features", "flow_features_*.csv")
    df = read_csv_safe(p)
    return df, p.name if p else ""


def load_positions() -> list[dict]:
    return read_json_list(DATA_ROOT / "positions.json")


def load_watchlist() -> list[dict]:
    return read_json_list(DATA_ROOT / "watchlist.json")


def load_trade_log() -> pd.DataFrame:
    p = DATA_ROOT / "trade_log.csv"
    if not p.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def load_trade_log_tail(n: int = 100) -> pd.DataFrame:
    df = load_trade_log()
    return df.tail(n) if len(df) > n else df


def load_equity_curve() -> pd.DataFrame:
    p = DATA_ROOT / "equity_curve.csv"
    if not p.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()
