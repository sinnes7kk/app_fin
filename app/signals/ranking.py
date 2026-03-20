"""Ranking helpers for scored ticker setups."""

from __future__ import annotations


def rank_signals(signals: list[dict]) -> list[dict]:
    """Sort signals by conviction score descending."""
    return sorted(
        signals,
        key=lambda x: (
            x["score"],
            x["ticker"],
        ),
        reverse=True,
    )