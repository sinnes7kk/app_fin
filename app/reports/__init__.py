"""Reports package — read-only side outputs for audit and review.

Modules under this package consume artefacts already written by the
core pipeline (``data/final_signals/``, ``data/flow_features/``, etc.)
and emit summary CSVs / markdown for human consumption. They never
mutate live signal generation.
"""
