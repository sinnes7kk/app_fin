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
