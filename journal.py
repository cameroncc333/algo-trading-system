"""
Paper Trading Journal — structured logging for documenting:
  What the model said -> What you did -> What happened -> What you learned
"""

import json
import os
from datetime import datetime
from config import JOURNAL_FILE


def load_journal():
    if os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE, "r") as f:
            return json.load(f)
    return []


def save_journal(entries):
    with open(JOURNAL_FILE, "w") as f:
        json.dump(entries, f, indent=2, default=str)


def add_entry(
    date, model_signal, sectors_recommended, action_taken, rationale,
    portfolio_value, benchmark_value, outcome="", lessons_learned="",
    confidence_level=5, market_context=""
):
    entries = load_journal()
    entry = {
        "id": len(entries) + 1,
        "date": date,
        "created_at": datetime.now().isoformat(),
        "model_signal": model_signal,
        "sectors_recommended": sectors_recommended,
        "action_taken": action_taken,
        "rationale": rationale,
        "portfolio_value": portfolio_value,
        "benchmark_value": benchmark_value,
        "alpha_vs_spy": portfolio_value - benchmark_value,
        "outcome": outcome,
        "lessons_learned": lessons_learned,
        "confidence_level": confidence_level,
        "market_context": market_context,
        "updated_at": None,
    }
    entries.append(entry)
    save_journal(entries)
    return entry


def update_entry(entry_id, outcome=None, lessons_learned=None):
    entries = load_journal()
    for entry in entries:
        if entry["id"] == entry_id:
            if outcome:
                entry["outcome"] = outcome
            if lessons_learned:
                entry["lessons_learned"] = lessons_learned
            entry["updated_at"] = datetime.now().isoformat()
            save_journal(entries)
            return entry
    raise ValueError(f"Entry {entry_id} not found")


def get_journal_summary():
    entries = load_journal()
    if not entries:
        return {"total_entries": 0, "avg_confidence": 0, "entries": []}
    return {
        "total_entries": len(entries),
        "avg_confidence": sum(e["confidence_level"] for e in entries) / len(entries),
        "entries": entries,
    }
