"""Deterministic synthetic interactions used in tests and the hosted demo."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

EVENT_TYPES = ("view", "addtocart", "transaction")


def generate_interactions(users: int = 120, days: int = 30, seed: int = 42) -> pd.DataFrame:
    """Generate temporal implicit-feedback data with learnable user clusters."""
    rng = np.random.default_rng(seed)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    item_count = max(30, users // 2)
    for user_id in range(users):
        cluster = user_id % 4
        preferred_candidates = np.arange(
            cluster * (item_count // 4), (cluster + 1) * (item_count // 4)
        )
        preferred = preferred_candidates[preferred_candidates < item_count]
        for day in range(days):
            if rng.random() > 0.35:
                continue
            session_size = int(rng.integers(2, 6))
            for position in range(session_size):
                explore = rng.random() < 0.15
                item_pool = np.arange(item_count) if explore or len(preferred) == 0 else preferred
                item_id = int(rng.choice(item_pool))
                event_roll = rng.random()
                if event_roll < 0.08:
                    event = "transaction"
                elif event_roll < 0.25:
                    event = "addtocart"
                else:
                    event = "view"
                rows.append(
                    {
                        "timestamp": start + timedelta(days=day, minutes=user_id + position * 4),
                        "visitorid": user_id,
                        "event": event,
                        "itemid": item_id,
                    }
                )
    frame = pd.DataFrame(rows).sort_values("timestamp", ignore_index=True)
    return frame.astype({"visitorid": "int64", "itemid": "int64", "event": "string"})
