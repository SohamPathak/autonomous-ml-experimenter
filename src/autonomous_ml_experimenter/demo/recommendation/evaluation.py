"""Leakage-aware offline ranking evaluation."""

from __future__ import annotations

import math
import time
from collections import Counter
from typing import Any, cast

import numpy as np
import pandas as pd


def ranking_metrics(
    predictions: dict[int, list[int]],
    relevant: dict[int, set[int]],
    catalog_size: int,
    k: int,
) -> dict[str, float]:
    """Calculate macro ranking quality and catalog-level behavior."""
    recalls: list[float] = []
    ndcgs: list[float] = []
    reciprocal_ranks: list[float] = []
    recommended: list[int] = []
    for entity_id, truth in relevant.items():
        ranked = predictions.get(entity_id, [])[:k]
        recommended.extend(ranked)
        if not truth:
            continue
        hits = [1.0 if item in truth else 0.0 for item in ranked]
        recalls.append(sum(hits) / len(truth))
        dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
        ideal_hits = min(len(truth), k)
        idcg = sum(1 / math.log2(index + 2) for index in range(ideal_hits))
        ndcgs.append(dcg / idcg if idcg else 0.0)
        first_hit = next((index + 1 for index, hit in enumerate(hits) if hit), None)
        reciprocal_ranks.append(1 / first_hit if first_hit else 0.0)

    counts = Counter(recommended)
    total = sum(counts.values())
    concentration = sum((count / total) ** 2 for count in counts.values()) if total else 0.0
    coverage = len(counts) / catalog_size if catalog_size else 0.0
    return {
        f"recall_at_{k}": _mean(recalls),
        f"ndcg_at_{k}": _mean(ndcgs),
        f"mrr_at_{k}": _mean(reciprocal_ranks),
        "catalog_coverage": coverage,
        "recommendation_concentration": concentration,
    }


def evaluate_recommender(
    model: Any,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    k: int = 10,
) -> dict[str, float]:
    """Evaluate ranking quality, cold-start behavior, and prediction latency."""
    relevant = {
        int(cast(int, user)): set(int(item) for item in group["itemid"].unique())
        for user, group in evaluation.groupby("visitorid", sort=False)
    }
    users = list(relevant)
    predictions: dict[int, list[int]] = {}
    latencies: list[float] = []
    for user in users:
        start = time.perf_counter()
        predictions.update(model.predict([user], k))
        latencies.append((time.perf_counter() - start) * 1000)

    known_users = set(int(value) for value in getattr(model, "known_users", set()))
    known_items = set(int(value) for value in getattr(model, "known_items", set()))
    warm_users = [user for user in users if user in known_users]
    cold_users = [user for user in users if user not in known_users]
    base = ranking_metrics(predictions, relevant, len(known_items), k)
    warm = ranking_metrics(
        {user: predictions[user] for user in warm_users},
        {user: relevant[user] for user in warm_users},
        len(known_items),
        k,
    )
    cold = ranking_metrics(
        {user: predictions[user] for user in cold_users},
        {user: relevant[user] for user in cold_users},
        len(known_items),
        k,
    )
    unknown_items = ~evaluation["itemid"].isin(known_items)
    recommendation_items = [item for values in predictions.values() for item in values]
    popularity = train.groupby("itemid")["weight"].sum()
    total_popularity = float(popularity.sum())
    average_popularity = (
        float(
            np.mean([popularity.get(item, 0.0) / total_popularity for item in recommendation_items])
        )
        if recommendation_items and total_popularity
        else 0.0
    )
    base.update(
        {
            f"warm_recall_at_{k}": warm[f"recall_at_{k}"],
            f"cold_recall_at_{k}": cold[f"recall_at_{k}"],
            "unknown_user_rate": len(cold_users) / len(users) if users else 0.0,
            "unknown_item_rate": float(unknown_items.mean()) if len(evaluation) else 0.0,
            "fallback_rate": len(cold_users) / len(users) if users else 0.0,
            "average_recommended_popularity": average_popularity,
            "mean_latency_ms": _mean(latencies),
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
        }
    )
    return base


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
