"""Recommendation model adapters used by the demonstration task."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares  # type: ignore[import-untyped]
from implicit.cpu.als import (  # type: ignore[import-untyped]
    AlternatingLeastSquares as CpuAlternatingLeastSquares,
)
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]


class PopularityModel:
    """Weighted popularity baseline with seen-item filtering."""

    name = "popularity"

    def __init__(self) -> None:
        self.item_scores: dict[int, float] = {}
        self.user_seen: dict[int, set[int]] = {}
        self.ranking: list[int] = []
        self.fallback_rate: float = 0.0

    @property
    def known_users(self) -> frozenset[int]:
        return frozenset(self.user_seen)

    @property
    def known_items(self) -> frozenset[int]:
        return frozenset(self.item_scores)

    def fit(self, data: pd.DataFrame) -> PopularityModel:
        required = {"visitorid", "itemid", "weight"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"missing model columns: {sorted(missing)}")
        grouped = data.groupby("itemid", sort=False)["weight"].sum()
        self.item_scores = {int(cast(int, item)): float(score) for item, score in grouped.items()}
        self.ranking = sorted(self.item_scores, key=lambda item: (-self.item_scores[item], item))
        self.user_seen = {
            int(cast(int, user)): set(int(item) for item in group["itemid"].unique())
            for user, group in data.groupby("visitorid", sort=False)
        }
        return self

    def predict(self, entity_ids: list[int], k: int) -> dict[int, list[int]]:
        if not self.ranking:
            raise RuntimeError("model must be fitted before prediction")
        predictions: dict[int, list[int]] = {}
        fallback_count = 0
        for entity_id in entity_ids:
            seen = self.user_seen.get(entity_id, set())
            if entity_id not in self.user_seen:
                fallback_count += 1
            predictions[entity_id] = [item for item in self.ranking if item not in seen][:k]
        self.fallback_rate = fallback_count / len(entity_ids) if entity_ids else 0.0
        return predictions

    def save(self, path: Path) -> None:
        target = path if path.suffix else path / "model.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "item_scores": {str(key): value for key, value in self.item_scores.items()},
            "user_seen": {str(key): sorted(value) for key, value in self.user_seen.items()},
        }
        target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> PopularityModel:
        target = path if path.suffix else path / "model.json"
        payload: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
        model = cls()
        model.item_scores = {
            int(key): float(value) for key, value in payload["item_scores"].items()
        }
        model.user_seen = {
            int(key): set(int(item) for item in value)
            for key, value in payload["user_seen"].items()
        }
        model.ranking = sorted(model.item_scores, key=lambda item: (-model.item_scores[item], item))
        return model


class CoVisitationModel:
    """Session-aware item-to-item model with recency-weighted retrieval."""

    name = "co_visitation"

    def __init__(
        self,
        session_minutes: int = 30,
        recency_decay: float = 0.9,
        neighbor_limit: int = 50,
    ) -> None:
        self.session_minutes = session_minutes
        self.recency_decay = recency_decay
        self.neighbor_limit = neighbor_limit
        self.similarities: dict[int, dict[int, float]] = {}
        self.user_history: dict[int, list[int]] = {}
        self.popularity = PopularityModel()

    @property
    def known_users(self) -> frozenset[int]:
        return frozenset(self.user_history)

    @property
    def known_items(self) -> frozenset[int]:
        return self.popularity.known_items

    def fit(self, data: pd.DataFrame) -> CoVisitationModel:
        required = {"timestamp", "visitorid", "itemid", "weight"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"missing model columns: {sorted(missing)}")
        ordered = data.sort_values(["visitorid", "timestamp"], kind="stable").copy()
        gap = ordered.groupby("visitorid")["timestamp"].diff()
        new_session = gap.isna() | (gap > pd.Timedelta(minutes=self.session_minutes))
        ordered["session_id"] = new_session.groupby(ordered["visitorid"]).cumsum()
        scores: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        for _, session in ordered.groupby(["visitorid", "session_id"], sort=False):
            items = [int(value) for value in session["itemid"]]
            weights = [float(value) for value in session["weight"]]
            for left_index, left_item in enumerate(items):
                for right_index in range(left_index + 1, len(items)):
                    right_item = items[right_index]
                    if left_item == right_item:
                        continue
                    distance = right_index - left_index
                    value = (
                        weights[left_index] * weights[right_index] * self.recency_decay**distance
                    )
                    scores[left_item][right_item] += value
                    scores[right_item][left_item] += value
        self.similarities = {
            item: dict(
                sorted(neighbors.items(), key=lambda pair: (-pair[1], pair[0]))[
                    : self.neighbor_limit
                ]
            )
            for item, neighbors in scores.items()
        }
        self.user_history = {
            int(cast(int, user)): [int(item) for item in group["itemid"]]
            for user, group in ordered.groupby("visitorid", sort=False)
        }
        self.popularity.fit(data)
        return self

    def predict(self, entity_ids: list[int], k: int) -> dict[int, list[int]]:
        predictions: dict[int, list[int]] = {}
        for entity_id in entity_ids:
            history = self.user_history.get(entity_id)
            if not history:
                predictions[entity_id] = self.popularity.predict([entity_id], k)[entity_id]
                continue
            seen = set(history)
            scores: dict[int, float] = defaultdict(float)
            for age, source_item in enumerate(reversed(history[-30:])):
                source_weight = self.recency_decay**age
                for candidate, similarity in self.similarities.get(source_item, {}).items():
                    if candidate not in seen:
                        scores[candidate] += source_weight * similarity
            ranked = sorted(scores, key=lambda item: (-scores[item], item))[:k]
            if len(ranked) < k:
                fallback = self.popularity.predict([entity_id], k)[entity_id]
                ranked.extend(item for item in fallback if item not in ranked)
            predictions[entity_id] = ranked[:k]
        return predictions

    def save(self, path: Path) -> None:
        target = path if path.suffix else path / "model.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "parameters": {
                "session_minutes": self.session_minutes,
                "recency_decay": self.recency_decay,
                "neighbor_limit": self.neighbor_limit,
            },
            "similarities": {
                str(item): {str(other): score for other, score in neighbors.items()}
                for item, neighbors in self.similarities.items()
            },
            "user_history": {str(user): history for user, history in self.user_history.items()},
            "popularity": {
                "item_scores": {
                    str(item): score for item, score in self.popularity.item_scores.items()
                },
                "user_seen": {
                    str(user): sorted(items) for user, items in self.popularity.user_seen.items()
                },
            },
        }
        target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CoVisitationModel:
        target = path if path.suffix else path / "model.json"
        payload: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
        model = cls(**payload["parameters"])
        model.similarities = {
            int(item): {int(other): float(score) for other, score in neighbors.items()}
            for item, neighbors in payload["similarities"].items()
        }
        model.user_history = {
            int(user): [int(item) for item in history]
            for user, history in payload["user_history"].items()
        }
        model.popularity.item_scores = {
            int(item): float(score) for item, score in payload["popularity"]["item_scores"].items()
        }
        model.popularity.user_seen = {
            int(user): set(int(item) for item in items)
            for user, items in payload["popularity"]["user_seen"].items()
        }
        model.popularity.ranking = sorted(
            model.popularity.item_scores,
            key=lambda item: (-model.popularity.item_scores[item], item),
        )
        return model


class ALSModel:
    """Implicit-feedback alternating least squares with popularity fallback."""

    name = "implicit_als"

    def __init__(
        self,
        factors: int = 16,
        regularization: float = 0.05,
        iterations: int = 10,
        alpha: float = 20.0,
        seed: int = 42,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.seed = seed
        self.user_to_index: dict[int, int] = {}
        self.item_to_index: dict[int, int] = {}
        self.index_to_item: list[int] = []
        self.user_items: Any = None
        self.model: Any = None
        self.popularity = PopularityModel()

    @property
    def known_users(self) -> frozenset[int]:
        return frozenset(self.user_to_index)

    @property
    def known_items(self) -> frozenset[int]:
        return frozenset(self.item_to_index)

    def fit(self, data: pd.DataFrame) -> ALSModel:
        users = sorted(int(value) for value in data["visitorid"].unique())
        items = sorted(int(value) for value in data["itemid"].unique())
        self.user_to_index = {value: index for index, value in enumerate(users)}
        self.item_to_index = {value: index for index, value in enumerate(items)}
        self.index_to_item = items
        aggregated = data.groupby(["visitorid", "itemid"], as_index=False)["weight"].sum()
        rows = [self.user_to_index[int(value)] for value in aggregated["visitorid"]]
        columns = [self.item_to_index[int(value)] for value in aggregated["itemid"]]
        values = aggregated["weight"].to_numpy(dtype=np.float32)
        self.user_items = csr_matrix(
            (values, (rows, columns)), shape=(len(users), len(items)), dtype=np.float32
        )
        self.model = AlternatingLeastSquares(
            factors=min(self.factors, max(2, len(items) - 1)),
            regularization=self.regularization,
            alpha=self.alpha,
            iterations=self.iterations,
            random_state=self.seed,
            num_threads=1,
        )
        self.model.fit(self.user_items, show_progress=False)
        self.popularity.fit(data)
        return self

    def predict(self, entity_ids: list[int], k: int) -> dict[int, list[int]]:
        if self.model is None or self.user_items is None:
            raise RuntimeError("model must be fitted before prediction")
        predictions: dict[int, list[int]] = {}
        for entity_id in entity_ids:
            user_index = self.user_to_index.get(entity_id)
            if user_index is None:
                predictions[entity_id] = self.popularity.predict([entity_id], k)[entity_id]
                continue
            available = max(0, len(self.index_to_item) - self.user_items[user_index].nnz)
            request_count = min(k, available)
            if request_count == 0:
                predictions[entity_id] = []
                continue
            item_indices, _ = self.model.recommend(
                user_index,
                self.user_items[user_index],
                N=request_count,
                filter_already_liked_items=True,
            )
            predictions[entity_id] = [
                self.index_to_item[int(index)] for index in item_indices if int(index) >= 0
            ]
        return predictions

    def save(self, path: Path) -> None:
        if self.model is None or self.user_items is None:
            raise RuntimeError("model must be fitted before saving")
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / "model.npz")
        np.savez_compressed(
            path / "user_items.npz",
            data=self.user_items.data,
            indices=self.user_items.indices,
            indptr=self.user_items.indptr,
            shape=np.asarray(self.user_items.shape),
        )
        self.popularity.save(path / "popularity.json")
        metadata = {
            "parameters": {
                "factors": self.factors,
                "regularization": self.regularization,
                "iterations": self.iterations,
                "alpha": self.alpha,
                "seed": self.seed,
            },
            "user_to_index": self.user_to_index,
            "index_to_item": self.index_to_item,
        }
        (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ALSModel:
        metadata: dict[str, Any] = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        model = cls(**metadata["parameters"])
        model.user_to_index = {
            int(user): int(index) for user, index in metadata["user_to_index"].items()
        }
        model.index_to_item = [int(item) for item in metadata["index_to_item"]]
        model.item_to_index = {item: index for index, item in enumerate(model.index_to_item)}
        sparse = np.load(path / "user_items.npz")
        shape = tuple(int(value) for value in sparse["shape"])
        model.user_items = csr_matrix(
            (sparse["data"], sparse["indices"], sparse["indptr"]), shape=shape
        )
        model.model = CpuAlternatingLeastSquares.load(path / "model.npz")
        model.popularity = PopularityModel.load(path / "popularity.json")
        return model
