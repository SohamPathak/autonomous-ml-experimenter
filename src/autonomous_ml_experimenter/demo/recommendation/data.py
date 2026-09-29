"""Retailrocket ingestion and leakage-safe temporal data preparation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from autonomous_ml_experimenter.config import DataConfig
from autonomous_ml_experimenter.core.contracts import DataManifest
from autonomous_ml_experimenter.demo.recommendation.fixtures import generate_interactions

REQUIRED_COLUMNS = {"timestamp", "visitorid", "event", "itemid"}
SUPPORTED_EVENTS = {"view", "addtocart", "transaction"}
EVENT_WEIGHTS = {"view": 1.0, "addtocart": 3.0, "transaction": 5.0}


@dataclass(frozen=True)
class InteractionSplits:
    """Prepared data plus training-only mappings and audit metadata."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    known_users: frozenset[int]
    known_items: frozenset[int]
    manifest: DataManifest
    cohort_report: dict[str, int]
    quality_report: dict[str, Any]


def validate_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the public interaction schema."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    normalized = frame.loc[:, sorted(REQUIRED_COLUMNS)].copy()
    normalized["timestamp"] = _parse_timestamp(normalized["timestamp"])
    normalized["visitorid"] = pd.to_numeric(normalized["visitorid"], errors="raise").astype("int64")
    normalized["itemid"] = pd.to_numeric(normalized["itemid"], errors="raise").astype("int64")
    normalized["event"] = normalized["event"].astype("string")
    invalid = set(normalized["event"].dropna().unique()).difference(SUPPORTED_EVENTS)
    if invalid:
        raise ValueError(f"unsupported event values: {sorted(invalid)}")
    if normalized[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("required interaction values cannot be null")
    if normalized.empty:
        raise ValueError("interaction data cannot be empty")
    return normalized.sort_values("timestamp", kind="stable", ignore_index=True)


def _parse_timestamp(series: pd.Series[Any]) -> pd.Series[Any]:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="ms", utc=True, errors="raise")
    return pd.to_datetime(series, utc=True, errors="raise")


def load_interactions(config: DataConfig, seed: int = 42) -> pd.DataFrame:
    """Load synthetic data or Retailrocket events.csv without storing raw copies."""
    if config.mode == "synthetic":
        return generate_interactions(config.sample_users or 120, seed=seed)
    if config.events_path is None:
        raise ValueError("events_path is required for public-data modes")
    path = Path(config.events_path)
    if not path.is_file():
        raise FileNotFoundError(f"events file not found: {path.name}")
    return pd.read_csv(path, usecols=sorted(REQUIRED_COLUMNS))


def prepare_interactions(
    frame: pd.DataFrame, config: DataConfig, seed: int = 42
) -> InteractionSplits:
    """Create deterministic global temporal splits without learning from future rows."""
    normalized = validate_events(frame)
    sampled = _sample_users_by_identity(normalized, config.sample_users, seed)
    train, validation, test = _global_temporal_split(
        sampled, config.validation_fraction, config.test_fraction
    )

    train_counts = train.groupby("visitorid", sort=False).size()
    eligible_users = frozenset(
        int(user_id) for user_id in train_counts[train_counts >= config.min_user_events].index
    )
    train = train[train["visitorid"].isin(eligible_users)].copy()
    if train.empty:
        raise ValueError("no training users remain after min_user_events filtering")

    for split in (train, validation, test):
        split["weight"] = split["event"].map(EVENT_WEIGHTS).astype("float64")

    known_users = frozenset(int(value) for value in train["visitorid"].unique())
    known_items = frozenset(int(value) for value in train["itemid"].unique())
    future = pd.concat([validation, test], ignore_index=True)
    cold_users = set(int(value) for value in future["visitorid"].unique()).difference(known_users)
    cold_items = set(int(value) for value in future["itemid"].unique()).difference(known_items)

    manifest = _manifest(sampled, config, seed)
    quality = {
        "missing_required_values": int(sampled[list(REQUIRED_COLUMNS)].isna().sum().sum()),
        "duplicate_events": int(sampled.duplicated(list(REQUIRED_COLUMNS)).sum()),
        "event_mix": {
            str(key): int(value) for key, value in sampled["event"].value_counts().items()
        },
        "train_rows_after_filter": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
    }
    return InteractionSplits(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        known_users=known_users,
        known_items=known_items,
        manifest=manifest,
        cohort_report={
            "warm_users": len(known_users),
            "cold_users": len(cold_users),
            "cold_items": len(cold_items),
        },
        quality_report=quality,
    )


def _sample_users_by_identity(frame: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    users = [int(value) for value in frame["visitorid"].unique()]
    if limit is None or len(users) <= limit:
        return frame.copy()

    def stable_order(user_id: int) -> str:
        return hashlib.sha256(f"{seed}:{user_id}".encode()).hexdigest()

    selected = set(sorted(users, key=stable_order)[:limit])
    return frame[frame["visitorid"].isin(selected)].reset_index(drop=True)


def _global_temporal_split(
    frame: pd.DataFrame, validation_fraction: float, test_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_times = frame["timestamp"].drop_duplicates().sort_values().reset_index(drop=True)
    if len(unique_times) < 5:
        raise ValueError("at least five distinct timestamps are required for temporal splitting")
    train_index = max(1, int(len(unique_times) * (1 - validation_fraction - test_fraction)))
    test_index = max(train_index + 1, int(len(unique_times) * (1 - test_fraction)))
    test_index = min(test_index, len(unique_times) - 1)
    validation_start = unique_times.iloc[train_index]
    test_start = unique_times.iloc[test_index]
    train = frame[frame["timestamp"] < validation_start].copy()
    validation = frame[
        (frame["timestamp"] >= validation_start) & (frame["timestamp"] < test_start)
    ].copy()
    test = frame[frame["timestamp"] >= test_start].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("temporal split produced an empty partition")
    return train, validation, test


def _manifest(frame: pd.DataFrame, config: DataConfig, seed: int) -> DataManifest:
    stable = frame.loc[:, ["timestamp", "visitorid", "event", "itemid"]].copy()
    stable["timestamp"] = stable["timestamp"].astype("int64")
    row_hashes = pd.util.hash_pandas_object(stable, index=False).to_numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(row_hashes)
    digest.update(f"{config.mode}:{config.sample_users}:{config.min_user_events}:{seed}".encode())
    start = frame["timestamp"].min().to_pydatetime()
    end = frame["timestamp"].max().to_pydatetime()
    return DataManifest(
        fingerprint=digest.hexdigest(),
        row_count=len(frame),
        user_count=int(frame["visitorid"].nunique()),
        item_count=int(frame["itemid"].nunique()),
        start_time=_as_datetime(start),
        end_time=_as_datetime(end),
        source="generated fixture" if config.mode == "synthetic" else "Retailrocket events.csv",
        sampling={"mode": config.mode, "user_limit": config.sample_users, "seed": seed},
    )


def _as_datetime(value: datetime) -> datetime:
    return value
