"""Secret-scanned presentation bundle serialization."""

from __future__ import annotations

from pathlib import Path

from autonomous_ml_experimenter.core.contracts import PresentationBundle

_FORBIDDEN_EXPORT_MARKERS = (
    "/users/",
    "/home/",
    "private_key",
    "access_token",
    "credentials.json",
    "vertax_creds",
)


def write_bundle(bundle: PresentationBundle, path: Path) -> None:
    """Write a compact cloud artifact after a conservative secret/path scan."""
    serialized = bundle.model_dump_json(indent=2)
    lowered = serialized.lower()
    found = [marker for marker in _FORBIDDEN_EXPORT_MARKERS if marker in lowered]
    if found:
        raise ValueError(f"unsafe presentation artifact markers: {', '.join(found)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def load_bundle(path: Path) -> PresentationBundle:
    """Validate a presentation bundle at the cloud boundary."""
    return PresentationBundle.model_validate_json(path.read_text(encoding="utf-8"))
