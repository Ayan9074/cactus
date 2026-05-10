from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class BundleManifest:
    schema_version: str
    created_at_utc: str
    source_mlir: str
    graph_file: str | None
    profile: str
    compatibility_tier: str
    weights_manifest: str | None
    extra: dict[str, Any]


def write_bundle_manifest(
    out_dir: Path,
    *,
    source_mlir: Path,
    graph_file: Path | None,
    profile: str,
    compatibility_tier: str,
    weights_manifest: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = BundleManifest(
        schema_version="cactus.bundle.v1",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_mlir=str(source_mlir),
        graph_file=str(graph_file) if graph_file is not None else None,
        profile=profile,
        compatibility_tier=compatibility_tier,
        weights_manifest=str(weights_manifest) if weights_manifest is not None else None,
        extra=extra or {},
    )
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")
    return path
