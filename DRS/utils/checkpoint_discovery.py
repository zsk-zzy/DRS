"""Seed-aware checkpoint discovery for DRS evaluation."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any


def discover_test_checkpoints(root: Path, seeds: list[int], filename_pattern: str) -> dict[int, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"test_checkpoint_dir does not exist: {root}")
    candidates = {seed: [] for seed in seeds}
    seed_pattern = re.compile(r"(?:^|[_-])seed(\d+)(?=$|[_-])", re.IGNORECASE)
    for path in sorted(root.rglob(filename_pattern)):
        if not path.is_file():
            continue
        directory_names = (root.name, *path.relative_to(root).parts[:-1])
        for name in reversed(directory_names):
            match = seed_pattern.search(name)
            if match:
                seed = int(match.group(1))
                if seed in candidates:
                    candidates[seed].append(path.resolve())
                break
    resolved = {}
    for seed, matches in candidates.items():
        if not matches:
            raise FileNotFoundError(f"No {filename_pattern} found for seed {seed} under {root}.")
        if len(matches) > 1:
            listed = "\n".join(f"  {path}" for path in matches)
            raise ValueError(
                f"Multiple best checkpoints found for seed {seed} under {root}:\n{listed}\n"
                "Narrow test_checkpoint_dir or set explicit test_checkpoint_paths."
            )
        resolved[seed] = matches[0]
    return resolved


def resolve_test_checkpoints(
    config: dict[str, Any], seeds: list[int], filename_pattern: str = "best_model_*.pth",
) -> dict[int, Path]:
    """Resolve all requested seeds before inference; never select ambiguous runs."""
    paths = config.get("test_checkpoint_paths")
    single_path = config.get("test_checkpoint_path")
    search_dir = str(config.get("test_checkpoint_dir") or "").strip()
    if sum((paths is not None, bool(single_path), bool(search_dir))) > 1:
        raise ValueError("Set only one of test_checkpoint_dir, test_checkpoint_paths, test_checkpoint_path.")
    if search_dir:
        paths = discover_test_checkpoints(Path(search_dir).expanduser().resolve(), seeds, filename_pattern)
    if paths is not None and not isinstance(paths, dict):
        raise ValueError("test_checkpoint_paths must map each train seed to a checkpoint path.")
    if paths is None and len(seeds) != 1:
        raise ValueError("Multi-seed test_only requires test_checkpoint_dir or test_checkpoint_paths.")
    resolved = {}
    for seed in seeds:
        path = paths.get(seed, paths.get(str(seed))) if paths is not None else single_path
        if not path:
            raise ValueError(f"Missing test checkpoint for seed {seed}.")
        checkpoint_path = Path(str(path)).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Test checkpoint for seed {seed} does not exist: {checkpoint_path}")
        resolved[seed] = checkpoint_path
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("Each test seed must use a different checkpoint file.")
    return resolved
