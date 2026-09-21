"""Record runtime Triton winners and force them during workload replay."""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Callable

import torch


RECORD_DIR_ENV = "SGLANG_TRITON_AUTOTUNE_RECORD_DIR"
REPLAY_DIR_ENV = "SGLANG_TRITON_AUTOTUNE_REPLAY_DIR"
STRICT_REPLAY_ENV = "SGLANG_TRITON_AUTOTUNE_REPLAY_STRICT"

_LOCK = threading.Lock()
_WRITTEN: set[tuple[str, int, str, str]] = set()
_REPLAY_CACHE: dict[tuple[str, int], list[dict[str, Any]]] = {}


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.getenv("RANK", "0"))


def _key_id(key: dict[str, Any]) -> str:
    return json.dumps(key, sort_keys=True, separators=(",", ":"))


def _config_value(config: Any) -> dict[str, Any]:
    kwargs = dict(getattr(config, "kwargs", {}) or {})
    return {
        "kwargs": kwargs,
        "num_warps": int(config.num_warps),
        "num_ctas": int(getattr(config, "num_ctas", 1)),
        "num_stages": int(config.num_stages),
        "maxnreg": getattr(config, "maxnreg", None),
    }


def record_winner(kernel: str, key: dict[str, Any], config: Any) -> None:
    output_dir = os.getenv(RECORD_DIR_ENV, "")
    if not output_dir:
        return
    rank = _rank()
    identity = (output_dir, rank, kernel, _key_id(key))
    with _LOCK:
        if identity in _WRITTEN:
            return
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"rank_{rank}.json"
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            value = {"schema_version": 1, "rank": rank, "configs": []}
        value["configs"] = [
            item
            for item in value.get("configs", [])
            if not (item.get("kernel") == kernel and item.get("key") == key)
        ]
        value["configs"].append(
            {"kernel": kernel, "key": key, "config": _config_value(config)}
        )
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
        _WRITTEN.add(identity)


def _load_replay_entries(rank: int) -> list[dict[str, Any]] | None:
    directory = os.getenv(REPLAY_DIR_ENV, "")
    if not directory:
        return None
    cache_key = (directory, rank)
    if cache_key not in _REPLAY_CACHE:
        path = Path(directory) / f"rank_{rank}.json"
        if not path.is_file():
            raise RuntimeError(
                f"missing recorded Triton config for rank {rank}: {path}"
            )
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("rank") != rank or not isinstance(value.get("configs"), list):
            raise RuntimeError(f"invalid recorded Triton config: {path}")
        _REPLAY_CACHE[cache_key] = value["configs"]
    return _REPLAY_CACHE[cache_key]


def _matches_config(candidate: Any, expected: dict[str, Any]) -> bool:
    actual = _config_value(candidate)
    return actual == expected


def find_best_config(kernel: Any) -> Any | None:
    """Find best_config through Triton's nested decorator wrappers."""
    current = kernel
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        best_config = getattr(current, "best_config", None)
        if best_config is not None:
            return best_config
        current = getattr(current, "fn", None)
    return None


def replay_pruner(
    kernel: str,
    key_builder: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    allow_unrecorded_kernel: bool = False,
) -> Callable[..., list[Any]]:
    """Build a Triton early-config pruner for one replayable kernel."""

    def prune(
        configs: list[Any], named_args: dict[str, Any], **kwargs: Any
    ) -> list[Any]:
        rank = _rank()
        entries = _load_replay_entries(rank)
        if entries is None:
            return configs
        # Triton versions differ here: positional bindings are supplied in
        # named_args, while keyword-only launch arguments may be passed through
        # **kwargs. The KDA launch uses keyword arguments, so preserve both.
        arguments = dict(named_args)
        arguments.update(kwargs)
        key = key_builder(arguments)
        entry = next(
            (
                item
                for item in entries
                if item.get("kernel") == kernel and item.get("key") == key
            ),
            None,
        )
        if entry is None:
            strict_replay = os.getenv(STRICT_REPLAY_ENV, "").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if (
                allow_unrecorded_kernel
                and not strict_replay
                and not any(
                    item.get("kernel") == kernel for item in entries
                )
            ):
                # Bundles produced before this kernel gained recording support
                # remain replayable through their original autotune path.
                return configs
            raise RuntimeError(
                f"rank {rank} has no recorded {kernel} config for key {key}"
            )
        matches = [
            config
            for config in configs
            if _matches_config(config, entry.get("config", {}))
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"rank {rank} recorded {kernel} config is not a candidate: "
                f"{entry.get('config')}"
            )
        return matches

    return prune
