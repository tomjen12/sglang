"""Opt-in capture helpers for Muse Spark width-8 replay validation.

Normal serving pays only an environment-variable check and a null context.
Capture artifacts are written per rank with atomic renames so a partial process
failure cannot look like a complete replay bundle.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import re
import tempfile
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch

CAPTURE_DIR_ENV = "SGLANG_MUSE_REPLAY_CAPTURE_DIR"
MARKERS_ENV = "SGLANG_MUSE_REPLAY_MARKERS"
SCHEMA_VERSION = 1
CAPTURE_STEP_ENV = "SGLANG_MUSE_REPLAY_CAPTURE_STEP"
CAPTURE_MIN_PREFIX_ENV = "SGLANG_MUSE_REPLAY_MIN_PREFIX"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_NOOP_SCOPE = contextlib.nullcontext()
_MARKERS_ENABLED = bool(os.getenv(CAPTURE_DIR_ENV)) or os.getenv(
    MARKERS_ENV, ""
).lower() in {"1", "true", "yes"}
_ACTIVE_METADATA: ContextVar[RuntimeMetadataCollector | None] = ContextVar(
    "muse_replay_metadata", default=None
)
_ACTIVE_ROUTING_OVERRIDE: ContextVar[Mapping[int, torch.Tensor] | None] = ContextVar(
    "muse_replay_routing_override", default=None
)


def markers_enabled() -> bool:
    return _MARKERS_ENABLED


def replay_scope(name: str):
    """Return a profiler range only when replay validation is enabled."""
    if not markers_enabled():
        return _NOOP_SCOPE
    return torch.profiler.record_function(f"muse_replay/{name}")


def replay_scoped(name: str):
    """Decorate a boundary without adding call overhead in normal serving."""

    def decorate(function):
        if not markers_enabled():
            return function

        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            with replay_scope(name):
                return function(*args, **kwargs)

        return wrapped

    return decorate


@contextlib.contextmanager
def force_eager_model_runner(model_runner):
    """Bypass both CUDA-graph families while collecting Python layer metadata."""
    attributes = ("decode_cuda_graph_runner", "prefill_cuda_graph_runner")
    saved = {name: getattr(model_runner, name, None) for name in attributes}
    try:
        for name in attributes:
            if hasattr(model_runner, name):
                setattr(model_runner, name, None)
        yield
    finally:
        for name, value in saved.items():
            if hasattr(model_runner, name):
                setattr(model_runner, name, value)


def tensor_descriptor(value: torch.Tensor | None) -> dict[str, Any] | None:
    if not isinstance(value, torch.Tensor):
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
        "contiguous": value.is_contiguous(),
    }


class RuntimeMetadataCollector:
    """Collect layer shapes and tiny workload statistics, never activations."""

    def __init__(self, writer: ReplayBundleWriter, name: str):
        self.writer = writer
        self.name = name
        self.metadata: dict[str, Any] = {}

    def tensor(self, name: str, value: torch.Tensor | None) -> None:
        descriptor = tensor_descriptor(value)
        if descriptor is not None:
            self.metadata[name] = descriptor

    def route(
        self,
        *,
        layer: int,
        topk_ids: torch.Tensor,
        num_experts: int,
        local_start: int,
        local_experts: int,
    ) -> None:
        counts = torch.bincount(
            topk_ids.reshape(-1).to(torch.int64), minlength=num_experts
        ).to(device="cpu")
        self.metadata[f"layer_{layer}.routing"] = {
            "rows": int(topk_ids.shape[0]),
            "top_k": int(topk_ids.shape[-1]),
            "num_experts": int(num_experts),
            "local_expert_start": int(local_start),
            "local_experts": int(local_experts),
            "global_token_counts": counts.tolist(),
            "local_token_counts": counts[
                local_start : local_start + local_experts
            ].tolist(),
        }

    def flush(self) -> None:
        self.writer.write_metadata(self.name, self.metadata)


@contextlib.contextmanager
def runtime_metadata_capture(
    writer: ReplayBundleWriter | None, name: str
) -> Iterator[RuntimeMetadataCollector | None]:
    if writer is None:
        yield None
        return
    collector = RuntimeMetadataCollector(writer, name)
    token = _ACTIVE_METADATA.set(collector)
    try:
        yield collector
    finally:
        _ACTIVE_METADATA.reset(token)
    collector.flush()


def record_tensor_descriptor(name: str, value: torch.Tensor | None) -> None:
    collector = _ACTIVE_METADATA.get()
    if collector is not None:
        collector.tensor(name, value)


def record_route_workload(
    *,
    layer: int,
    topk_ids: torch.Tensor,
    num_experts: int,
    local_start: int,
    local_experts: int,
) -> None:
    collector = _ACTIVE_METADATA.get()
    if collector is not None:
        collector.route(
            layer=layer,
            topk_ids=topk_ids,
            num_experts=num_experts,
            local_start=local_start,
            local_experts=local_experts,
        )


@contextlib.contextmanager
def routing_override(
    layer_topk_ids: Mapping[int, torch.Tensor] | None,
) -> Iterator[None]:
    """Install replay-only routed expert ids after sequence-parallel gather."""
    token = _ACTIVE_ROUTING_OVERRIDE.set(layer_topk_ids)
    try:
        yield
    finally:
        _ACTIVE_ROUTING_OVERRIDE.reset(token)


def apply_routing_override(
    *, layer: int, topk_scores: torch.Tensor, topk_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    overrides = _ACTIVE_ROUTING_OVERRIDE.get()
    if overrides is None or layer not in overrides:
        return topk_scores, topk_ids
    replacement = overrides[layer]
    if replacement.shape != topk_ids.shape:
        raise ValueError(
            f"Replay routing override layer {layer} has shape "
            f"{tuple(replacement.shape)}, expected {tuple(topk_ids.shape)}."
        )
    if replacement.device != topk_ids.device or replacement.dtype != topk_ids.dtype:
        raise ValueError(
            f"Replay routing override layer {layer} has "
            f"{replacement.device}/{replacement.dtype}, expected "
            f"{topk_ids.device}/{topk_ids.dtype}."
        )
    return topk_scores, replacement


_FORWARD_BATCH_TENSORS = (
    "input_ids",
    "positions",
    "input_embeds",
    "req_pool_indices",
    "seq_lens",
    "seq_lens_cpu",
    "out_cache_loc",
    "out_cache_loc_virtual",
    "num_token_non_padded",
    "global_num_tokens_gpu",
    "global_num_token_non_padded",
    "mamba_track_indices",
    "mamba_track_mask",
)
_FORWARD_BATCH_SCALARS = (
    "batch_size",
    "seq_lens_sum",
    "global_num_token_non_padded_cpu",
    "can_run_decode_cuda_graph",
    "out_cache_loc_is_physical",
    "attention_kv_len_lower_bound",
)


def snapshot_forward_batch(
    batch: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Extract replay-stable fields without serializing the live Python object."""
    tensors = {
        name: value
        for name in _FORWARD_BATCH_TENSORS
        if isinstance((value := getattr(batch, name, None)), torch.Tensor)
    }
    metadata = {
        name: value
        for name in _FORWARD_BATCH_SCALARS
        if isinstance(
            (value := getattr(batch, name, None)),
            (bool, int, float, str, type(None)),
        )
    }
    for name in ("forward_mode", "capture_hidden_mode", "spec_algorithm"):
        value = getattr(batch, name, None)
        metadata[name] = getattr(value, "name", str(value))

    spec_info = getattr(batch, "spec_info", None)
    if spec_info is not None:
        struct_fields = getattr(spec_info, "__struct_fields__", ())
        if struct_fields:
            values = {
                name: getattr(spec_info, name)
                for name in struct_fields
            }
        else:
            values = getattr(spec_info, "__dict__", {})
        for name, value in values.items():
            key = f"spec_info.{name}"
            if isinstance(value, torch.Tensor):
                tensors[key] = value
            elif isinstance(value, (bool, int, float, str, type(None))):
                metadata[key] = value
    return tensors, metadata


def describe_forward_batch(batch: Any) -> dict[str, Any]:
    tensors, metadata = snapshot_forward_batch(batch)
    metadata["tensors"] = {
        name: tensor_descriptor(value) for name, value in tensors.items()
    }
    return metadata


def snapshot_named_tensors(
    value: Any, *, prefix: str, max_depth: int = 3
) -> dict[str, torch.Tensor]:
    """Flatten a small output struct without following private/runtime links."""
    result: dict[str, torch.Tensor] = {}
    seen: set[int] = set()

    def visit(item: Any, name: str, depth: int) -> None:
        if isinstance(item, torch.Tensor):
            result[name] = item
            return
        if item is None or depth >= max_depth or id(item) in seen:
            return
        if isinstance(item, Mapping):
            seen.add(id(item))
            for key, child in item.items():
                visit(child, f"{name}.{key}", depth + 1)
            return
        if isinstance(item, (list, tuple)):
            seen.add(id(item))
            for index, child in enumerate(item):
                visit(child, f"{name}.{index}", depth + 1)
            return
        values = getattr(item, "__dict__", None)
        if values is not None:
            seen.add(id(item))
            for key, child in values.items():
                if not key.startswith("_"):
                    visit(child, f"{name}.{key}", depth + 1)

    visit(value, prefix, 0)
    return result


def snapshot_iteration_ids(
    plan: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors = {}
    for name in (
        "req_pool_indices",
        "seq_lens",
        "seq_lens_cpu",
        "write_virtual",
        "write_physical",
    ):
        value = getattr(plan, name, None)
        if isinstance(value, torch.Tensor):
            tensors[name] = value
    for key, value in getattr(plan, "_write_ids", {}).items():
        if isinstance(value, torch.Tensor):
            tensors[f"write_ids.{_SAFE_NAME.sub('_', str(key))}"] = value
    return tensors, {"read_extent": int(getattr(plan, "read_extent", 0))}


def describe_iteration_ids(plan: Any) -> dict[str, Any]:
    tensors, metadata = snapshot_iteration_ids(plan)
    metadata["tensors"] = {
        name: tensor_descriptor(value) for name, value in tensors.items()
    }
    return metadata


def snapshot_read_table(
    plan: Any,
    reader: Any,
    *,
    kind: Any,
    name: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Capture the table exactly as attention resolves it for this iteration."""
    table = reader.read_table(plan, kind=kind)
    tensors = {
        f"{name}.ids": table.ids,
        f"{name}.row_ids": table.row_ids,
    }
    if table.v2p is not None:
        tensors[f"{name}.v2p"] = table.v2p
    metadata = {
        f"{name}.row_stride": int(table.row_stride),
        f"{name}.entry_page_size": int(table.entry_page_size),
        f"{name}.is_translated": bool(table.is_translated),
    }
    return tensors, metadata


def snapshot_kv_cache(
    plan: Any,
    reader: Any,
    pool: Any,
    *,
    layer_ids: range | tuple[int, ...] | list[int],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Snapshot active KV pages/rows for selected layers in their native layout."""
    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, Any] = {}
    wanted = set(layer_ids)
    router = pool.layer_router
    for sub_pool in router.sub_pools():
        selected_layers = sorted(wanted.intersection(sub_pool.layer_ids))
        if not selected_layers:
            continue
        table = reader.read_table(plan, kind=sub_pool.name)
        if table.v2p is not None:
            raise RuntimeError(
                f"Replay capture cannot snapshot unresolved v2p table {sub_pool.name!r}."
            )
        rows = table.ids.index_select(
            0, table.row_ids[: plan.req_pool_indices.numel()].to(torch.long)
        )
        max_length = int(plan.seq_lens.max().item()) + int(plan.read_extent)
        entry_count = (
            max_length + int(table.entry_page_size) - 1
        ) // int(table.entry_page_size)
        entries = torch.unique(rows[:, :entry_count].reshape(-1).to(torch.long))
        # Slot/page zero is the sink and remains part of the snapshot.
        view = router.view(sub_pool.name)
        page_size = int(getattr(view, "page_size", table.entry_page_size))
        layout = str(getattr(view, "kv_cache_layout", "nhd"))
        page_indexed = layout in {"vectorized_5d", "hnd", "page_major"}
        if page_indexed:
            storage_ids = (
                entries
                if int(table.entry_page_size) > 1
                else torch.div(entries, page_size, rounding_mode="floor")
            )
        elif int(table.entry_page_size) > 1:
            offsets = torch.arange(page_size, device=entries.device)
            storage_ids = (entries[:, None] * page_size + offsets).reshape(-1)
        else:
            storage_ids = entries
        storage_ids = torch.unique(storage_ids)
        prefix = f"kv.{_SAFE_NAME.sub('_', sub_pool.name)}"
        tensors[f"{prefix}.storage_ids"] = storage_ids
        tensors[f"{prefix}.table_ids"] = table.ids
        tensors[f"{prefix}.table_row_ids"] = table.row_ids
        metadata[f"{prefix}.entry_page_size"] = int(table.entry_page_size)
        metadata[f"{prefix}.layout"] = layout
        metadata[f"{prefix}.page_size"] = page_size
        for layer_id in selected_layers:
            layer = router.layer(layer_id)
            key, value = layer.kv()
            tensors[f"{prefix}.layer_{layer_id}.k"] = key.index_select(
                0, storage_ids
            )
            tensors[f"{prefix}.layer_{layer_id}.v"] = value.index_select(
                0, storage_ids
            )
    return tensors, metadata


def describe_kv_cache(
    plan: Any,
    reader: Any,
    pool: Any,
    *,
    layer_ids: range | tuple[int, ...] | list[int],
) -> dict[str, Any]:
    """Describe active cache geometry without storing page IDs or K/V values."""
    result: dict[str, Any] = {}
    wanted = set(layer_ids)
    router = pool.layer_router
    for sub_pool in router.sub_pools():
        selected_layers = sorted(wanted.intersection(sub_pool.layer_ids))
        if not selected_layers:
            continue
        table = reader.read_table(plan, kind=sub_pool.name)
        view = router.view(sub_pool.name)
        prefix = f"kv.{_SAFE_NAME.sub('_', sub_pool.name)}"
        max_length = int(plan.seq_lens.max().item()) + int(plan.read_extent)
        result[prefix] = {
            "table": tensor_descriptor(table.ids),
            "row_ids": tensor_descriptor(table.row_ids),
            "row_stride": int(table.row_stride),
            "entry_page_size": int(table.entry_page_size),
            "is_translated": bool(table.is_translated),
            "has_v2p": table.v2p is not None,
            "active_length": max_length,
            "page_size": int(getattr(view, "page_size", table.entry_page_size)),
            "layout": str(getattr(view, "kv_cache_layout", "nhd")),
            "layers": {},
        }
        for layer_id in selected_layers:
            key, value = router.layer(layer_id).kv()
            result[prefix]["layers"][str(layer_id)] = {
                "k": tensor_descriptor(key),
                "v": tensor_descriptor(value),
            }
    return result


class CaptureSelector:
    """Select one eligible width-8 decode cycle after short server warmups."""

    def __init__(self):
        self.root = os.getenv(CAPTURE_DIR_ENV)
        self.target_step = int(os.getenv(CAPTURE_STEP_ENV, "0"))
        self.min_prefix = int(os.getenv(CAPTURE_MIN_PREFIX_ENV, "8192"))
        self.decode_step = 0

    @property
    def enabled(self) -> bool:
        return bool(self.root)

    def next(
        self, *, rank: int, width: int, min_prefix_length: int | None = None
    ) -> ReplayBundleWriter | None:
        if not self.root:
            return None
        if min_prefix_length is None or min_prefix_length < self.min_prefix:
            return None
        step = self.decode_step
        self.decode_step += 1
        if step != self.target_step:
            return None
        if width != 8:
            raise ValueError(
                f"Muse replay capture requires width 8, got width={width}."
            )
        return ReplayBundleWriter(self.root, rank=rank, step=step)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class ReplayBundleWriter:
    """Write one rank's immutable replay inputs and an indexed manifest."""

    def __init__(self, root: str | os.PathLike[str], *, rank: int, step: int):
        self.rank = int(rank)
        self.step = int(step)
        self.directory = (
            Path(root) / f"step_{self.step:06d}" / f"rank_{self.rank:04d}"
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "manifest.json"
        self._manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "rank": self.rank,
            "step": self.step,
            "complete": False,
            "artifacts": [],
        }
        _atomic_json(self.manifest_path, self._manifest)

    def write_metadata(self, name: str, metadata: Mapping[str, Any]) -> Path:
        safe_name = _SAFE_NAME.sub("_", name).strip("_")
        if not safe_name:
            raise ValueError("Replay artifact name must contain a safe character.")
        path = self.directory / f"{safe_name}.json"
        _atomic_json(path, dict(metadata))
        self._record(name=name, kind="metadata", path=path)
        return path

    def write_tensors(
        self,
        name: str,
        tensors: Mapping[str, torch.Tensor],
        *,
        metadata: Mapping[str, str] | None = None,
        forbid_target_weights: bool = True,
    ) -> Path:
        """Synchronously snapshot tensors; call only in explicit capture runs."""
        if forbid_target_weights:
            forbidden = [
                key
                for key in tensors
                if "target" in key.lower() and "weight" in key.lower()
            ]
            if forbidden:
                raise ValueError(
                    "Target weights must be regenerated from the replay seed, "
                    f"not captured: {forbidden}"
                )
        safe_name = _SAFE_NAME.sub("_", name).strip("_")
        if not safe_name:
            raise ValueError("Replay artifact name must contain a safe character.")
        from safetensors.torch import save_file

        snapshots = {
            key: tensor.detach().to(device="cpu").contiguous()
            for key, tensor in tensors.items()
        }
        destination = self.directory / f"{safe_name}.safetensors"
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        try:
            save_file(snapshots, str(temporary), metadata=dict(metadata or {}))
            os.replace(temporary, destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        self._record(
            name=name,
            kind="tensors",
            path=destination,
            tensors={
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in snapshots.items()
            },
        )
        return destination

    def finish(self, metadata: Mapping[str, Any] | None = None) -> None:
        if metadata:
            self._manifest["metadata"] = dict(metadata)
        self._manifest["complete"] = True
        _atomic_json(self.manifest_path, self._manifest)

    def _record(self, *, name: str, kind: str, path: Path, **extra: Any) -> None:
        self._manifest["artifacts"].append(
            {
                "name": name,
                "kind": kind,
                "path": path.name,
                **extra,
            }
        )
        _atomic_json(self.manifest_path, self._manifest)


@contextlib.contextmanager
def maybe_capture_writer(
    *, rank: int, step: int
) -> Iterator[ReplayBundleWriter | None]:
    root = os.getenv(CAPTURE_DIR_ENV)
    if not root:
        yield None
        return
    writer = ReplayBundleWriter(root, rank=rank, step=step)
    try:
        yield writer
    except BaseException:
        raise
    else:
        writer.finish()
