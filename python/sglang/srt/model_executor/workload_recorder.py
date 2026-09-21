"""Opt-in all-forward census and deep prefill workload recording.

The all-forward recorder writes CPU-only composition metadata. The independent
prefill recorder also accumulates MoE tokens-per-expert on the GPU and records
KDA/Mamba state-control metadata. A completed prefill uses one asynchronous
device-to-host copy, and background threads write without synchronizing the
model thread.
"""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.step_span_utils import _decode_query_width

logger = logging.getLogger(__name__)

_FORWARD_WORKLOAD_RECORDER: Optional["ForwardWorkloadRecorder"] = None
_PREFILL_WORKLOAD_RECORDER: Optional["PrefillWorkloadRecorder"] = None
_PREFILL_EXECUTION_CONTEXT: contextvars.ContextVar[
    Optional[Dict[str, Any]]
] = contextvars.ContextVar("prefill_workload_execution_context", default=None)


def get_or_create_forward_workload_recorder(
    output_dir: str,
    *,
    gpu_timing_enabled: bool = False,
) -> "ForwardWorkloadRecorder":
    global _FORWARD_WORKLOAD_RECORDER
    if _FORWARD_WORKLOAD_RECORDER is None:
        _FORWARD_WORKLOAD_RECORDER = ForwardWorkloadRecorder(
            output_dir, gpu_timing_enabled=gpu_timing_enabled
        )
    elif _FORWARD_WORKLOAD_RECORDER.output_dir != Path(output_dir):
        raise RuntimeError(
            "all-forward workload recorder is already configured for "
            f"{_FORWARD_WORKLOAD_RECORDER.output_dir}, not {output_dir}"
        )
    elif _FORWARD_WORKLOAD_RECORDER.gpu_timing_enabled != gpu_timing_enabled:
        raise RuntimeError(
            "all-forward workload recorder GPU timing mode does not match"
        )
    return _FORWARD_WORKLOAD_RECORDER


def set_global_prefill_workload_recorder(
    recorder: Optional["PrefillWorkloadRecorder"],
) -> None:
    global _PREFILL_WORKLOAD_RECORDER
    _PREFILL_WORKLOAD_RECORDER = recorder


def record_prefill_topk(layer_id: int, topk_ids: torch.Tensor) -> None:
    """Hot-path MoE hook. It is a no-op outside a recorded prefill."""
    recorder = _PREFILL_WORKLOAD_RECORDER
    if recorder is not None:
        recorder.record_topk(layer_id, topk_ids)


def commit_prefill_dspark_injection(
    forward_pass_id: int, *, status: str
) -> None:
    """Commit a DSpark record after its post-model KV injection is launched."""
    recorder = _PREFILL_WORKLOAD_RECORDER
    if recorder is not None:
        recorder.commit_dspark_injection(forward_pass_id, status=status)


@contextlib.contextmanager
def dspark_prefill_gpu_timing(forward_pass_id: int, device: Any):
    """Time one complete DSpark prefill and attach it to both recorders."""
    forward_recorder = _FORWARD_WORKLOAD_RECORDER
    timing = (
        forward_recorder.start_gpu_timing(device)
        if forward_recorder is not None
        and forward_recorder.gpu_timing_enabled
        else None
    )
    try:
        yield
    finally:
        if timing is not None:
            forward_recorder.finish_gpu_timing(timing)
            forward_recorder.attach_dspark_gpu_timing(
                forward_pass_id, timing
            )
            prefill_recorder = _PREFILL_WORKLOAD_RECORDER
            if prefill_recorder is not None:
                prefill_recorder.attach_dspark_gpu_timing(
                    forward_pass_id, timing
                )


@contextlib.contextmanager
def prefill_workload_execution_context(metadata: Dict[str, Any]):
    """Attach wrapper metadata to a nested target-model prefill forward."""
    token = _PREFILL_EXECUTION_CONTEXT.set(dict(metadata))
    try:
        yield
    finally:
        _PREFILL_EXECUTION_CONTEXT.reset(token)


def _phase_shape(query_lens: List[int], kv_lens: List[int]) -> Dict[str, Any]:
    if not query_lens:
        return {
            "requests": 0,
            "query_lens": [],
            "kv_lens": [],
            "sum_query_tokens": 0,
            "sum_kv_tokens": 0,
        }
    return {
        "requests": len(query_lens),
        "query_lens": query_lens,
        "kv_lens": kv_lens,
        "sum_query_tokens": sum(query_lens),
        "sum_kv_tokens": sum(kv_lens),
    }


def build_forward_workload_record(
    forward_batch,
    *,
    forward_pass_id: int,
    tp_rank: int,
    pp_rank: int,
    gpu_id: int,
    model_role: str,
    draft_model_idx: Optional[int],
) -> Dict[str, Any]:
    """Build a CPU-only description for every model forward."""
    mode = forward_batch.forward_mode
    request_ids = list(getattr(forward_batch, "rids", None) or [])
    context_query_lens: List[int] = []
    context_kv_lens: List[int] = []
    generation_query_lens: List[int] = []
    generation_kv_lens: List[int] = []
    context_request_ids: List[str] = []
    generation_request_ids: List[str] = []

    query_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
    prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    context_flags = getattr(forward_batch, "is_context_request_cpu", None)
    if query_lens is not None and prefix_lens is not None:
        for index, (prefix_len, query_len) in enumerate(
            zip(prefix_lens, query_lens)
        ):
            query_size = int(query_len)
            kv_size = int(prefix_len) + query_size
            is_context = (
                bool(context_flags[index])
                if context_flags is not None
                else not (mode.is_mixed() and query_size == 1)
            )
            if is_context:
                context_query_lens.append(query_size)
                context_kv_lens.append(kv_size)
                context_request_ids.append(
                    request_ids[index] if index < len(request_ids) else ""
                )
            else:
                generation_query_lens.append(query_size)
                generation_kv_lens.append(kv_size)
                generation_request_ids.append(
                    request_ids[index] if index < len(request_ids) else ""
                )
    elif mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is not None:
            generation_kv_lens = [int(value) for value in seq_lens_cpu.tolist()]
            generation_query_lens = [
                _decode_query_width(forward_batch)
            ] * len(generation_kv_lens)
            generation_request_ids = request_ids[: len(generation_kv_lens)]

    spec_info = getattr(forward_batch, "spec_info", None)
    spec_algorithm = getattr(forward_batch, "spec_algorithm", None)
    spec_algorithm_name = getattr(spec_algorithm, "name", None)
    if spec_algorithm_name is None and spec_algorithm is not None:
        spec_algorithm_name = str(spec_algorithm)
    global_forward_mode = getattr(forward_batch, "global_forward_mode", None)
    wrapper_context = _PREFILL_EXECUTION_CONTEXT.get()

    prefill = _phase_shape(context_query_lens, context_kv_lens)
    prefill["request_ids"] = context_request_ids
    decode = _phase_shape(generation_query_lens, generation_kv_lens)
    decode["request_ids"] = generation_request_ids

    return {
        "schema_version": 1,
        "record_type": "forward",
        "timestamp_ns": time.time_ns(),
        "completed_timestamp_ns": None,
        "host_elapsed_ns": None,
        "forward_id": forward_pass_id,
        "mode": mode.name,
        "logical_batch_size": int(forward_batch.batch_size),
        "input_token_count": (
            int(forward_batch.input_ids.numel())
            if getattr(forward_batch, "input_ids", None) is not None
            else 0
        ),
        "request_ids": request_ids,
        "prefill": prefill,
        "decode": decode,
        "is_prefill_only": bool(
            getattr(forward_batch, "is_prefill_only", False)
        ),
        "distributed": {
            "global_forward_mode": (
                global_forward_mode.name
                if global_forward_mode is not None
                else None
            ),
            "global_num_tokens": getattr(
                forward_batch, "global_num_tokens_cpu", None
            ),
            "original_global_num_tokens": getattr(
                forward_batch, "original_global_num_tokens_cpu", None
            ),
            "global_num_token_non_padded": getattr(
                forward_batch, "global_num_token_non_padded_cpu", None
            ),
            "attn_tp_sequence_sharded": bool(
                getattr(forward_batch, "attn_tp_sequence_sharded", False)
            ),
        },
        "speculative": {
            "enabled": bool(
                spec_algorithm is not None
                and getattr(spec_algorithm, "is_speculative", lambda: False)()
            ),
            "active_this_forward": spec_info is not None,
            "algorithm": spec_algorithm_name,
            "tokens_per_request": (
                int(spec_info.num_tokens_per_req)
                if spec_info is not None
                and getattr(spec_info, "num_tokens_per_req", -1) >= 0
                else None
            ),
        },
        "model_role": model_role,
        "draft_model_idx": draft_model_idx,
        "wrapper": (
            wrapper_context.get("wrapper")
            if wrapper_context is not None
            else None
        ),
        "execution": None,
        "graph_batch_size": None,
        "graph_num_tokens": None,
        "status": "started",
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "gpu_id": gpu_id,
    }


@dataclass
class _ForwardGpuTiming:
    anchor_event: Any
    start_event: Any
    device: Any
    end_event: Any = None


@dataclass
class _ForwardWrite:
    record: Dict[str, Any]
    model_timing: Optional[_ForwardGpuTiming] = None
    wrapper_timing: Optional[_ForwardGpuTiming] = None


def _add_gpu_timing_fields(
    record: Dict[str, Any],
    *,
    model_timing: Optional[_ForwardGpuTiming],
    wrapper_timing: Optional[_ForwardGpuTiming],
) -> None:
    def values(timing: _ForwardGpuTiming) -> tuple[float, float, float]:
        if timing.end_event is None:
            raise RuntimeError("GPU timing interval has no end event")
        timing.end_event.synchronize()
        start_ms = timing.anchor_event.elapsed_time(timing.start_event)
        end_ms = timing.anchor_event.elapsed_time(timing.end_event)
        return start_ms, end_ms, timing.start_event.elapsed_time(timing.end_event)

    if model_timing is not None:
        model_start, model_end, model_elapsed = values(model_timing)
        record["model_gpu_start_offset_ms"] = round(model_start, 6)
        record["model_gpu_end_offset_ms"] = round(model_end, 6)
        record["model_gpu_elapsed_ms"] = round(model_elapsed, 6)

    effective = wrapper_timing or model_timing
    if effective is not None:
        start_ms, end_ms, elapsed_ms = values(effective)
        record["gpu_start_offset_ms"] = round(start_ms, 6)
        record["gpu_end_offset_ms"] = round(end_ms, 6)
        record["gpu_elapsed_ms"] = round(elapsed_ms, 6)
        record["gpu_timing_scope"] = (
            "dspark_prefill_wrapper"
            if wrapper_timing is not None
            else "model_forward"
        )
    if wrapper_timing is not None:
        record["dspark_prefill_gpu_elapsed_ms"] = record["gpu_elapsed_ms"]


class ForwardWorkloadRecorder:
    """Bounded asynchronous writer for the all-forward workload timeline."""

    _STOP = object()

    def __init__(
        self,
        output_dir: str,
        flush_interval: int = 1,
        queue_size: int = 65536,
        gpu_timing_enabled: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "forwards.jsonl"
        self.metadata_path = self.output_dir / "run_metadata.json"
        self._file = self.path.open("a", encoding="utf-8", buffering=1024 * 1024)
        self._flush_interval = flush_interval
        self.gpu_timing_enabled = gpu_timing_enabled
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._gpu_anchor_event = None
        self._pending_dspark: Dict[int, _ForwardWrite] = {}
        self._closed = False
        self.dropped_records = 0
        self.dropped_after_close = 0
        self.written_records = 0
        self.write_errors = 0
        self._stats_lock = threading.Lock()
        self._write_error_logged = False
        self._thread = threading.Thread(
            target=self._run,
            name="sglang-forward-workload-recorder",
            daemon=True,
        )
        self._thread.start()
        atexit.register(self.close)

    def configure(self, *, server_args, parallel_state) -> None:
        metadata = {
            "schema_version": 1,
            "record_type": "forward_workload_metadata",
            "model": getattr(server_args, "model_path", None),
            "parallel": {
                "tp_size": int(parallel_state.tp_size),
                "dcp_size": int(parallel_state.attn_dcp_size),
                "ep_size": int(parallel_state.moe_ep_size),
                "pp_size": int(parallel_state.pp_size),
            },
            "page_size": int(server_args.page_size),
            "chunked_prefill_size": int(server_args.chunked_prefill_size),
            "cuda_graph_disabled": bool(
                getattr(server_args, "disable_cuda_graph", False)
            ),
            "cuda_graph_max_bs_decode": getattr(
                server_args, "cuda_graph_max_bs_decode", None
            ),
            "max_running_requests": getattr(
                server_args, "max_running_requests", None
            ),
            "context_length": getattr(server_args, "context_length", None),
            "attention_backend": getattr(
                server_args, "attention_backend", None
            ),
            "prefill_attention_backend": getattr(
                server_args, "prefill_attention_backend", None
            ),
            "decode_attention_backend": getattr(
                server_args, "decode_attention_backend", None
            ),
            "speculative_algorithm": getattr(
                server_args, "speculative_algorithm", None
            ),
            "speculative_draft_model_path": getattr(
                server_args, "speculative_draft_model_path", None
            ),
            "timing_semantics": (
                "host wall-clock plus optional asynchronous CUDA-event spans; "
                "GPU spans include bubbles inside each timed forward but not "
                "gaps between forwards"
            ),
            "gpu_timing_enabled": self.gpu_timing_enabled,
        }
        tmp_path = self.metadata_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(self.metadata_path)

    def start_gpu_timing(self, device: Any) -> _ForwardGpuTiming:
        if not self.gpu_timing_enabled:
            raise RuntimeError("forward GPU timing is disabled")
        device_module = torch.get_device_module(device)
        stream = device_module.current_stream(device)
        with self._stats_lock:
            if self._gpu_anchor_event is None:
                self._gpu_anchor_event = device_module.Event(enable_timing=True)
                self._gpu_anchor_event.record(stream)
            anchor = self._gpu_anchor_event
        start = device_module.Event(enable_timing=True)
        start.record(stream)
        return _ForwardGpuTiming(
            anchor_event=anchor, start_event=start, device=device
        )

    @staticmethod
    def finish_gpu_timing(timing: _ForwardGpuTiming) -> None:
        if timing.end_event is not None:
            raise RuntimeError("GPU timing interval was already finished")
        device_module = torch.get_device_module(timing.device)
        stream = device_module.current_stream(timing.device)
        end = device_module.Event(enable_timing=True)
        end.record(stream)
        timing.end_event = end

    def write(
        self,
        record: Dict[str, Any],
        *,
        model_timing: Optional[_ForwardGpuTiming] = None,
    ) -> None:
        item = _ForwardWrite(record=record, model_timing=model_timing)
        if (
            self.gpu_timing_enabled
            and record.get("wrapper") == "dspark_prefill"
            and record.get("mode") == "EXTEND"
        ):
            forward_id = int(record["forward_id"])
            if forward_id in self._pending_dspark:
                raise RuntimeError(
                    f"duplicate pending DSpark forward {forward_id}"
                )
            self._pending_dspark[forward_id] = item
            return
        self._enqueue_write(item)

    def attach_dspark_gpu_timing(
        self, forward_pass_id: int, timing: _ForwardGpuTiming
    ) -> None:
        item = self._pending_dspark.pop(forward_pass_id, None)
        if item is None:
            return
        item.wrapper_timing = timing
        self._enqueue_write(item)

    def _enqueue_write(self, item: _ForwardWrite) -> None:
        with self._stats_lock:
            if self._closed:
                self.dropped_after_close += 1
                return
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                self.dropped_records += 1

    def _run(self) -> None:
        pending = 0
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    break
                try:
                    _add_gpu_timing_fields(
                        item.record,
                        model_timing=item.model_timing,
                        wrapper_timing=item.wrapper_timing,
                    )
                    self._file.write(
                        json.dumps(item.record, separators=(",", ":")) + "\n"
                    )
                    self.written_records += 1
                    pending += 1
                    if pending >= self._flush_interval:
                        self._file.flush()
                        pending = 0
                except Exception:
                    self.write_errors += 1
                    if not self._write_error_logged:
                        logger.exception("Forward workload recorder write failed")
                        self._write_error_logged = True
            finally:
                self._queue.task_done()
        stats_record = {
            "schema_version": 1,
            "record_type": "recorder_stats",
            "timestamp_ns": time.time_ns(),
            "output_dir": str(self.output_dir),
            "written_records": self.written_records,
            "dropped_records": self.dropped_records,
            "dropped_after_close": self.dropped_after_close,
            "write_errors": self.write_errors,
            "closed": True,
        }
        self._file.write(json.dumps(stats_record, separators=(",", ":")) + "\n")
        self._file.flush()

    def close(self) -> None:
        global _FORWARD_WORKLOAD_RECORDER
        with self._stats_lock:
            if self._closed:
                return
            self._closed = True
        for item in self._pending_dspark.values():
            self._queue.put(item)
        self._pending_dspark.clear()
        self._queue.put(self._STOP)
        self._thread.join()
        self._file.close()
        if _FORWARD_WORKLOAD_RECORDER is self:
            _FORWARD_WORKLOAD_RECORDER = None


def get_moe_layer_ids(
    num_hidden_layers: int,
    first_k_dense_replace: int,
    moe_layer_freq: int,
    has_experts: bool,
) -> List[int]:
    if not has_experts:
        return []
    return [
        layer_id
        for layer_id in range(num_hidden_layers)
        if layer_id >= first_k_dense_replace
        and layer_id % moe_layer_freq == 0
    ]


def build_prefill_workload_record(
    forward_batch,
    *,
    forward_pass_id: int,
    tp_rank: int,
    pp_rank: int,
    gpu_id: int,
) -> Optional[Dict[str, Any]]:
    """Snapshot one prefill before graph replay can pad its metadata."""
    mode = forward_batch.forward_mode
    if not mode.is_extend_or_draft_extend_or_mixed(include_draft_extend_v2=True):
        return None

    query_lens = forward_batch.extend_seq_lens_cpu
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    if query_lens is None or prefix_lens is None:
        return None

    request_ids = list(getattr(forward_batch, "rids", None) or [])
    context_flags = getattr(forward_batch, "is_context_request_cpu", None)
    original_lens = getattr(forward_batch, "original_input_lens_cpu", None)
    remaining_lens = getattr(forward_batch, "remaining_prefill_tokens_cpu", None)
    final_chunks = getattr(forward_batch, "is_final_prefill_chunk_cpu", None)
    chunk_indices = getattr(forward_batch, "prefill_chunk_indices_cpu", None)
    cache_device = getattr(forward_batch, "cache_device_hit_tokens_cpu", None)
    cache_host = getattr(forward_batch, "cache_host_hit_tokens_cpu", None)
    cache_storage = getattr(forward_batch, "cache_storage_hit_tokens_cpu", None)
    sampling_params = getattr(forward_batch, "sampling_params_cpu", None)

    requests = []
    generation_requests = []
    for i, (prefix_len, query_len) in enumerate(zip(prefix_lens, query_lens)):
        query_size = int(query_len)
        prefix_kv_size = int(prefix_len)
        is_context = (
            bool(context_flags[i])
            if context_flags is not None
            else not (mode.is_mixed() and query_size == 1)
        )
        rid = request_ids[i] if i < len(request_ids) else ""
        sampling = (
            dict(sampling_params[i])
            if sampling_params is not None and i < len(sampling_params)
            else None
        )
        if not is_context:
            generation_request = {
                "request_id": rid,
                "query_size": query_size,
                "prefix_kv_size": prefix_kv_size,
            }
            if sampling is not None:
                generation_request["sampling_params"] = sampling
            generation_requests.append(generation_request)
            continue
        request = {
            "request_id": rid,
            "query_size": query_size,
            "prefix_kv_size": prefix_kv_size,
            "original_prompt_size": (
                int(original_lens[i])
                if original_lens is not None
                else prefix_kv_size + query_size
            ),
            "remaining_prefill_tokens": (
                int(remaining_lens[i]) if remaining_lens is not None else 0
            ),
            "chunk_index": (
                int(chunk_indices[i]) if chunk_indices is not None else 0
            ),
            "is_final_chunk": (
                bool(final_chunks[i]) if final_chunks is not None else True
            ),
            "cache_hit": {
                "device": int(cache_device[i]) if cache_device is not None else 0,
                "host": int(cache_host[i]) if cache_host is not None else 0,
                "storage": (
                    int(cache_storage[i]) if cache_storage is not None else 0
                ),
            },
        }
        if sampling is not None:
            request["sampling_params"] = sampling
        requests.append(request)

    if not requests:
        return None

    state_slots = getattr(
        forward_batch, "mamba_state_slot_present_cpu", None
    )
    track_mask = getattr(forward_batch, "mamba_track_mask_cpu", None)
    track_seqlens = getattr(
        forward_batch, "mamba_track_seqlens_cpu", None
    )
    spec_algorithm = getattr(forward_batch, "spec_algorithm", None)
    spec_algorithm_name = getattr(spec_algorithm, "name", None)
    if spec_algorithm_name is None and spec_algorithm is not None:
        spec_algorithm_name = str(spec_algorithm)
    capture_hidden_mode = getattr(forward_batch, "capture_hidden_mode", None)
    capture_hidden_mode_name = getattr(capture_hidden_mode, "name", None)
    wrapper_context = _PREFILL_EXECUTION_CONTEXT.get()
    execution_context = {
        "wrapper": (
            wrapper_context.get("wrapper")
            if wrapper_context is not None
            else None
        ),
        "speculative_algorithm": spec_algorithm_name,
        "capture_hidden_mode": capture_hidden_mode_name,
        "return_hidden_states_before_norm": bool(
            getattr(forward_batch, "return_hidden_states_before_norm", False)
        ),
        "record_scope": "target_model_forward",
    }
    return {
        "schema_version": 1,
        "record_type": "prefill_forward",
        "timestamp_ns": time.time_ns(),
        "forward_id": forward_pass_id,
        "mode": mode.name,
        "logical_batch_size": int(forward_batch.batch_size),
        "requests": requests,
        # A MIXED forward's MoE matrix includes these tokens too. Keeping their
        # shapes makes the recorded compute replayable without storing token IDs.
        "generation_requests": generation_requests,
        "execution_context": execution_context,
        "dspark_prefill": (
            dict(wrapper_context["dspark_prefill"])
            if wrapper_context is not None
            and isinstance(wrapper_context.get("dspark_prefill"), dict)
            else None
        ),
        "kda_state": {
            "enabled": state_slots is not None,
            "slot_present": state_slots,
            "tracking_enabled": track_mask is not None,
            "track_mask": track_mask,
            "track_seqlens": track_seqlens,
            "cow_count": int(
                getattr(forward_batch, "mamba_cow_count_cpu", 0)
            ),
            "clear_count": int(
                getattr(forward_batch, "mamba_clear_count_cpu", 0)
            ),
        },
        "moe_counts": None,
        "execution": None,
        "graph_batch_size": None,
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "gpu_id": gpu_id,
    }


@dataclass
class _MoeCopySlot:
    gpu_counts: torch.Tensor
    gpu_uint16: torch.Tensor
    cpu_uint16: torch.Tensor
    event: Any


@dataclass
class _PendingRecord:
    record: Dict[str, Any]
    slot: Optional[_MoeCopySlot]
    model_timing: Optional[_ForwardGpuTiming] = None
    wrapper_timing: Optional[_ForwardGpuTiming] = None
    injection_committed: bool = False


class PrefillWorkloadRecorder:
    """GPU histogram collector and bounded asynchronous sidecar writer."""

    _STOP = object()

    def __init__(
        self,
        output_dir: str,
        # Model workers are commonly stopped with SIGTERM, which bypasses
        # Python atexit handlers. Flush every completed binary+JSON pair so a
        # normal server stop cannot leave a large binary-only tail.
        flush_interval: int = 1,
        queue_size: int = 65536,
        copy_slots: int = 8,
        gpu_timing_enabled: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "prefill_forwards.jsonl"
        self.binary_path = self.output_dir / "moe_counts.uint16.bin"
        self.metadata_path = self.output_dir / "run_metadata.json"
        self._jsonl_file = self.jsonl_path.open(
            "a", encoding="utf-8", buffering=1024 * 1024
        )
        self._binary_file = self.binary_path.open("ab", buffering=1024 * 1024)
        self._flush_interval = flush_interval
        self.gpu_timing_enabled = gpu_timing_enabled
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._free_slots: queue.Queue = queue.Queue(maxsize=copy_slots)
        self._copy_slots = copy_slots
        self._copy_stream = None
        self._layer_to_row: Dict[int, int] = {}
        self._moe_shape: Optional[List[int]] = None
        self._active_slot: Optional[_MoeCopySlot] = None
        self._pending_dspark: Optional[_PendingRecord] = None
        self._active = False
        self._closed = False
        self.dropped_records = 0
        self.dropped_moe_counts = 0
        self.dropped_after_close = 0
        self.written_records = 0
        self.write_errors = 0
        self._stats_lock = threading.Lock()
        self._write_error_logged = False
        self._thread = threading.Thread(
            target=self._run,
            name="sglang-workload-recorder",
            daemon=True,
        )
        self._thread.start()
        atexit.register(self.close)

    def configure(
        self,
        *,
        model_config,
        server_args,
        parallel_state,
        gpu_id: int,
        kv_cache_dtype,
    ) -> None:
        hf_config = model_config.hf_text_config
        num_hidden_layers = int(hf_config.num_hidden_layers)
        first_dense = int(getattr(hf_config, "first_k_dense_replace", 0) or 0)
        moe_frequency = int(getattr(hf_config, "moe_layer_freq", 1) or 1)
        num_experts = getattr(hf_config, "num_experts", None)
        if parallel_state.pp_size != 1:
            raise NotImplementedError(
                "Prefill workload recording currently requires PP size 1"
            )
        if num_experts is not None and parallel_state.moe_ep_size != 1:
            raise NotImplementedError(
                "Prefill MoE workload recording currently requires EP size 1"
            )
        moe_layer_ids = get_moe_layer_ids(
            num_hidden_layers,
            first_dense,
            moe_frequency,
            has_experts=num_experts is not None,
        )

        if moe_layer_ids:
            self._layer_to_row = {
                layer_id: row for row, layer_id in enumerate(moe_layer_ids)
            }
            self._moe_shape = [len(moe_layer_ids), int(num_experts)]
            device = torch.device(f"cuda:{gpu_id}")
            device_module = torch.get_device_module(device)
            self._copy_stream = device_module.Stream(device=device)
            for _ in range(self._copy_slots):
                gpu_counts = torch.zeros(
                    self._moe_shape, dtype=torch.int32, device=device
                )
                self._free_slots.put(
                    _MoeCopySlot(
                        gpu_counts=gpu_counts,
                        gpu_uint16=torch.empty(
                            self._moe_shape, dtype=torch.uint16, device=device
                        ),
                        cpu_uint16=torch.empty(
                            self._moe_shape,
                            dtype=torch.uint16,
                            device="cpu",
                            pin_memory=True,
                        ),
                        event=device_module.Event(),
                    )
                )

        top_k = getattr(hf_config, "num_experts_per_tok", None)
        if top_k is None:
            top_k = getattr(hf_config, "num_experts_per_token", None)
        runner = getattr(server_args, "moe_runner_backend", None)
        if runner is not None and hasattr(runner, "value"):
            runner = runner.value
        spec_algorithm = getattr(server_args, "speculative_algorithm", None)
        if spec_algorithm is not None and hasattr(spec_algorithm, "value"):
            spec_algorithm = spec_algorithm.value
        metadata = {
            "schema_version": 1,
            "model": getattr(server_args, "model_path", None),
            "parallel": {
                "tp_size": int(parallel_state.tp_size),
                "dcp_size": int(parallel_state.attn_dcp_size),
                "ep_size": int(parallel_state.moe_ep_size),
            },
            "prefill": {
                "chunked_prefill_size": int(server_args.chunked_prefill_size),
                "page_size": int(server_args.page_size),
                "execution": (
                    "eager"
                    if getattr(server_args, "disable_cuda_graph", False)
                    else "recorded_per_forward"
                ),
            },
            "speculative": {
                "algorithm": spec_algorithm,
                "draft_model_path": getattr(
                    server_args, "speculative_draft_model_path", None
                ),
                "dspark_block_size": getattr(
                    server_args, "speculative_dspark_block_size", None
                ),
            },
            "moe": {
                "num_hidden_layers": num_hidden_layers,
                "num_moe_layers": len(moe_layer_ids),
                "moe_layer_ids": moe_layer_ids,
                "num_experts": int(num_experts) if num_experts is not None else 0,
                "num_experts_per_token": int(top_k) if top_k is not None else 0,
                "num_shared_experts": int(
                    getattr(hf_config, "num_shared_experts", 0) or 0
                ),
                "first_k_dense_replace": first_dense,
                "moe_layer_freq": moe_frequency,
                "runner": str(runner) if runner is not None else None,
                "counts_dtype": "uint16",
                "counts_layout": "layer_major_expert_minor",
                # These are the rows actually dispatched to MoE. MLP-sync
                # padding is intentionally retained because it consumes AITER
                # sorting/GEMM work and therefore affects replay latency.
                "counts_include_mlp_padding": True,
            },
            "cache": {"kv_cache_dtype": str(kv_cache_dtype).removeprefix("torch.")},
            "gpu_timing_enabled": self.gpu_timing_enabled,
        }
        tmp_path = self.metadata_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(self.metadata_path)

    def begin(self, record: Optional[Dict[str, Any]]) -> None:
        if self._active:
            raise RuntimeError("Nested prefill workload recording is not supported")
        if self._pending_dspark is not None:
            raise RuntimeError(
                "Previous DSpark prefill recording was not committed"
            )
        if record is None:
            return
        self._active = True
        if self._moe_shape is None:
            self._active_slot = None
            return
        try:
            self._active_slot = self._free_slots.get_nowait()
        except queue.Empty:
            self._active_slot = None
            with self._stats_lock:
                self.dropped_moe_counts += 1
        if self._active_slot is not None:
            self._active_slot.gpu_counts.zero_()

    def record_topk(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        if not self._active or self._active_slot is None:
            return
        row = self._layer_to_row.get(layer_id)
        if row is None:
            return
        ids = topk_ids.flatten()
        valid = (ids >= 0) & (ids < self._active_slot.gpu_counts.shape[1])
        self._active_slot.gpu_counts[row].scatter_add_(
            0,
            ids.masked_fill(~valid, 0).long(),
            valid.to(torch.int32),
        )

    def finish(
        self,
        record: Optional[Dict[str, Any]],
        *,
        model_timing: Optional[_ForwardGpuTiming] = None,
    ) -> None:
        if record is None:
            return
        if not self._active:
            raise RuntimeError("Prefill workload recording was not started")
        slot = self._active_slot
        self._active = False
        self._active_slot = None
        if slot is not None:
            device_module = torch.get_device_module(slot.gpu_counts.device)
            current_stream = device_module.current_stream(slot.gpu_counts.device)
            self._copy_stream.wait_stream(current_stream)
            with device_module.stream(self._copy_stream):
                slot.gpu_uint16.copy_(slot.gpu_counts)
                slot.cpu_uint16.copy_(slot.gpu_uint16, non_blocking=True)
                slot.event.record(self._copy_stream)
        pending = _PendingRecord(
            record=record, slot=slot, model_timing=model_timing
        )
        if isinstance(record.get("dspark_prefill"), dict):
            record["dspark_prefill"]["injection_status"] = "pending"
            self._pending_dspark = pending
        else:
            self._enqueue(pending)

    def commit_dspark_injection(
        self, forward_pass_id: int, *, status: str
    ) -> None:
        pending = self._pending_dspark
        if pending is None:
            raise RuntimeError("No DSpark prefill recording is pending")
        if pending.record.get("forward_id") != forward_pass_id:
            raise RuntimeError(
                "DSpark prefill commit forward ID mismatch: "
                f"pending={pending.record.get('forward_id')}, "
                f"commit={forward_pass_id}"
            )
        dspark = pending.record.get("dspark_prefill")
        if not isinstance(dspark, dict):
            raise RuntimeError("Pending DSpark prefill metadata is missing")
        dspark["injection_status"] = status
        pending.injection_committed = True
        if not self.gpu_timing_enabled or pending.wrapper_timing is not None:
            self._pending_dspark = None
            self._enqueue(pending)

    def attach_dspark_gpu_timing(
        self, forward_pass_id: int, timing: _ForwardGpuTiming
    ) -> None:
        pending = self._pending_dspark
        if pending is None:
            return
        if pending.record.get("forward_id") != forward_pass_id:
            raise RuntimeError(
                "DSpark prefill GPU timing forward ID mismatch: "
                f"pending={pending.record.get('forward_id')}, "
                f"timing={forward_pass_id}"
            )
        pending.wrapper_timing = timing
        if pending.injection_committed:
            self._pending_dspark = None
            self._enqueue(pending)

    def abort(self) -> None:
        slot = self._active_slot
        self._active = False
        self._active_slot = None
        if slot is not None:
            self._free_slots.put(slot)

    def _enqueue(self, pending: _PendingRecord) -> None:
        with self._stats_lock:
            if self._closed:
                self.dropped_after_close += 1
                if pending.slot is not None:
                    self._free_slots.put(pending.slot)
                return
            try:
                self._queue.put_nowait(pending)
            except queue.Full:
                self.dropped_records += 1
                if pending.slot is not None:
                    self._free_slots.put(pending.slot)

    def _run(self) -> None:
        pending = 0
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    break
                try:
                    if item.slot is not None:
                        item.slot.event.synchronize()
                        offset = self._binary_file.tell()
                        self._binary_file.write(
                            memoryview(item.slot.cpu_uint16.numpy())
                        )
                        item.record["moe_counts"] = {
                            "storage": self.binary_path.name,
                            "offset_bytes": offset,
                            "shape": self._moe_shape,
                            "dtype": "uint16",
                        }
                    _add_gpu_timing_fields(
                        item.record,
                        model_timing=item.model_timing,
                        wrapper_timing=item.wrapper_timing,
                    )
                    self._jsonl_file.write(
                        json.dumps(item.record, separators=(",", ":")) + "\n"
                    )
                    self.written_records += 1
                    pending += 1
                    if pending >= self._flush_interval:
                        self._binary_file.flush()
                        self._jsonl_file.flush()
                        pending = 0
                except Exception:
                    self.write_errors += 1
                    if not self._write_error_logged:
                        logger.exception("Prefill workload recorder write failed")
                        self._write_error_logged = True
                finally:
                    if item.slot is not None:
                        self._free_slots.put(item.slot)
            finally:
                self._queue.task_done()
        with self._stats_lock:
            dropped_records = self.dropped_records
            dropped_after_close = self.dropped_after_close
        stats_record = {
            "schema_version": 1,
            "record_type": "recorder_stats",
            "timestamp_ns": time.time_ns(),
            "output_dir": str(self.output_dir),
            "written_records": self.written_records,
            "dropped_records": dropped_records,
            "dropped_moe_counts": self.dropped_moe_counts,
            "dropped_after_close": dropped_after_close,
            "write_errors": self.write_errors,
            "closed": True,
        }
        try:
            self._jsonl_file.write(
                json.dumps(stats_record, separators=(",", ":")) + "\n"
            )
        except Exception:
            self.write_errors += 1
            logger.exception("Prefill workload recorder failed to write final stats")
        self._binary_file.flush()
        self._jsonl_file.flush()

    def close(self) -> None:
        pending_dspark = self._pending_dspark
        if pending_dspark is not None:
            dspark = pending_dspark.record.get("dspark_prefill")
            if isinstance(dspark, dict):
                dspark["injection_status"] = "not_committed"
            self._pending_dspark = None
            self._queue.put(pending_dspark)
        with self._stats_lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(self._STOP)
        self._thread.join()
        self._binary_file.close()
        self._jsonl_file.close()
        if _PREFILL_WORKLOAD_RECORDER is self:
            set_global_prefill_workload_recorder(None)
        if self.dropped_records:
            logger.warning(
                "Prefill workload recorder dropped %d records because "
                "its queue was full",
                self.dropped_records,
            )
        if self.dropped_moe_counts:
            logger.warning(
                "Prefill workload recorder omitted MoE counts for %d records "
                "because all copy slots were busy",
                self.dropped_moe_counts,
            )
