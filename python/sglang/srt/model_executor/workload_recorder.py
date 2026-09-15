"""Low-overhead, opt-in model-forward workload census.

This records CPU-side batch-shape metadata only. It does not synchronize the
device or measure latency; a later profiling run can use the census to choose
representative forward shapes.
"""

from __future__ import annotations

import atexit
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.step_span_utils import _decode_query_width

logger = logging.getLogger(__name__)


def _phase_shape(query_lens: List[int], kv_lens: List[int]) -> Dict[str, Any]:
    if not query_lens:
        return {
            "requests": 0,
            "query_lens": [],
            "kv_lens": [],
            "sum_query_tokens": 0,
            "sum_kv_tokens": 0,
            "sum_query_sq": 0,
            "sum_query_x_kv": 0,
            "min_kv_length": 0,
            "max_kv_length": 0,
        }

    return {
        "requests": len(query_lens),
        "query_lens": query_lens,
        "kv_lens": kv_lens,
        "sum_query_tokens": sum(query_lens),
        "sum_kv_tokens": sum(kv_lens),
        "sum_query_sq": sum(nq * nq for nq in query_lens),
        "sum_query_x_kv": sum(
            nq * nkv for nq, nkv in zip(query_lens, kv_lens)
        ),
        "min_kv_length": min(kv_lens),
        "max_kv_length": max(kv_lens),
    }


def build_workload_record(
    forward_batch,
    *,
    forward_pass_id: int,
    tp_rank: int,
    pp_rank: int,
    gpu_id: int,
) -> Dict[str, Any]:
    """Build a JSON-serializable snapshot before graph replay can pad metadata."""
    mode = forward_batch.forward_mode
    context_query_lens: List[int] = []
    context_kv_lens: List[int] = []
    generation_query_lens: List[int] = []
    generation_kv_lens: List[int] = []
    context_request_ids: List[str] = []
    generation_request_ids: List[str] = []
    original_input_tokens: List[int] = []
    remaining_prefill_tokens: List[int] = []
    is_final_chunk: List[bool] = []
    chunk_indices: List[int] = []
    device_hit_tokens: List[int] = []
    host_hit_tokens: List[int] = []
    storage_hit_tokens: List[int] = []
    recompute_tokens: List[int] = []
    cached_prefix_tokens: Optional[int] = None
    request_ids = list(getattr(forward_batch, "rids", None) or [])

    if mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
        if forward_batch.seq_lens_cpu is not None:
            generation_kv_lens = [
                int(x) for x in forward_batch.seq_lens_cpu.tolist()
            ]
            generation_query_lens = [
                _decode_query_width(forward_batch)
            ] * len(generation_kv_lens)
            generation_request_ids = request_ids[: len(generation_kv_lens)]
    elif mode in (
        ForwardMode.EXTEND,
        ForwardMode.DRAFT_EXTEND_V2,
        ForwardMode.MIXED,
    ):
        query_lens = forward_batch.extend_seq_lens_cpu
        prefix_lens = forward_batch.extend_prefix_lens_cpu
        if query_lens is not None and prefix_lens is not None:
            cached_prefix_tokens = sum(int(x) for x in prefix_lens)
            context_flags = getattr(
                forward_batch, "is_context_request_cpu", None
            )
            original_lens = getattr(
                forward_batch, "original_input_lens_cpu", None
            )
            remaining_lens = getattr(
                forward_batch, "remaining_prefill_tokens_cpu", None
            )
            final_chunks = getattr(
                forward_batch, "is_final_prefill_chunk_cpu", None
            )
            prefill_chunk_indices = getattr(
                forward_batch, "prefill_chunk_indices_cpu", None
            )
            cache_device = getattr(
                forward_batch, "cache_device_hit_tokens_cpu", None
            )
            cache_host = getattr(
                forward_batch, "cache_host_hit_tokens_cpu", None
            )
            cache_storage = getattr(
                forward_batch, "cache_storage_hit_tokens_cpu", None
            )
            cache_recompute = getattr(
                forward_batch, "cache_recompute_tokens_cpu", None
            )
            for i, (prefix_len, query_len) in enumerate(
                zip(prefix_lens, query_lens)
            ):
                nq = int(query_len)
                nkv = int(prefix_len) + nq
                is_context = (
                    bool(context_flags[i])
                    if context_flags is not None
                    else not (mode == ForwardMode.MIXED and nq == 1)
                )
                rid = request_ids[i] if i < len(request_ids) else ""
                if not is_context:
                    generation_query_lens.append(nq)
                    generation_kv_lens.append(nkv)
                    generation_request_ids.append(rid)
                else:
                    context_query_lens.append(nq)
                    context_kv_lens.append(nkv)
                    context_request_ids.append(rid)
                    original_input_tokens.append(
                        int(original_lens[i]) if original_lens is not None else nkv
                    )
                    remaining_prefill_tokens.append(
                        int(remaining_lens[i]) if remaining_lens is not None else 0
                    )
                    is_final_chunk.append(
                        bool(final_chunks[i]) if final_chunks is not None else True
                    )
                    chunk_indices.append(
                        int(prefill_chunk_indices[i])
                        if prefill_chunk_indices is not None
                        else 0
                    )
                    device_hit_tokens.append(
                        int(cache_device[i]) if cache_device is not None else 0
                    )
                    host_hit_tokens.append(
                        int(cache_host[i]) if cache_host is not None else 0
                    )
                    storage_hit_tokens.append(
                        int(cache_storage[i]) if cache_storage is not None else 0
                    )
                    recompute_tokens.append(
                        int(cache_recompute[i])
                        if cache_recompute is not None
                        else nq
                    )

    context = _phase_shape(context_query_lens, context_kv_lens)
    context.update(
        {
            "request_ids": context_request_ids,
            "original_input_tokens": original_input_tokens,
            "remaining_prefill_tokens": remaining_prefill_tokens,
            "is_final_chunk": is_final_chunk,
            "chunk_indices": chunk_indices,
            "cache": {
                "device_hit_tokens": device_hit_tokens,
                "host_hit_tokens": host_hit_tokens,
                "storage_hit_tokens": storage_hit_tokens,
                "recompute_tokens": recompute_tokens,
            },
        }
    )
    generation = _phase_shape(generation_query_lens, generation_kv_lens)
    generation["request_ids"] = generation_request_ids

    spec_info = getattr(forward_batch, "spec_info", None)
    spec_algorithm = getattr(forward_batch, "spec_algorithm", None)
    spec_algorithm_name = getattr(spec_algorithm, "name", None)
    if spec_algorithm_name is None and spec_algorithm is not None:
        spec_algorithm_name = str(spec_algorithm)
    spec_enabled = bool(
        spec_algorithm is not None
        and getattr(spec_algorithm, "is_speculative", lambda: False)()
    )
    global_forward_mode = getattr(forward_batch, "global_forward_mode", None)

    return {
        "schema_version": 3,
        "record_type": "forward",
        "timestamp_ns": time.time_ns(),
        "forward_pass_id": forward_pass_id,
        "mode": mode.name,
        "logical_batch_size": int(forward_batch.batch_size),
        # Stable SGLang request IDs correlate chunked EXTEND passes and the
        # following DECODE passes without recording prompt or token contents.
        "request_ids": request_ids,
        "context": context,
        "generation": generation,
        "cached_prefix_tokens": cached_prefix_tokens,
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
            "global_num_tokens_for_logprob": getattr(
                forward_batch, "global_num_tokens_for_logprob_cpu", None
            ),
            "global_num_token_non_padded": getattr(
                forward_batch, "global_num_token_non_padded_cpu", None
            ),
            "attn_tp_sequence_sharded": bool(
                getattr(forward_batch, "attn_tp_sequence_sharded", False)
            ),
        },
        "speculative": {
            "enabled": spec_enabled,
            "active_this_forward": spec_info is not None,
            "algorithm": spec_algorithm_name,
            "tokens_per_request": (
                int(spec_info.num_tokens_per_req)
                if spec_info is not None
                and getattr(spec_info, "num_tokens_per_req", -1) >= 0
                else None
            ),
        },
        "execution": None,
        "graph_batch_size": None,
        "tp_rank": tp_rank,
        "pp_rank": pp_rank,
        "gpu_id": gpu_id,
    }


class WorkloadRecorder:
    """Bounded, asynchronous JSONL writer for the model-forward hot path."""

    _STOP = object()

    def __init__(
        self,
        path: str,
        flush_interval: int = 256,
        queue_size: int = 65536,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8", buffering=1024 * 1024)
        self._flush_interval = flush_interval
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._closed = False
        self.dropped_records = 0
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

    def write(self, record: Dict[str, Any]) -> None:
        with self._stats_lock:
            if self._closed:
                self.dropped_after_close += 1
                return
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                self.dropped_records += 1

    def _run(self) -> None:
        pending = 0
        while True:
            record = self._queue.get()
            try:
                if record is self._STOP:
                    break
                try:
                    self._file.write(
                        json.dumps(record, separators=(",", ":")) + "\n"
                    )
                    self.written_records += 1
                    pending += 1
                    if pending >= self._flush_interval:
                        self._file.flush()
                        pending = 0
                except Exception:
                    self.write_errors += 1
                    if not self._write_error_logged:
                        logger.exception("Workload recorder failed to write JSONL")
                        self._write_error_logged = True
            finally:
                self._queue.task_done()
        with self._stats_lock:
            dropped_records = self.dropped_records
            dropped_after_close = self.dropped_after_close
        stats_record = {
            "schema_version": 3,
            "record_type": "recorder_stats",
            "timestamp_ns": time.time_ns(),
            "path": str(self.path),
            "written_records": self.written_records,
            "dropped_records": dropped_records,
            "dropped_after_close": dropped_after_close,
            "write_errors": self.write_errors,
            "closed": True,
        }
        try:
            self._file.write(
                json.dumps(stats_record, separators=(",", ":")) + "\n"
            )
        except Exception:
            self.write_errors += 1
            logger.exception("Workload recorder failed to write final stats")
        self._file.flush()

    def close(self) -> None:
        with self._stats_lock:
            if self._closed:
                return
            self._closed = True
        self._queue.put(self._STOP)
        self._thread.join()
        self._file.close()
        if self.dropped_records:
            logger.warning(
                "Workload recorder dropped %d records because its queue was full",
                self.dropped_records,
            )
