"""Opt-in per-forward kernel traces for live DSpark target prefills."""

from __future__ import annotations

import contextlib
import gzip
import json
import os
import shutil
import time
from pathlib import Path
from typing import Iterator, Sequence

import torch


class ExtendKernelTracer:
    """Record each qualifying EXTEND as an independent Chrome trace."""

    def __init__(
        self,
        *,
        enabled: bool,
        output_dir: str,
        tp_rank: int,
        device: torch.device,
        arm_file: str = "",
    ) -> None:
        self.enabled = bool(enabled) and tp_rank == 0
        self.tp_rank = int(tp_rank)
        self.device = device
        self.output_dir = Path(output_dir).expanduser()
        self.arm_file = (
            Path(arm_file).expanduser() if arm_file else None
        )
        self.manifest_path = self.output_dir / "manifest.jsonl"
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def capture(
        self,
        *,
        source_forward_id: int,
        request_ids: Sequence[str],
        query_sizes: Sequence[int],
        prefix_kv_sizes: Sequence[int],
    ) -> Iterator[None]:
        if not self.enabled or (
            self.arm_file is not None and not self.arm_file.is_file()
        ):
            yield
            return

        source_forward_id = int(source_forward_id)
        query_sizes = [int(value) for value in query_sizes]
        prefix_kv_sizes = [int(value) for value in prefix_kv_sizes]
        request_ids = [str(value) for value in request_ids]
        if len(query_sizes) != len(prefix_kv_sizes):
            raise ValueError(
                "EXTEND kernel trace query and prefix lengths must have equal size"
            )

        stem = (
            f"agentx_prefill_source_{source_forward_id}"
            f"-TP-{self.tp_rank}.trace.json"
        )
        raw_path = self.output_dir / stem
        trace_path = raw_path.with_suffix(raw_path.suffix + ".gz")
        label = (
            f"agentx_prefill_source_{source_forward_id}"
            f"_bs_{len(query_sizes)}"
            f"_q_{sum(query_sizes)}"
            f"_prefix_{sum(prefix_kv_sizes)}"
        )
        started_timestamp_ns = time.time_ns()
        status = "completed"

        torch.cuda.synchronize(self.device)
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,
            profile_memory=False,
        )
        profiler.start()
        try:
            with torch.profiler.record_function(label):
                yield
        except BaseException:
            status = "failed"
            raise
        finally:
            torch.cuda.synchronize(self.device)
            profiler.stop()
            try:
                profiler.export_chrome_trace(str(raw_path))
                with raw_path.open("rb") as source, gzip.open(
                    trace_path, "wb"
                ) as output:
                    shutil.copyfileobj(source, output)
            finally:
                raw_path.unlink(missing_ok=True)

            manifest_record = {
                "schema_version": 1,
                "record_type": "extend_kernel_trace",
                "source_forward_id": source_forward_id,
                "started_timestamp_ns": started_timestamp_ns,
                "completed_timestamp_ns": time.time_ns(),
                "status": status,
                "trace": trace_path.name,
                "tp_rank": self.tp_rank,
                "logical_batch_size": len(query_sizes),
                "request_ids": request_ids,
                "query_sizes": query_sizes,
                "prefix_kv_sizes": prefix_kv_sizes,
                "sum_query_tokens": sum(query_sizes),
                "sum_prefix_kv_tokens": sum(prefix_kv_sizes),
            }
            line = json.dumps(manifest_record, separators=(",", ":")) + "\n"
            fd = os.open(
                self.manifest_path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o644,
            )
            try:
                os.write(fd, line.encode("utf-8"))
            finally:
                os.close(fd)
