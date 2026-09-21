import gzip
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.extend_kernel_tracer import ExtendKernelTracer


class _FakeProfiler:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def export_chrome_trace(self, path):
        Path(path).write_text('{"traceEvents":[]}', encoding="utf-8")


class TestExtendKernelTracer(unittest.TestCase):
    def test_records_compressed_trace_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = _FakeProfiler()
            tracer = ExtendKernelTracer(
                enabled=True,
                output_dir=tmpdir,
                tp_rank=0,
                device=torch.device("cuda:0"),
            )
            with (
                patch("torch.cuda.synchronize") as synchronize,
                patch("torch.profiler.profile", return_value=profiler),
                patch(
                    "torch.profiler.record_function",
                    side_effect=lambda _: nullcontext(),
                ),
            ):
                with tracer.capture(
                    source_forward_id=42,
                    request_ids=["request-1"],
                    query_sizes=[512],
                    prefix_kv_sizes=[4096],
                ):
                    pass

            self.assertTrue(profiler.started)
            self.assertTrue(profiler.stopped)
            self.assertEqual(synchronize.call_count, 2)
            trace_path = (
                Path(tmpdir)
                / "agentx_prefill_source_42-TP-0.trace.json.gz"
            )
            with gzip.open(trace_path, "rt", encoding="utf-8") as trace:
                self.assertEqual(json.load(trace), {"traceEvents": []})
            manifest = json.loads(
                (Path(tmpdir) / "manifest.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["source_forward_id"], 42)
            self.assertEqual(manifest["logical_batch_size"], 1)
            self.assertEqual(manifest["query_sizes"], [512])
            self.assertEqual(manifest["prefix_kv_sizes"], [4096])
            self.assertEqual(manifest["status"], "completed")

    def test_nonzero_rank_is_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "traces"
            tracer = ExtendKernelTracer(
                enabled=True,
                output_dir=str(output_dir),
                tp_rank=1,
                device=torch.device("cuda:1"),
            )
            with patch("torch.profiler.profile", Mock()) as profile:
                with tracer.capture(
                    source_forward_id=1,
                    request_ids=[],
                    query_sizes=[],
                    prefix_kv_sizes=[],
                ):
                    pass
            profile.assert_not_called()
            self.assertFalse(output_dir.exists())

    def test_missing_arm_file_skips_profiler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracer = ExtendKernelTracer(
                enabled=True,
                output_dir=tmpdir,
                tp_rank=0,
                device=torch.device("cuda:0"),
                arm_file=str(Path(tmpdir) / "trace.armed"),
            )
            with patch("torch.profiler.profile", Mock()) as profile:
                with tracer.capture(
                    source_forward_id=1,
                    request_ids=["request-1"],
                    query_sizes=[128],
                    prefix_kv_sizes=[1024],
                ):
                    pass
            profile.assert_not_called()
            self.assertFalse((Path(tmpdir) / "manifest.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
