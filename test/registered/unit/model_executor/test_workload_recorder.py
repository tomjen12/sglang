"""CPU-only tests for the prefill workload recorder."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.model_executor.workload_recorder import (
    ForwardWorkloadRecorder,
    PrefillWorkloadRecorder,
    _MoeCopySlot,
    _PendingRecord,
    build_forward_workload_record,
    build_prefill_workload_record,
    get_moe_layer_ids,
    prefill_workload_execution_context,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _batch(
    mode,
    *,
    batch_size,
    query_lens=None,
    prefix_lens=None,
    rids=None,
    original_input_lens=None,
    remaining_prefill_tokens=None,
    final_chunks=None,
    prefill_chunk_indices=None,
    context_flags=None,
    cache_device=None,
    cache_host=None,
    cache_storage=None,
    mamba_slots=None,
    mamba_track_mask=None,
    mamba_track_seqlens=None,
    mamba_cow_count=0,
    mamba_clear_count=0,
):
    return SimpleNamespace(
        forward_mode=mode,
        batch_size=batch_size,
        extend_seq_lens_cpu=query_lens,
        extend_prefix_lens_cpu=prefix_lens,
        rids=rids,
        original_input_lens_cpu=original_input_lens,
        remaining_prefill_tokens_cpu=remaining_prefill_tokens,
        is_final_prefill_chunk_cpu=final_chunks,
        prefill_chunk_indices_cpu=prefill_chunk_indices,
        is_context_request_cpu=context_flags,
        cache_device_hit_tokens_cpu=cache_device,
        cache_host_hit_tokens_cpu=cache_host,
        cache_storage_hit_tokens_cpu=cache_storage,
        mamba_state_slot_present_cpu=mamba_slots,
        mamba_track_mask_cpu=mamba_track_mask,
        mamba_track_seqlens_cpu=mamba_track_seqlens,
        mamba_cow_count_cpu=mamba_cow_count,
        mamba_clear_count_cpu=mamba_clear_count,
    )


def _record(batch):
    return build_prefill_workload_record(
        batch,
        forward_pass_id=7,
        tp_rank=0,
        pp_rank=0,
        gpu_id=0,
    )


class TestPrefillWorkloadShape(CustomTestCase):
    def test_kimi_k3_moe_layer_layout(self):
        self.assertEqual(
            get_moe_layer_ids(93, 1, 1, has_experts=True),
            list(range(1, 93)),
        )

    def test_decode_is_not_recorded(self):
        self.assertIsNone(_record(_batch(ForwardMode.DECODE, batch_size=2)))

    def test_extend_request_schema(self):
        record = _record(
            _batch(
                ForwardMode.EXTEND,
                batch_size=2,
                query_lens=[8, 4],
                prefix_lens=[2, 6],
                rids=["request-a", "request-b"],
                original_input_lens=[12, 20],
                remaining_prefill_tokens=[2, 10],
                final_chunks=[False, False],
                prefill_chunk_indices=[0, 2],
                context_flags=[True, True],
                cache_device=[2, 3],
                cache_host=[0, 2],
                cache_storage=[0, 1],
            )
        )

        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["record_type"], "prefill_forward")
        self.assertEqual(record["forward_id"], 7)
        self.assertEqual(record["logical_batch_size"], 2)
        self.assertEqual(record["generation_requests"], [])
        self.assertEqual(
            record["requests"],
            [
                {
                    "request_id": "request-a",
                    "query_size": 8,
                    "prefix_kv_size": 2,
                    "original_prompt_size": 12,
                    "remaining_prefill_tokens": 2,
                    "chunk_index": 0,
                    "is_final_chunk": False,
                    "cache_hit": {"device": 2, "host": 0, "storage": 0},
                },
                {
                    "request_id": "request-b",
                    "query_size": 4,
                    "prefix_kv_size": 6,
                    "original_prompt_size": 20,
                    "remaining_prefill_tokens": 10,
                    "chunk_index": 2,
                    "is_final_chunk": False,
                    "cache_hit": {"device": 3, "host": 2, "storage": 1},
                },
            ],
        )

    def test_mixed_keeps_generation_shape(self):
        record = _record(
            _batch(
                ForwardMode.MIXED,
                batch_size=3,
                query_lens=[16, 1, 1],
                prefix_lens=[80, 99, 119],
                rids=["prefill", "decode-a", "decode-b"],
                context_flags=[True, False, False],
            )
        )

        self.assertEqual(len(record["requests"]), 1)
        self.assertEqual(
            record["generation_requests"],
            [
                {
                    "request_id": "decode-a",
                    "query_size": 1,
                    "prefix_kv_size": 99,
                },
                {
                    "request_id": "decode-b",
                    "query_size": 1,
                    "prefix_kv_size": 119,
                },
            ],
        )

    def test_mixed_without_prefill_is_not_recorded(self):
        self.assertIsNone(
            _record(
                _batch(
                    ForwardMode.MIXED,
                    batch_size=1,
                    query_lens=[1],
                    prefix_lens=[99],
                    context_flags=[False],
                )
            )
        )

    def test_kda_state_metadata(self):
        record = _record(
            _batch(
                ForwardMode.EXTEND,
                batch_size=2,
                query_lens=[8, 4],
                prefix_lens=[0, 8],
                context_flags=[True, True],
                mamba_slots=[True, True],
                mamba_track_mask=[True, False],
                mamba_track_seqlens=[8, -1],
                mamba_cow_count=1,
                mamba_clear_count=1,
            )
        )
        self.assertEqual(
            record["kda_state"],
            {
                "enabled": True,
                "slot_present": [True, True],
                "tracking_enabled": True,
                "track_mask": [True, False],
                "track_seqlens": [8, -1],
                "cow_count": 1,
                "clear_count": 1,
            },
        )

    def test_dspark_prefill_execution_context(self):
        batch = _batch(
            ForwardMode.EXTEND,
            batch_size=1,
            query_lens=[8],
            prefix_lens=[16],
            context_flags=[True],
        )
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch.return_hidden_states_before_norm = False
        batch.spec_algorithm = SpeculativeAlgorithm.DSPARK
        dspark = {
            "enabled": True,
            "target_hidden_projection_enabled": True,
            "target_hidden_shape": None,
            "target_hidden_dtype": None,
            "kv_injection_backend": "unified_kv_triton",
            "kv_injection_token_count": 8,
            "tp_sync_enabled": True,
            "draft_kv_injection_enabled": True,
        }
        with prefill_workload_execution_context(
            {"wrapper": "dspark_prefill", "dspark_prefill": dspark}
        ):
            record = _record(batch)

        self.assertEqual(
            record["execution_context"],
            {
                "wrapper": "dspark_prefill",
                "speculative_algorithm": "DSPARK",
                "capture_hidden_mode": "FULL",
                "return_hidden_states_before_norm": False,
                "record_scope": "target_model_forward",
            },
        )
        self.assertEqual(record["dspark_prefill"], dspark)
        self.assertIsNone(
            _record(batch)["execution_context"]["wrapper"],
            "wrapper context must not leak to the next forward",
        )


class TestAllForwardWorkloadRecorder(unittest.TestCase):
    def test_every_forward_mode_is_recorded(self):
        for mode in ForwardMode:
            batch = _batch(mode, batch_size=0)
            batch.input_ids = None
            batch.is_prefill_only = False
            batch.spec_info = None
            batch.spec_algorithm = None
            batch.global_forward_mode = None
            record = build_forward_workload_record(
                batch,
                forward_pass_id=1,
                tp_rank=0,
                pp_rank=0,
                gpu_id=0,
                model_role="target",
                draft_model_idx=None,
            )
            self.assertEqual(record["mode"], mode.name)

    def test_mixed_composition(self):
        batch = _batch(
            ForwardMode.MIXED,
            batch_size=3,
            query_lens=[16, 1, 1],
            prefix_lens=[80, 99, 119],
            rids=["prefill", "decode-a", "decode-b"],
            context_flags=[True, False, False],
        )
        batch.is_prefill_only = False
        batch.spec_info = None
        batch.spec_algorithm = None
        batch.global_forward_mode = None
        record = build_forward_workload_record(
            batch,
            forward_pass_id=9,
            tp_rank=0,
            pp_rank=0,
            gpu_id=0,
            model_role="target",
            draft_model_idx=None,
        )
        self.assertEqual(record["mode"], "MIXED")
        self.assertEqual(record["prefill"]["requests"], 1)
        self.assertEqual(record["prefill"]["sum_query_tokens"], 16)
        self.assertEqual(record["decode"]["requests"], 2)
        self.assertEqual(record["decode"]["sum_query_tokens"], 2)

    def test_writer_outputs_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = ForwardWorkloadRecorder(tmpdir)
            writer.write({"schema_version": 1, "record_type": "forward"})
            writer.close()
            lines = (
                Path(tmpdir) / "forwards.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(json.loads(lines[0])["record_type"], "forward")
            self.assertEqual(json.loads(lines[1])["record_type"], "recorder_stats")


class TestPrefillWorkloadWriter(unittest.TestCase):
    def test_dspark_record_waits_for_injection_commit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = PrefillWorkloadRecorder(tmpdir, flush_interval=1)
            record = {
                "record_type": "prefill_forward",
                "forward_id": 7,
                "moe_counts": None,
                "dspark_prefill": {"enabled": True},
            }
            writer.begin(record)
            writer.finish(record)

            self.assertIsNotNone(writer._pending_dspark)
            self.assertTrue(writer._queue.empty())

            writer.commit_dspark_injection(7, status="completed")
            writer.close()

            first_line = json.loads(
                (Path(tmpdir) / "prefill_forwards.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(
                first_line["dspark_prefill"]["injection_status"],
                "completed",
            )

    def test_writes_jsonl_and_stats_without_moe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = PrefillWorkloadRecorder(tmpdir, flush_interval=1)
            record = {"record_type": "prefill_forward", "moe_counts": None}
            writer.begin(record)
            writer.finish(record)
            writer.close()

            output_dir = Path(tmpdir)
            lines = (
                output_dir / "prefill_forwards.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(
                json.loads(lines[0])["record_type"], "prefill_forward"
            )
            self.assertIsNone(json.loads(lines[0])["moe_counts"])
            stats = json.loads(lines[1])
            self.assertEqual(stats["record_type"], "recorder_stats")
            self.assertEqual(stats["written_records"], 1)
            self.assertEqual(stats["dropped_records"], 0)
            self.assertEqual(stats["dropped_moe_counts"], 0)
            self.assertEqual(stats["write_errors"], 0)
            self.assertEqual(
                (output_dir / "moe_counts.uint16.bin").stat().st_size, 0
            )

    def test_writes_uint16_sidecar_and_reference(self):
        class _CompletedEvent:
            def synchronize(self):
                pass

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = PrefillWorkloadRecorder(tmpdir, flush_interval=1)
            writer._moe_shape = [2, 3]
            cpu_counts = torch.arange(6, dtype=torch.uint16).reshape(2, 3)
            slot = _MoeCopySlot(
                gpu_counts=None,
                gpu_uint16=None,
                cpu_uint16=cpu_counts,
                event=_CompletedEvent(),
            )
            record = {"record_type": "prefill_forward", "moe_counts": None}
            writer._enqueue(_PendingRecord(record=record, slot=slot))
            writer.close()

            output_dir = Path(tmpdir)
            self.assertEqual(
                (output_dir / "moe_counts.uint16.bin").read_bytes(),
                cpu_counts.numpy().tobytes(),
            )
            first_line = json.loads(
                (output_dir / "prefill_forwards.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(
                first_line["moe_counts"],
                {
                    "storage": "moe_counts.uint16.bin",
                    "offset_bytes": 0,
                    "shape": [2, 3],
                    "dtype": "uint16",
                },
            )


if __name__ == "__main__":
    unittest.main()
