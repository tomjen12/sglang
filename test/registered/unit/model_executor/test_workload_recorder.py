"""CPU-only tests for the per-forward workload census."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.workload_recorder import (
    WorkloadRecorder,
    build_workload_record,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _CpuMirror:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


def _batch(
    mode,
    *,
    batch_size,
    seq_lens=None,
    query_lens=None,
    prefix_lens=None,
    num_tokens_per_req=None,
    rids=None,
    original_input_lens=None,
    remaining_prefill_tokens=None,
    final_chunks=None,
    prefill_chunk_indices=None,
    context_flags=None,
    cache_device=None,
    cache_host=None,
    cache_storage=None,
    cache_recompute=None,
    is_prefill_only=False,
    global_forward_mode=None,
    global_num_tokens=None,
    original_global_num_tokens=None,
    global_num_tokens_for_logprob=None,
    global_num_token_non_padded=None,
    attn_tp_sequence_sharded=False,
    spec_algorithm=None,
):
    spec_info = (
        None
        if num_tokens_per_req is None
        else SimpleNamespace(num_tokens_per_req=num_tokens_per_req)
    )
    return SimpleNamespace(
        forward_mode=mode,
        batch_size=batch_size,
        seq_lens_cpu=None if seq_lens is None else _CpuMirror(seq_lens),
        extend_seq_lens_cpu=query_lens,
        extend_prefix_lens_cpu=prefix_lens,
        spec_info=spec_info,
        spec_algorithm=spec_algorithm,
        rids=rids,
        original_input_lens_cpu=original_input_lens,
        remaining_prefill_tokens_cpu=remaining_prefill_tokens,
        is_final_prefill_chunk_cpu=final_chunks,
        prefill_chunk_indices_cpu=prefill_chunk_indices,
        is_context_request_cpu=context_flags,
        cache_device_hit_tokens_cpu=cache_device,
        cache_host_hit_tokens_cpu=cache_host,
        cache_storage_hit_tokens_cpu=cache_storage,
        cache_recompute_tokens_cpu=cache_recompute,
        is_prefill_only=is_prefill_only,
        global_forward_mode=global_forward_mode,
        global_num_tokens_cpu=global_num_tokens,
        original_global_num_tokens_cpu=original_global_num_tokens,
        global_num_tokens_for_logprob_cpu=global_num_tokens_for_logprob,
        global_num_token_non_padded_cpu=global_num_token_non_padded,
        attn_tp_sequence_sharded=attn_tp_sequence_sharded,
    )


def _record(batch):
    return build_workload_record(
        batch,
        forward_pass_id=7,
        tp_rank=0,
        pp_rank=0,
        gpu_id=0,
    )


class TestWorkloadShape(CustomTestCase):
    def test_decode_shape(self):
        record = _record(
            _batch(ForwardMode.DECODE, batch_size=2, seq_lens=[10, 20])
        )
        self.assertEqual(record["mode"], "DECODE")
        self.assertEqual(record["logical_batch_size"], 2)
        self.assertEqual(record["context"]["requests"], 0)
        self.assertEqual(record["generation"]["query_lens"], [1, 1])
        self.assertEqual(record["generation"]["kv_lens"], [10, 20])
        self.assertEqual(record["generation"]["sum_query_x_kv"], 30)

    def test_extend_shape_and_cached_prefix(self):
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
                cache_recompute=[8, 4],
                is_prefill_only=True,
            )
        )
        self.assertEqual(record["schema_version"], 3)
        self.assertEqual(record["record_type"], "forward")
        self.assertEqual(record["request_ids"], ["request-a", "request-b"])
        self.assertEqual(
            record["context"]["request_ids"], ["request-a", "request-b"]
        )
        self.assertEqual(record["context"]["original_input_tokens"], [12, 20])
        self.assertEqual(
            record["context"]["remaining_prefill_tokens"], [2, 10]
        )
        self.assertEqual(record["context"]["is_final_chunk"], [False, False])
        self.assertEqual(record["context"]["chunk_indices"], [0, 2])
        self.assertEqual(
            record["context"]["cache"]["device_hit_tokens"], [2, 3]
        )
        self.assertEqual(
            record["context"]["cache"]["host_hit_tokens"], [0, 2]
        )
        self.assertEqual(
            record["context"]["cache"]["storage_hit_tokens"], [0, 1]
        )
        self.assertEqual(
            record["context"]["cache"]["recompute_tokens"], [8, 4]
        )
        self.assertTrue(record["is_prefill_only"])
        self.assertEqual(record["context"]["query_lens"], [8, 4])
        self.assertEqual(record["context"]["kv_lens"], [10, 10])
        self.assertEqual(record["context"]["sum_query_x_kv"], 120)
        self.assertEqual(record["cached_prefix_tokens"], 8)

    def test_missing_request_ids_uses_empty_list(self):
        record = _record(
            _batch(ForwardMode.DECODE, batch_size=1, seq_lens=[10])
        )
        self.assertEqual(record["request_ids"], [])

    def test_mixed_shape_splits_prefill_and_decode(self):
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
        self.assertEqual(record["context"]["requests"], 1)
        self.assertEqual(record["context"]["query_lens"], [16])
        self.assertEqual(record["context"]["kv_lens"], [96])
        self.assertEqual(record["generation"]["requests"], 2)
        self.assertEqual(record["generation"]["kv_lens"], [100, 120])
        self.assertEqual(record["context"]["request_ids"], ["prefill"])
        self.assertEqual(
            record["generation"]["request_ids"], ["decode-a", "decode-b"]
        )

    def test_spec_decode_query_width(self):
        record = _record(
            _batch(
                ForwardMode.TARGET_VERIFY,
                batch_size=2,
                seq_lens=[10, 20],
                num_tokens_per_req=3,
                rids=["request-a", "request-b"],
                spec_algorithm=SimpleNamespace(
                    name="EAGLE", is_speculative=lambda: True
                ),
            )
        )
        self.assertEqual(record["generation"]["query_lens"], [3, 3])
        self.assertEqual(record["generation"]["sum_query_x_kv"], 90)
        self.assertTrue(record["speculative"]["enabled"])
        self.assertTrue(record["speculative"]["active_this_forward"])
        self.assertEqual(record["speculative"]["tokens_per_request"], 3)
        self.assertEqual(record["speculative"]["algorithm"], "EAGLE")

    def test_distributed_shape(self):
        record = _record(
            _batch(
                ForwardMode.DECODE,
                batch_size=1,
                seq_lens=[10],
                global_forward_mode=ForwardMode.DECODE,
                global_num_tokens=[1, 0],
                original_global_num_tokens=[1, 0],
                global_num_tokens_for_logprob=[1, 0],
                global_num_token_non_padded=1,
                attn_tp_sequence_sharded=True,
            )
        )
        self.assertEqual(
            record["distributed"]["global_forward_mode"], "DECODE"
        )
        self.assertEqual(record["distributed"]["global_num_tokens"], [1, 0])
        self.assertEqual(
            record["distributed"]["original_global_num_tokens"], [1, 0]
        )
        self.assertEqual(
            record["distributed"]["global_num_tokens_for_logprob"], [1, 0]
        )
        self.assertEqual(
            record["distributed"]["global_num_token_non_padded"], 1
        )
        self.assertTrue(record["distributed"]["attn_tp_sequence_sharded"])


class TestWorkloadWriter(unittest.TestCase):
    def test_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "workload.jsonl"
            writer = WorkloadRecorder(str(path), flush_interval=1)
            writer.write({"mode": "DECODE", "logical_batch_size": 2})
            writer.close()

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["mode"], "DECODE")
            stats = json.loads(lines[1])
            self.assertEqual(stats["record_type"], "recorder_stats")
            self.assertEqual(stats["written_records"], 1)
            self.assertEqual(stats["dropped_records"], 0)
            self.assertEqual(stats["dropped_after_close"], 0)
            self.assertEqual(stats["write_errors"], 0)
            self.assertTrue(stats["closed"])
            self.assertEqual(stats["path"], str(path))


if __name__ == "__main__":
    unittest.main()
