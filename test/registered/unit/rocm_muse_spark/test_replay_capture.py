import json
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.rocm_muse_spark.replay_capture import (
    CaptureSelector,
    ReplayBundleWriter,
    force_eager_model_runner,
    record_route_workload,
    record_tensor_descriptor,
    runtime_metadata_capture,
    snapshot_forward_batch,
    snapshot_kv_cache,
)


def test_capture_selector_skips_short_server_warmups(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_MUSE_REPLAY_CAPTURE_DIR", str(tmp_path))
    monkeypatch.setenv("SGLANG_MUSE_REPLAY_MIN_PREFIX", "8192")
    selector = CaptureSelector()

    assert selector.next(rank=0, width=8, min_prefix_length=208) is None
    writer = selector.next(rank=0, width=8, min_prefix_length=8192)

    assert writer is not None
    assert writer.step == 0


def test_force_eager_model_runner_restores_graph_runners():
    decode_graph = object()
    prefill_graph = object()
    runner = SimpleNamespace(
        decode_cuda_graph_runner=decode_graph,
        prefill_cuda_graph_runner=prefill_graph,
    )

    with force_eager_model_runner(runner):
        assert runner.decode_cuda_graph_runner is None
        assert runner.prefill_cuda_graph_runner is None

    assert runner.decode_cuda_graph_runner is decode_graph
    assert runner.prefill_cuda_graph_runner is prefill_graph


def test_metadata_manifest_is_atomic_and_incomplete_until_finish(tmp_path):
    writer = ReplayBundleWriter(tmp_path, rank=2, step=7)
    artifact = writer.write_metadata("forward/batch", {"width": 8, "prefix": 8192})

    manifest = json.loads(writer.manifest_path.read_text())
    assert manifest["complete"] is False
    assert manifest["rank"] == 2
    assert manifest["step"] == 7
    assert manifest["artifacts"] == [
        {
            "kind": "metadata",
            "name": "forward/batch",
            "path": artifact.name,
        }
    ]

    writer.finish({"target_seed": 20261117})
    manifest = json.loads(writer.manifest_path.read_text())
    assert manifest["complete"] is True
    assert manifest["metadata"]["target_seed"] == 20261117


def test_target_weights_cannot_enter_runtime_bundle(tmp_path):
    writer = ReplayBundleWriter(tmp_path, rank=0, step=0)

    with pytest.raises(ValueError, match="Target weights"):
        writer.write_tensors(
            "runtime",
            {"target.layer_0.weight": torch.zeros(1)},
        )


def test_forward_batch_snapshot_handles_msgspec_style_spec_info():
    class SpecInfo:
        __struct_fields__ = ("draft_token_num", "bonus_tokens")
        draft_token_num = 8
        bonus_tokens = torch.arange(2)

    batch = SimpleNamespace(
        input_ids=torch.arange(16),
        positions=torch.arange(16),
        batch_size=2,
        seq_lens_sum=16384,
        can_run_decode_cuda_graph=True,
        forward_mode=SimpleNamespace(name="TARGET_VERIFY"),
        capture_hidden_mode=SimpleNamespace(name="FULL"),
        spec_algorithm=SimpleNamespace(name="DFLASH"),
        spec_info=SpecInfo(),
    )

    tensors, metadata = snapshot_forward_batch(batch)

    assert tensors["spec_info.bonus_tokens"].tolist() == [0, 1]
    assert metadata["spec_info.draft_token_num"] == 8
    assert metadata["forward_mode"] == "TARGET_VERIFY"


def test_kv_snapshot_preserves_vectorized_pages():
    table = SimpleNamespace(
        ids=torch.tensor([[1, 2]], dtype=torch.int64),
        row_ids=torch.tensor([0], dtype=torch.int64),
        row_stride=2,
        entry_page_size=16,
        is_translated=True,
        v2p=None,
    )
    key = torch.arange(4 * 2).view(4, 2)
    value = key + 100
    layer = SimpleNamespace(kv=lambda: (key, value))
    view = SimpleNamespace(page_size=16, kv_cache_layout="vectorized_5d")
    sub_pool = SimpleNamespace(name="swa", layer_ids=(2,))

    class Router:
        def sub_pools(self):
            return (sub_pool,)

        def view(self, _name):
            return view

        def layer(self, _layer_id):
            return layer

    class Reader:
        def read_table(self, _plan, *, kind):
            assert kind == "swa"
            return table

    plan = SimpleNamespace(
        req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([32]),
        read_extent=0,
    )
    pool = SimpleNamespace(layer_router=Router())

    tensors, metadata = snapshot_kv_cache(
        plan, Reader(), pool, layer_ids=range(10)
    )

    assert tensors["kv.swa.storage_ids"].tolist() == [1, 2]
    assert torch.equal(tensors["kv.swa.layer_2.k"], key[[1, 2]])
    assert metadata["kv.swa.layout"] == "vectorized_5d"


def test_lightweight_runtime_capture_stores_only_descriptors_and_counts(tmp_path):
    writer = ReplayBundleWriter(tmp_path, rank=0, step=0)

    with runtime_metadata_capture(writer, "target_layers"):
        record_tensor_descriptor(
            "layer_0.input.stream", torch.empty(8, 8192, dtype=torch.bfloat16)
        )
        record_route_workload(
            layer=0,
            topk_ids=torch.tensor([[0, 1], [1, 3]], dtype=torch.int32),
            num_experts=4,
            local_start=0,
            local_experts=2,
        )

    metadata = json.loads(
        (writer.directory / "target_layers.json").read_text()
    )
    assert metadata["layer_0.input.stream"]["shape"] == [8, 8192]
    assert metadata["layer_0.routing"]["global_token_counts"] == [1, 2, 0, 1]
    assert not list(writer.directory.glob("*.safetensors"))
