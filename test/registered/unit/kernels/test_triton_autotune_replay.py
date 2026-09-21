import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.kernels.ops.attention.fla.autotune_replay import (
    RECORD_DIR_ENV,
    REPLAY_DIR_ENV,
    STRICT_REPLAY_ENV,
    find_best_config,
    record_winner,
    replay_pruner,
)


def _config(*, bv: int):
    return SimpleNamespace(
        kwargs={"BK": 64, "BV": bv},
        num_warps=8,
        num_ctas=1,
        num_stages=2,
        maxnreg=None,
    )


class TestTritonAutotuneReplay(unittest.TestCase):
    def test_finds_best_config_through_wrappers(self):
        expected = _config(bv=64)
        wrapped = SimpleNamespace(
            fn=SimpleNamespace(fn=SimpleNamespace(best_config=expected))
        )

        self.assertIs(find_best_config(wrapped), expected)

    def test_records_and_replays_rank_config(self):
        key = {"BT": 64, "IS_VARLEN": True}
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {RECORD_DIR_ENV: tmpdir},
            clear=False,
        ):
            with patch(
                "sglang.kernels.ops.attention.fla.autotune_replay._rank",
                return_value=3,
            ):
                record_winner("kernel", key, _config(bv=64))

            value = json.loads(
                (Path(tmpdir) / "rank_3.json").read_text(encoding="utf-8")
            )
            self.assertEqual(value["rank"], 3)
            self.assertEqual(
                value["configs"][0]["config"]["kwargs"]["BV"], 64
            )

            with patch.dict(
                os.environ,
                {REPLAY_DIR_ENV: tmpdir},
                clear=False,
            ), patch(
                "sglang.kernels.ops.attention.fla.autotune_replay._rank",
                return_value=3,
            ):
                prune = replay_pruner(
                    "kernel",
                    lambda arguments: {
                        "BT": arguments["BT"],
                        "IS_VARLEN": arguments["IS_VARLEN"],
                    },
                )
                selected = prune(
                    [_config(bv=64), _config(bv=128)],
                    named_args={},
                    BT=64,
                    IS_VARLEN=True,
                )
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0].kwargs["BV"], 64)

    def test_new_kernel_falls_back_only_for_legacy_bundle(self):
        key = {"T": 64}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rank_0.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "rank": 0,
                        "configs": [
                            {
                                "kernel": "older_kernel",
                                "key": key,
                                "config": {},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {REPLAY_DIR_ENV: tmpdir},
                clear=False,
            ), patch(
                "sglang.kernels.ops.attention.fla.autotune_replay._rank",
                return_value=0,
            ):
                prune = replay_pruner(
                    "new_kernel",
                    lambda arguments: {"T": arguments["T"]},
                    allow_unrecorded_kernel=True,
                )
                configs = [_config(bv=64), _config(bv=128)]
                self.assertEqual(
                    prune(configs, named_args={}, T=64),
                    configs,
                )

    def test_strict_replay_rejects_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rank_0.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "rank": 0,
                        "configs": [
                            {
                                "kernel": "older_kernel",
                                "key": {"T": 64},
                                "config": {},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {
                    REPLAY_DIR_ENV: tmpdir,
                    STRICT_REPLAY_ENV: "1",
                },
                clear=False,
            ), patch(
                "sglang.kernels.ops.attention.fla.autotune_replay._rank",
                return_value=0,
            ):
                prune = replay_pruner(
                    "new_kernel",
                    lambda arguments: {"T": arguments["T"]},
                    allow_unrecorded_kernel=True,
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "has no recorded new_kernel config",
                ):
                    prune(
                        [_config(bv=64), _config(bv=128)],
                        named_args={},
                        T=64,
                    )


if __name__ == "__main__":
    unittest.main()
