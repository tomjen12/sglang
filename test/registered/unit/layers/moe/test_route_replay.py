import unittest

import torch

from sglang.srt.layers.moe.route_replay import (
    has_moe_route_replay_provider,
    maybe_replay_topk,
    set_moe_route_replay_provider,
)


class TestMoeRouteReplay(unittest.TestCase):
    def tearDown(self):
        set_moe_route_replay_provider(None)

    def test_no_provider_is_identity(self):
        weights = torch.ones((2, 2))
        ids = torch.tensor([[0, 1], [1, 2]])
        actual_weights, actual_ids = maybe_replay_topk(3, weights, ids)
        self.assertIs(actual_weights, weights)
        self.assertIs(actual_ids, ids)
        self.assertFalse(has_moe_route_replay_provider())

    def test_provider_replaces_routing(self):
        weights = torch.ones((2, 2))
        ids = torch.tensor([[0, 1], [1, 2]])
        replacement = torch.tensor([[2, 3], [3, 4]])

        def provider(layer_id, actual_weights, actual_ids):
            self.assertEqual(layer_id, 7)
            self.assertIs(actual_weights, weights)
            self.assertIs(actual_ids, ids)
            return actual_weights, replacement

        set_moe_route_replay_provider(provider)
        actual_weights, actual_ids = maybe_replay_topk(7, weights, ids)
        self.assertIs(actual_weights, weights)
        self.assertIs(actual_ids, replacement)
        self.assertTrue(has_moe_route_replay_provider())

    def test_provider_shape_is_checked(self):
        weights = torch.ones((2, 2))
        ids = torch.zeros((2, 2), dtype=torch.int64)
        set_moe_route_replay_provider(
            lambda layer_id, actual_weights, actual_ids: (
                actual_weights,
                actual_ids[:1],
            )
        )
        with self.assertRaisesRegex(ValueError, "ids shape mismatch"):
            maybe_replay_topk(1, weights, ids)


if __name__ == "__main__":
    unittest.main()
