"""Opt-in MoE routing override used by offline workload replay tools."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

MoeRouteReplayProvider = Callable[
    [int, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]

_PROVIDER: Optional[MoeRouteReplayProvider] = None


def set_moe_route_replay_provider(
    provider: Optional[MoeRouteReplayProvider],
) -> None:
    """Install a process-local routing provider, or clear it with ``None``."""
    global _PROVIDER
    _PROVIDER = provider


def has_moe_route_replay_provider() -> bool:
    return _PROVIDER is not None


def maybe_replay_topk(
    layer_id: int,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return recorded routing when replay is active; otherwise return inputs."""
    provider = _PROVIDER
    if provider is None:
        return topk_weights, topk_ids

    replay_weights, replay_ids = provider(layer_id, topk_weights, topk_ids)
    if replay_weights.shape != topk_weights.shape:
        raise ValueError(
            "MoE replay weights shape mismatch: "
            f"expected {tuple(topk_weights.shape)}, "
            f"got {tuple(replay_weights.shape)}"
        )
    if replay_ids.shape != topk_ids.shape:
        raise ValueError(
            "MoE replay ids shape mismatch: "
            f"expected {tuple(topk_ids.shape)}, got {tuple(replay_ids.shape)}"
        )
    if replay_weights.device != topk_weights.device:
        raise ValueError("MoE replay weights must remain on the original device")
    if replay_ids.device != topk_ids.device:
        raise ValueError("MoE replay ids must remain on the original device")
    return replay_weights, replay_ids
