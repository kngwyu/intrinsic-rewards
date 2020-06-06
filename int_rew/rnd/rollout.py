from typing import NamedTuple, Optional

from torch import Tensor

from rainy.lib.rollout import RolloutSampler
from rainy.net import RnnState
from rainy.prelude import Array
from rainy.utils import normalize_

from ..rollout import IntValueRolloutStorage


class RNDRolloutBatch(NamedTuple):
    states: Tensor
    actions: Tensor
    masks: Tensor
    returns: Tensor
    values: Tensor
    old_log_probs: Tensor
    int_values: Tensor
    int_returns: Tensor
    advantages: Tensor
    targets: Optional[Tensor]
    rnn_init: RnnState


class RNDRolloutSampler(RolloutSampler):
    def __init__(
        self,
        sampler: RolloutSampler,
        storage: IntValueRolloutStorage,
        target: Optional[Tensor],
        ext_coeff: float,
        int_coeff: float,
        adv_normalize_eps: Optional[float] = None,
    ) -> None:
        self.__dict__.update(sampler.__dict__)
        self.int_returns = storage.int_returns.flatten()
        self.int_values = storage.batch_int_values()
        int_advs = storage.int_gae[:-1].flatten().to(self.advantages.device)
        self.advantages.mul_(ext_coeff).add_(int_advs * int_coeff)
        self.targets = target
        if adv_normalize_eps is not None:
            normalize_(self.advantages, adv_normalize_eps)

    def _make_batch(self, i: Array[int]) -> RNDRolloutBatch:
        return RNDRolloutBatch(
            self.states[i],
            self.actions[i],
            self.masks[i],
            self.returns[i],
            self.values[i],
            self.old_log_probs[i],
            self.int_values[i],
            self.int_returns[i],
            self.advantages[i],
            None if self.targets is None else self.targets[i],
            self.rnn_init[i[: len(i) // self.nsteps]],
        )
