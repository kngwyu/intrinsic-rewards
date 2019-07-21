from rainy.lib.rollout import RolloutSampler, RolloutStorage
from rainy.net import RnnState
from rainy.prelude import Array, State
from rainy.utils import Device, normalize_
import torch
from torch import Tensor
from typing import List, NamedTuple, Optional


class RndRolloutStorage(RolloutStorage[State]):
    def __init__(
            self,
            nsteps: int,
            nworkers: int,
            device: Device,
            gamma: float,
            rnd_device: Device = Device(use_cpu=True)
    ) -> None:
        super().__init__(nsteps, nworkers, device)
        self.int_rewards: List[Tensor] = []
        self.int_values: List[Tensor] = []
        self.int_gae = rnd_device.zeros((nsteps + 1, nworkers))
        self.int_returns = rnd_device.zeros((nsteps, nworkers))
        self.rnd_device = rnd_device

    def push_int_value(self, pval: Tensor) -> None:
        self.int_values.append(self.rnd_device.tensor(pval))

    def reset(self) -> None:
        super().reset()
        self.int_rewards.clear()
        self.int_values.clear()

    def batch_int_values(self) -> Tensor:
        return torch.cat(self.values[:self.nsteps])

    def calc_int_returns(
            self,
            next_value: Tensor,
            rewards: Tensor,
            gamma: float,
            lambda_: float,
            use_mask: bool = False,
    ) -> None:
        """Calcurates the GAE return of pseudo rewards
        """
        self.int_values.append(self.rnd_device.tensor(next_value))
        masks = self.masks if use_mask else self.rnd_device.ones(self.nsteps + 1)
        self.int_gae.fill_(0.0)
        for i in reversed(range(self.nsteps)):
            td_error = rewards[i] + \
                gamma * self.int_values[i + 1] * masks[i + 1] - self.int_values[i]
            self.int_gae[i] = td_error + gamma * lambda_ * masks[i] * self.int_gae[i + 1]
            self.int_returns[i] = self.int_gae[i] + self.int_values[i]


class RndRolloutBatch(NamedTuple):
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


class RndRolloutSampler(RolloutSampler):
    def __init__(
            self,
            sampler: RolloutSampler,
            storage: RndRolloutStorage,
            target: Optional[Tensor],
            ext_coeff: float,
            int_coeff: float,
            adv_normalize_eps: Optional[float] = None
    ) -> None:
        self.__dict__.update(sampler.__dict__)
        self.int_returns = storage.int_returns.flatten()
        self.int_values = storage.batch_int_values()
        int_advs = storage.int_gae[:-1].flatten().to(self.advantages.device)
        self.advantages.mul_(ext_coeff).add_(int_advs * int_coeff)
        self.targets = target
        if adv_normalize_eps is not None:
            normalize_(self.advantages, adv_normalize_eps)

    def _make_batch(self, i: Array[int]) -> RndRolloutBatch:
        return RndRolloutBatch(
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
            self.rnn_init[i[:len(i) // self.nsteps]]
        )
