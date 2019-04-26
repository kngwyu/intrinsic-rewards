from rainy.envs import ParallelEnv
from rainy.lib import RolloutSampler, RolloutStorage
from rainy.net import DummyRnn, RnnBlock, RnnState
from rainy.prelude import Array, State
from rainy.utils import Device
import torch
from torch import Tensor
from typing import List, NamedTuple, Optional


class RndRolloutStorage(RolloutStorage[State]):
    def __init__(self, nsteps: int, nworkers: int, device: Device) -> None:
        super().__init__(nsteps, nworkers, device)
        self.pseudo_rewards: List[Array[float]] = []
        self.pseudo_values: List[Tensor] = []
        self.pseudo_returns = self.device.zeros((nsteps + 1, nworkers))

    def reset(self) -> None:
        super().reset()
        self.pseudo_rewards.clear()
        self.pseudo_values.clear()

    def batch_pseudo_values(self) -> Tensor:
        return torch.cat(self.values[:self.nsteps])

    def batch_pseudo_rewards(self) -> Tensor:
        return self.device.tensor(self.pseudo_rewards).flatten()

    def batch_pseudo_returns(self) -> Tensor:
        return self.pseudo_returns[:self.nsteps].flatten()

    def calc_ac_pseudo_returns(self, next_value: Tensor, gamma: float, use_mask: bool) -> None:
        self.pseudo_returns[-1] = next_value
        self.pseudo_values.append(next_value)
        rewards = self.device.tensor(self.pseudo_rewards)
        masks = self.masks if use_mask else self.device.ones(self.nsteps + 1)

        for i in reversed(range(self.nsteps)):
            self.returns[i] = self.returns[i + 1] * gamma * masks[i + 1] + rewards[i]

    def calc_gae_pseudo_returns(
            self,
            next_value: Tensor,
            gamma: float,
            tau: float,
            use_mask: bool
    ) -> None:
        self.pseudo_returns[-1] = next_value
        self.pseudo_values.append(next_value)
        rewards = self.device.tensor(self.pseudo_rewards)
        masks = self.masks if use_mask else self.device.ones(self.nsteps + 1)
        gae = self.device.zeros(self.nworkers)
        for i in reversed(range(self.nsteps)):
            td_error = rewards[i] + \
                gamma * self.pseudo_values[i + 1] * masks[i + 1] - self.pseudo_values[i]
            gae = td_error + gamma * tau * masks[i] * gae
            self.pseudo_returns[i] = gae + self.pseudo_values[i]


class RndRolloutBatch(NamedTuple):
    states: Tensor
    actions: Tensor
    masks: Tensor
    returns: Tensor
    values: Tensor
    old_log_probs: Tensor
    advantages: Tensor
    pseudo_values: Tensor
    pseudo_returns: Tensor
    rnn_init: RnnState


class RndRolloutSampler(RolloutSampler):
    def __init__(
            self,
            storage: RndRolloutStorage,
            penv: ParallelEnv,
            minibatch_size: int,
            rnn: RnnBlock = DummyRnn(),
            adv_normalize_eps: Optional[float] = None,
    ) -> None:
        super().__init__(storage, penv, minibatch_size, rnn, adv_normalize_eps)
        self.pseudo_returns = storage.batch_pseudo_returns()
        self.pseudo_rewards = storage.batch_pseudo_rewards()
        self.pseudo_values = storage.batch_pseudo_values()
