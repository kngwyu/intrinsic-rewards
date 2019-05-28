from rainy.envs import ParallelEnv
from rainy.lib.rollout import RolloutSampler, RolloutStorage
from rainy.net import DummyRnn, RnnBlock, RnnState
from rainy.prelude import Array, State
from rainy.utils import Device, normalize_, RunningMeanStdTorch
import torch
from torch import Tensor
from typing import List, NamedTuple, Optional


class RewardForwardFilter:
    def __init__(self, gamma: float, nworkers: int, device: Device) -> None:
        self.gamma = gamma
        self.nonepisodic_return = device.zeros(nworkers)

    def update(self, prew: Tensor) -> Tensor:
        self.nonepisodic_return.div_(self.gamma).add_(prew)
        return self.nonepisodic_return.clone().detach()


class RndRolloutStorage(RolloutStorage[State]):
    def __init__(
            self,
            ros: RolloutStorage[State],
            nsteps: int,
            nworkers: int,
            gamma: float,
            rnd_device: Device = Device(use_cpu=True)
    ) -> None:
        self.__dict__.update(ros.__dict__)
        self.rff = RewardForwardFilter(gamma, nworkers, rnd_device)
        self.rff_rms = RunningMeanStdTorch(shape=(), device=rnd_device)
        self.pseudo_rewards: List[Tensor] = []
        self.pseudo_values: List[Tensor] = []
        self.pseudo_gae = rnd_device.zeros((nsteps + 1, nworkers))
        self.pseudo_returns = rnd_device.zeros((nsteps, nworkers))
        self.rnd_device = rnd_device

    def push_pseudo_rewards(self, prew: Tensor, pval: Tensor) -> None:
        self.pseudo_rewards.append(self.rff.update(self.rnd_device.tensor(prew)))
        self.pseudo_values.append(self.rnd_device.tensor(pval))

    def reset(self) -> None:
        super().reset()
        self.pseudo_rewards.clear()
        self.pseudo_values.clear()

    def batch_pseudo_values(self) -> Tensor:
        return torch.cat(self.values[:self.nsteps])

    def normalize_pseudo_rewards(self) -> Tensor:
        rewards = torch.cat(self.pseudo_rewards)
        self.rff_rms.update(rewards)
        return rewards.div_(self.rff_rms.var.sqrt())

    def calc_pseudo_returns(
            self,
            next_value: Tensor,
            gamma: float,
            lambda_: float,
            use_mask: bool = False,
    ) -> None:
        """Calcurates the GAE return of pseudo rewards
        """
        self.pseudo_values.append(self.rnd_device.tensor(next_value))
        rewards = self.normalize_pseudo_rewards()
        masks = self.masks if use_mask else self.rnd_device.ones(self.nsteps + 1)
        self.pseudo_gae.fill_(0.0)
        for i in reversed(range(self.nsteps)):
            td_error = rewards[i] + \
                gamma * self.pseudo_values[i + 1] * masks[i + 1] - self.pseudo_values[i]
            self.pseudo_gae[i] = td_error + gamma * lambda_ * masks[i] * self.pseudo_gae[i + 1]
            self.pseudo_returns[i] = self.pseudo_gae[i] + self.pseudo_values[i]


class RndRolloutBatch(NamedTuple):
    states: Tensor
    actions: Tensor
    masks: Tensor
    returns: Tensor
    values: Tensor
    old_log_probs: Tensor
    pseudo_values: Tensor
    pseudo_returns: Tensor
    advantages: Tensor
    rnn_init: RnnState


class RndRolloutSampler(RolloutSampler):
    def __init__(
            self,
            storage: RndRolloutStorage,
            penv: ParallelEnv,
            minibatch_size: int,
            ext_coeff: float,
            int_coeff: float,
            rnn: RnnBlock = DummyRnn(),
            adv_normalize_eps: Optional[float] = None
    ) -> None:
        super().__init__(storage, penv, minibatch_size, rnn, None)
        self.pseudo_returns = storage.pseudo_returns.flatten()
        self.pseudo_values = storage.batch_pseudo_values()
        pseudo_advs = storage.pseudo_gae[:-1].flatten().to(self.advantages.device)
        self.advantages.mul_(ext_coeff).add_(pseudo_advs * int_coeff)
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
            self.pseudo_values[i],
            self.pseudo_returns[i],
            self.advantages[i],
            self.rnn_init[i[:len(i) // self.nsteps]]
        )
