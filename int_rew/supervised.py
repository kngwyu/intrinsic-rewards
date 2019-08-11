from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor
from typing import Callable, Optional, Sequence
from rainy.utils import Device
from rainy.utils.log import ExpStats
from rainy.utils.rms import RunningMeanStdTorch

from .rff import RewardForwardFilter
from .unsupervised import UnsupervisedIRewGen


def preprocess_default(t: Tensor, device: Device) -> Tensor:
    """Extract one channel and rescale to 0..255
    """
    return t.to(device.unwrapped)[:, -1].mul_(255.0)


def normalize_s_default(t: Tensor, rms: RunningMeanStdTorch) -> Tensor:
    t = t.reshape(-1, 1, *t.shape[-2:])
    t.sub_(rms.mean.float()).div_(rms.std().float())
    return torch.clamp(t, -5.0, 5.0)


def normalize_r_default(t: Tensor, rms: RunningMeanStdTorch) -> Tensor:
    return t.div(rms.std())


class SupervisedBlock(nn.Module, ABC):
    @abstractmethod
    def rewards(self, state: Tensor, target: Tensor, action: Optional[Tensor] = None) -> Tensor:
        pass

    @abstractmethod
    def loss(self, state: Tensor, target: Tensor, action: Optional[Tensor] = None) -> Tensor:
        pass

    @property
    @abstractmethod
    def input_dim(self) -> Sequence[int]:
        pass


class SupervisedIRewGen(UnsupervisedIRewGen):
    def __init__(
            self,
            intrew_block: SupervisedBlock,
            gamma: float,
            nworkers: int,
            device: Device,
            preprocess: Callable[[Tensor, Device], Tensor],
            state_normalizer: Callable[[Tensor, RunningMeanStdTorch], Tensor],
            reward_normalizer: Callable[[Tensor, RunningMeanStdTorch], Tensor]
    ) -> None:
        super().__init__()
        self.block = intrew_block
        self.block.to(device.unwrapped)
        self.device = device
        self.ob_rms = RunningMeanStdTorch(self.block.input_dim[1:], device)
        self.rff = RewardForwardFilter(gamma, nworkers, device)
        self.rff_rms = RunningMeanStdTorch(shape=(), device=device)
        self.nworkers = nworkers
        self.cached_target = device.ones(0)
        self._preprocess = preprocess
        self.state_normalizer = state_normalizer
        self.reward_normalizer = reward_normalizer
        self.normalize_reward = True

    def gen_rewards(
            self,
            state: Tensor,
            target: Tensor,
            action: Optional[Tensor],
            reporter: Optional[ExpStats] = None
    ) -> Tensor:
        s = self.preprocess(state)
        self.ob_rms.update(s.double().view(-1, *self.ob_rms.mean.shape))
        with torch.no_grad():
            normalized_s = self.state_normalizer(s, self.ob_rms)
            error, self.cached_target = self.block.rewards(normalized_s)
        nsteps = s.size(0) // self.nworkers
        rewards = error.view(nsteps, self.nworkers, -1).mean(-1)
        report_dict = dict(intrew_raw_mean=rewards)
        if self.normalize_reward:
            rewards = self._normalize_reward(rewards, nsteps, report_dict)
        if reporter is not None:
            reporter.update(report_dict)
        return rewards

    def aux_loss(
            self,
            state: Tensor,
            target: Tensor,
            action: Optional[Tensor],
            use_ratio: float
    ) -> Tensor:
        mask = torch.empty(state.size(0)).uniform_() < use_ratio
        s = self.preprocess(state[mask])
        normalized_s = self.state_normalizer(s, self.ob_rms)
        return self.block.loss(normalized_s, None if target is None else target[mask])
