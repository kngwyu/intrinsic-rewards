from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor
from typing import Callable, Optional, Sequence, Tuple
from rainy.utils import Device
from rainy.utils.log import ExpStats
from rainy.utils.rms import RunningMeanStdTorch
from rainy.utils.state_dict import HasStateDict, TensorStateDict


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


class RewardForwardFilter(TensorStateDict):
    def __init__(self, gamma: float, nworkers: int, device: Device) -> None:
        self.gamma = gamma
        self.nonepisodic_return = device.zeros(nworkers)

    def update(self, prew: Tensor) -> Tensor:
        self.nonepisodic_return.mul_(self.gamma).add_(prew)
        return self.nonepisodic_return


class UnsupervisedBlock(nn.Module, ABC):
    @abstractmethod
    def rewards(self, states: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        pass

    @abstractmethod
    def loss(self, states: Tensor, target: Optional[Tensor]) -> Tensor:
        pass

    @property
    @abstractmethod
    def input_dim(self) -> Sequence[int]:
        pass


class UnsupervisedIRewGen(HasStateDict):
    def __init__(
        self,
        intrew_block: UnsupervisedBlock,
        gamma: float,
        nworkers: int,
        device: Device,
        preprocess: Callable[[Tensor, Device], Tensor],
        state_normalizer: Callable[[Tensor, RunningMeanStdTorch], Tensor],
        reward_normalizer: Callable[[Tensor, RunningMeanStdTorch], Tensor],
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

    def state_dict(self) -> dict:
        return {
            "block": self.block.state_dict(),
            "ob_rms": self.ob_rms.state_dict(),
            "rff": self.rff.state_dict(),
            "rff_rms": self.rff_rms.state_dict(),
        }

    def load_state_dict(self, d: dict) -> None:
        for key in d.keys():
            obj = getattr(self, key)
            obj.load_state_dict(d[key])

    def preprocess(self, t: Tensor) -> Tensor:
        return self._preprocess(t, self.device)

    def gen_rewards(self, state: Tensor, reporter: Optional[ExpStats] = None) -> Tensor:
        s = self.preprocess(state)
        self.ob_rms.update(s.double().view(-1, *self.ob_rms.mean.shape))
        with torch.no_grad():
            normalized_s = self.state_normalizer(s, self.ob_rms)
            error, self.cached_target = self.block.rewards(normalized_s)
        nsteps = s.size(0) // self.nworkers
        rewards = error.view(nsteps, self.nworkers, -1).mean(-1)
        rffs_int = torch.cat([self.rff.update(rewards[i]) for i in range(nsteps)])
        self.rff_rms.update(rffs_int.view(-1))
        normalized_rewards = self.reward_normalizer(rewards, self.rff_rms)
        if reporter is not None:
            reporter.update(
                {
                    "intrew_raw_mean": rewards.mean().item(),
                    "intrew_mean": normalized_rewards.mean().item(),
                    "rffs_mean": rffs_int.mean().item(),
                    "rffs_rms_mean": self.rff_rms.mean.mean().item(),
                    "rffs_rms_std": self.rff_rms.std().mean().item(),
                }
            )
        return normalized_rewards

    def aux_loss(
        self, state: Tensor, target: Optional[Tensor], use_ratio: float
    ) -> Tensor:
        mask = torch.empty(state.size(0)).uniform_() < use_ratio
        s = self.preprocess(state[mask])
        normalized_s = self.state_normalizer(s, self.ob_rms)
        return self.block.loss(normalized_s, None if target is None else target[mask])
