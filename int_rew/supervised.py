from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor
from typing import Optional, Sequence, Tuple
from rainy.utils import Device, RunningMeanStdTorch
from rainy.utils.state_dict import HasStateDict
from .prelude import Normalizer, PreProcessor
from .unsupervised import RewardForwardFilter


class SupervisedBlock(nn.Module, ABC):
    @abstractmethod
    def rewards(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss(self, states: Tensor, actions: Tensor, next_states: Tensor) -> Tensor:
        pass

    @property
    @abstractmethod
    def input_dim(self) -> Sequence[int]:
        pass


class SupervisedIRewGen(HasStateDict):
    def __init__(
        self,
        intrew_block: SupervisedBlock,
        gamma: float,
        nworkers: int,
        device: Device,
        preprocess_state: PreProcessor,
        preprocess_action: PreProcessor,
        state_normalizer: Normalizer,
        reward_normalizer: Normalizer,
        ob_rms_shape: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.block = intrew_block
        self.block.to(device.unwrapped)
        self.device = device
        if ob_rms_shape is None:
            self.ob_rms = RunningMeanStdTorch(self.block.input_dim[1:], device)
        else:
            self.ob_rms = RunningMeanStdTorch(tuple(ob_rms_shape), device)
        self.rff = RewardForwardFilter(gamma, nworkers, device)
        self.rff_rms = RunningMeanStdTorch(shape=(), device=device)
        self.nworkers = nworkers
        self.cached_target = device.ones(0)
        self._preprocess_state = preprocess_state
        self._preprocess_action = preprocess_action
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

    def preprocess_state(self, t: Tensor) -> Tensor:
        return self._preprocess_state(t, self.device)

    def preprocess_action(self, t: Tensor) -> Tensor:
        return self._preprocess_action(t, self.device)

    def gen_rewards(
        self, state: Tensor, action: Tensor, next_state: Tensor
    ) -> Tuple[Tensor, dict]:
        s = self.preprocess_state(state)
        a = self.preprocess_action(action)
        ns = self.preprocess_state(next_state)
        self.ob_rms.update(s.double().view(-1, *self.ob_rms.mean.shape))
        with torch.no_grad():
            normalized_s = self.state_normalizer(s, self.ob_rms)
            normalized_ns = self.state_normalizer(ns, self.ob_rms)
            error = self.block.rewards(normalized_ns, a,  normalized_s)  # TODO: shape
        nsteps = s.size(0) // self.nworkers
        rewards = error.view(nsteps, self.nworkers, -1).mean(-1)
        rffs_int = torch.cat([self.rff.update(rewards[i]) for i in range(nsteps)])
        self.rff_rms.update(rffs_int.view(-1))
        normalized_rewards = self.reward_normalizer(rewards, self.rff_rms)
        stats = dict(
            intrew_raw_mean=rewards.mean().item(),
            intrew_mean=normalized_rewards.mean().item(),
            rffs_mean=rffs_int.mean().item(),
            rffs_rms_mean=self.rff_rms.mean.mean().item(),
            rffs_rms_std=self.rff_rms.std().mean().item(),
        )
        return normalized_rewards, stats

    def aux_loss(
        self, state: Tensor, target: Optional[Tensor], use_ratio: float
    ) -> Tensor:
        mask = torch.empty(state.size(0)).uniform_() < use_ratio
        s = self.preprocess(state[mask])
        normalized_s = self.state_normalizer(s, self.ob_rms)
        return self.block.loss(normalized_s, None if target is None else target[mask])
