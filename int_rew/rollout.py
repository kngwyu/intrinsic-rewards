from typing import List, Optional

import torch
from torch import Tensor

from rainy.lib.rollout import RolloutStorage
from rainy.prelude import State
from rainy.utils import Device


class IntValueRolloutStorage(RolloutStorage[State]):
    def __init__(
        self,
        nsteps: int,
        nworkers: int,
        device: Device,
        gamma: float,
        another_device: Device = Device(use_cpu=True),
    ) -> None:
        super().__init__(nsteps, nworkers, device)
        self.int_rewards: List[Tensor] = []
        self.int_values: List[Tensor] = []
        self.int_gae = another_device.zeros((nsteps + 1, nworkers))
        self.int_returns = another_device.zeros((nsteps, nworkers))
        self.another_device = another_device

    def push(self, *args, pvalue: Optional[Tensor] = None, **kwargs) -> None:
        super().push(*args, **kwargs)
        if pvalue is not None:
            self.int_values.append(self.another_device.tensor(pvalue))

    def reset(self) -> None:
        super().reset()
        self.int_rewards.clear()
        self.int_values.clear()

    def batch_int_values(self) -> Tensor:
        return torch.cat(self.values[: self.nsteps])

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
        self.int_values.append(self.another_device.tensor(next_value))
        masks = self.masks if use_mask else self.another_device.ones(self.nsteps + 1)
        self.int_gae.fill_(0.0)
        for i in reversed(range(self.nsteps)):
            td_error = (
                rewards[i]
                + gamma * self.int_values[i + 1] * masks[i + 1]
                - self.int_values[i]
            )
            self.int_gae[i] = (
                td_error + gamma * lambda_ * masks[i] * self.int_gae[i + 1]
            )
            self.int_returns[i] = self.int_gae[i] + self.int_values[i]
