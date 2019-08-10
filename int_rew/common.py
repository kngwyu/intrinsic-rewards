from torch import Tensor
from rainy.utils import Device
from rainy.utils.state_dict import TensorStateDict


class RewardForwardFilter(TensorStateDict):
    def __init__(self, gamma: float, nworkers: int, device: Device) -> None:
        self.gamma = gamma
        self.nonepisodic_return = device.zeros(nworkers)

    def update(self, prew: Tensor) -> Tensor:
        self.nonepisodic_return.mul_(self.gamma).add_(prew)
        return self.nonepisodic_return
