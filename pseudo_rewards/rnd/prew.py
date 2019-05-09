import copy
import torch
from torch import nn, Tensor
from typing import Callable, List, Sequence, Tuple
from rainy.net import Activator, Initializer, make_cnns, NetworkBlock
from rainy.net.prelude import Params
from rainy.prelude import Array
from rainy.utils import Device


class PseudoRewardGenerator:
    def __init__(self, target: NetworkBlock, predictor: NetworkBlock, device: Device) -> None:
        device.to(target)
        device.to(predictor)
        self.target = target
        self.predictor = predictor
        self.device = device

    def pseudo_reward(self, state: Array[float]) -> Tensor:
        s = self.device.tensor(state)
        with torch.no_grad():
            target = self.target(s)
            prediction = self.predictor(s)
        return (target - prediction).pow(2).sum(dim=-1)

    def aux_loss(self, state: Tensor) -> Tensor:
        s = self.device.tensor(state)
        with torch.no_grad():
            target = self.target(s)
        prediction = self.predictor(s)
        return (target - prediction).pow(2).mean()

    def params(self) -> Params:
        return self.predictor.parameters()


class RndConvBody(NetworkBlock):
    def __init__(
            self,
            cnns: List[nn.Module],
            fcs: List[nn.Module],
            input_dim: Tuple[int, int, int],
            activ1: nn.Module = nn.LeakyReLU(negative_slope=0.2, inplace=True),
            activ2: Activator = nn.ReLU(inplace=True),
            init: Initializer = Initializer(nonlinearity='relu'),
    ) -> None:
        super().__init__()
        self.conv = init.make_list(cnns)
        self.fc = init.make_list(fcs)
        self._input_dim = input_dim
        self.init = init
        self.activ1 = activ1
        self.activ2 = activ2

    @property
    def input_dim(self) -> Tuple[int, ...]:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self.fc.out_features

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.cnns:
            x = self.activ1(conv(x), inplace=True)
        x = x.view(x.size(0), -1)
        for fc in self.fcs[:-1]:
            x = self.activ2(fc(x), inplace=True)
        return self.fcs[-1](x)


def prew_gen_deafult(
        params: Sequence[tuple] = ((8, 4), (4, 2), (3, 1)),
        channels: Sequence[int] = (32, 64, 64),
        output_dim: int = 512,
) -> Callable[[Tuple[int, int, int], Device], PseudoRewardGenerator]:
    def _make_prew_gen(input_dim: Tuple[int, ...], device: Device) -> PseudoRewardGenerator:
        cnns, hidden = make_cnns(input_dim, params, channels)
        target = RndConvBody(cnns, [nn.Linear(hidden, output_dim)], input_dim)
        predictor_fc = [
            nn.Linear(hidden, output_dim),
            nn.Linear(output_dim, output_dim),
            nn.Linear(output_dim, output_dim)
        ]
        predictor = RndConvBody(copy.deepcopy(cnns), predictor_fc, input_dim)
        return PseudoRewardGenerator(target, predictor, device)
    return _make_prew_gen
