import torch
from torch import nn, Tensor
from typing import Callable, List, Sequence, Tuple
from rainy.net import Initializer, make_cnns, NetworkBlock
from rainy.net.prelude import Params
from rainy.prelude import Array
from rainy.utils import Device
from rainy.utils.rms import RunningMeanStdTorch


class IntRewardGenerator:
    def __init__(
            self,
            target: NetworkBlock,
            predictor: NetworkBlock,
            device: Device,
    ) -> None:
        target.to(device.unwrapped)
        predictor.to(device.unwrapped)
        self.target = target
        self.predictor = predictor
        self.device = device
        self.ob_rms = RunningMeanStdTorch(target.input_dim, device)

    def __call__(self, state: Array[float]) -> Tensor:
        s = self.device.tensor(state).mul_(255.0)
        self.ob_rms.update(s.double())
        with torch.no_grad():
            normalized_s = torch.clamp((s - self.ob_rms.mean.float())
                                       / self.ob_rms.std().float(), -5.0, 5.0)
            target = self.target(normalized_s)
            prediction = self.predictor(normalized_s)
        return (target - prediction).pow(2).sum(dim=-1)

    def aux_loss(self, state: Tensor) -> Tensor:
        s = self.device.tensor(state).mul_(255.0)
        normalized_s = torch.clamp((s - self.ob_rms.mean.float())
                                   / self.ob_rms.std().float(), -5.0, 5.0)
        with torch.no_grad():
            target = self.target(normalized_s)
        prediction = self.predictor(normalized_s)
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
            activ2: nn.Module = nn.ReLU(inplace=True),
            init: Initializer = Initializer(nonlinearity='relu'),
    ) -> None:
        super().__init__()
        self.cnns = init.make_list(cnns)
        self.fcs = init.make_list(fcs)
        self._input_dim = input_dim
        self.init = init
        self.activ1 = activ1
        self.activ2 = activ2

    @property
    def input_dim(self) -> Tuple[int, ...]:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self.fcs[-1].out_features

    def forward(self, x: Tensor) -> Tensor:
        for cnn in self.cnns:
            x = self.activ1(cnn(x))
        x = x.view(x.size(0), -1)
        for fc in self.fcs[:-1]:
            x = self.activ2(fc(x))
        return self.fcs[-1](x)


def irew_gen_deafult(
        params: Sequence[tuple] = ((8, 4), (4, 2), (3, 1)),
        channels: Sequence[int] = (32, 64, 64),
        output_dim: int = 512,
) -> Callable[[Tuple[int, int, int], Device], IntRewardGenerator]:
    def _make_irew_gen(input_dim: Tuple[int, ...], device: Device) -> IntRewardGenerator:
        cnns, hidden = make_cnns(input_dim, params, channels)
        target = RndConvBody(cnns, [nn.Linear(hidden, output_dim)], input_dim)
        predictor_fc = [
            nn.Linear(hidden, output_dim),
            nn.Linear(output_dim, output_dim),
            nn.Linear(output_dim, output_dim)
        ]
        cnns, _ = make_cnns(input_dim, params, channels)
        predictor = RndConvBody(cnns, predictor_fc, input_dim)
        return IntRewardGenerator(target, predictor, device)
    return _make_irew_gen
