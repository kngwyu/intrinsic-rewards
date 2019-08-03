from torch import nn, Tensor
from typing import Callable, List, Optional, Sequence, Tuple
from rainy.net import Initializer, make_cnns, NetworkBlock
from rainy.net.prelude import Params
from rainy.utils import Device
from rainy.utils.rms import RunningMeanStdTorch

from ..unsupervised import UnsupervisedBlock, UnsupervisedIRewGen
from ..unsupervised import preprocess_default, normalize_r_default, normalize_s_default


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


class RndUnsupervisedBlock(UnsupervisedBlock):
    def __init__(self, predictor: NetworkBlock, target: NetworkBlock) -> None:
        super().__init__()
        self.target = target
        self.predictor = predictor

    def rewards(self, states: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        t = self.target(states)
        p = self.predictor(states)
        return (t - p).pow(2), t

    def loss(self, states: Tensor, target: Optional[Tensor]) -> Tensor:
        return (target - self.predictor(states)).pow(2).mean()

    @property
    def input_dim(self) -> Sequence[int]:
        return self.target.input_dim

    def parameters(self) -> Params:
        return self.predictor.parameters()


def irew_gen_default(
        params: Sequence[tuple] = ((8, 4), (4, 2), (3, 1)),
        channels: Sequence[int] = (32, 64, 64),
        output_dim: int = 512,
        preprocess: Callable[[Tensor, Device], Tensor] = preprocess_default,
        state_normalizer: Callable[[Tensor, RunningMeanStdTorch], Tensor] = normalize_s_default,
        reward_normalizer: Callable[[Tensor, RunningMeanStdTorch], Tensor] = normalize_r_default,
) -> Callable[['RndConfig', Device], UnsupervisedIRewGen]:
    def _make_irew_gen(cfg: 'RndConfig', device: Device) -> UnsupervisedIRewGen:
        input_dim = 1, *cfg.state_dim[1:]
        cnns, hidden = make_cnns(input_dim, params, channels)
        target = RndConvBody(cnns, [nn.Linear(hidden, output_dim)], input_dim)
        predictor_fc = [
            nn.Linear(hidden, output_dim),
            nn.Linear(output_dim, output_dim),
            nn.Linear(output_dim, output_dim)
        ]
        cnns, _ = make_cnns(input_dim, params, channels)
        predictor = RndConvBody(cnns, predictor_fc, input_dim)
        return UnsupervisedIRewGen(
            RndUnsupervisedBlock(target, predictor),
            cfg.int_discount_factor,
            cfg.nworkers,
            device,
            preprocess=preprocess,
            state_normalizer=state_normalizer,
            reward_normalizer=reward_normalizer,
        )
    return _make_irew_gen
