from torch import nn, Tensor
from typing import Callable, List, Optional, Sequence, Type, Tuple
from rainy import Config
from rainy.net import FCBody, LinearHead, make_cnns, NetworkBlock
from rainy.net.init import Initializer, orthogonal
from rainy.prelude import Params
from rainy.utils import Device

from ..prelude import Normalizer, PreProcessor
from ..unsupervised import UnsupervisedBlock, UnsupervisedIRewGen
from ..unsupervised import preprocess_default, normalize_r_default, normalize_s_default


class RNDConvBody(NetworkBlock):
    def __init__(
        self,
        cnns: List[nn.Module],
        fcs: List[nn.Module],
        input_dim: Tuple[int, int, int],
        activ1: nn.Module = nn.LeakyReLU(negative_slope=0.2, inplace=True),
        activ2: nn.Module = nn.ReLU(inplace=True),
        init: Initializer = Initializer(weight_init=orthogonal(nonlinearity="relu")),
    ) -> None:
        super().__init__()
        self.cnns = init.make_list(cnns)
        self.fcs = init.make_list(fcs)
        self.input_dim = input_dim
        self.init = init
        self.activ1 = activ1
        self.activ2 = activ2

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


class RNDFCBody(NetworkBlock):
    def __init__(
        self, input_dim: int, out_dim: int, units: List[int] = [64, 64],
    ):
        super().__init__()
        self.body = FCBody(input_dim, units=units)
        self.head = LinearHead(units[-1], out_dim)
        self.input_dim = input_dim
        self.output_dim = out_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.body(x))


class RNDUnsupervisedBlock(UnsupervisedBlock):
    def __init__(self, predictor: NetworkBlock, target: NetworkBlock) -> None:
        super().__init__()
        self.target = target
        self.predictor = predictor

    def rewards(self, states: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        t = self.target(states)
        p = self.predictor(states)
        return (t - p).pow(2), t

    def loss(self, states: Tensor, target: Optional[Tensor]) -> Tensor:
        return (target - self.predictor(states)).pow(2)

    @property
    def input_dim(self) -> Sequence[int]:
        return self.target.input_dim

    def parameters(self) -> Params:
        return self.predictor.parameters()


def irew_gen_default(
    cnn_params: Sequence[tuple] = ((8, 4), (4, 2), (3, 1)),
    hidden_channels: Sequence[int] = (32, 64, 64),
    feature_dim: int = 512,
    preprocess: PreProcessor = preprocess_default,
    state_normalizer: Normalizer = normalize_s_default,
    reward_normalizer: Normalizer = normalize_r_default,
    cls: Type[UnsupervisedIRewGen] = UnsupervisedIRewGen,
    **kwargs,
) -> Callable[[Config, Device], UnsupervisedIRewGen]:
    def _make_irew_gen(cfg: Config, device: Device) -> UnsupervisedIRewGen:
        input_dim = 1, *cfg.state_dim[1:]
        cnns, hidden = make_cnns(input_dim, cnn_params, hidden_channels)
        target = RNDConvBody(cnns, [nn.Linear(hidden, feature_dim)], input_dim)
        predictor_fc = [
            nn.Linear(hidden, feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.Linear(feature_dim, feature_dim),
        ]
        cnns, _ = make_cnns(input_dim, cnn_params, hidden_channels)
        predictor = RNDConvBody(cnns, predictor_fc, input_dim)
        return cls(
            RNDUnsupervisedBlock(target, predictor),
            cfg.int_discount_factor,
            cfg.nworkers,
            device,
            preprocess=preprocess,
            state_normalizer=state_normalizer,
            reward_normalizer=reward_normalizer,
            **kwargs,
        )

    return _make_irew_gen


def irew_gen_fc(
    hidden_units: List[int] = [64, 64],
    output_dim: int = 64,
    preprocess: PreProcessor = lambda x, _: x,
    state_normalizer: Normalizer = lambda x, _: x,
    reward_normalizer: Normalizer = normalize_r_default,
    cls: Type[UnsupervisedIRewGen] = UnsupervisedIRewGen,
    **kwargs,
) -> Callable[[Config, Device], UnsupervisedIRewGen]:
    def _make_irew_gen(cfg: Config, device: Device) -> UnsupervisedIRewGen:
        input_dim = cfg.state_dim[0]

        def net_gen():
            return RNDFCBody(input_dim, output_dim, hidden_units)

        return cls(
            RNDUnsupervisedBlock(net_gen(), net_gen()),
            cfg.int_discount_factor,
            cfg.nworkers,
            device,
            preprocess=preprocess,
            state_normalizer=state_normalizer,
            reward_normalizer=reward_normalizer,
            ob_rms_shape=(input_dim,),
            **kwargs,
        )

    return _make_irew_gen
