"""Variational Auto Encoders as intrinsic rewards
"""

from abc import ABC, abstractmethod
import copy
import enum
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.distributions import Normal
from torch.nn import functional as F
from torch import Tensor, nn

from rainy import Config, net
from rainy.utils import Device
from rainy.utils.rms import RunningMeanStdTorch

from .prelude import Normalizer, PreProcessor
from .unsupervised import (
    UnsupervisedBlock,
    UnsupervisedIRewGen,
    normalize_r_default,
    preprocess_default,
)


class DecoderKind(enum.Enum):
    BERNOULLI = 1
    GAUSSIAN = 2
    CATEGORICAL = 3

    def wrap(self, net: nn.Module, init: Optional[net.Initializer] = None) -> nn.Module:
        if self == DecoderKind.BERNOULLI:
            return BernoulliHead(net, init)
        elif self == DecoderKind.GAUSSIAN:
            return GaussianHead(net, init)
        elif self == DecoderKind.CATEGORICAL:
            return CategoricalHead(net, init)
        else:
            raise NotImplementedError()


DECORDERS = {
    "bernoulli": DecoderKind.BERNOULLI,
    "gaussian": DecoderKind.GAUSSIAN,
    "categorical": DecoderKind.CATEGORICAL,
}


class DecoderDist(ABC):
    @abstractmethod
    def loss(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def sample(self) -> Tensor:
        pass


class BernoulliDist(DecoderDist):
    def __init__(self, logits: Tensor) -> None:
        self.logits = logits

    def loss(self, x: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(self.logits, x, reduction="none")

    def sample(self) -> Tensor:
        return torch.distributions.Categorical(logits=self.logits).sample()


class CategoricalDist(DecoderDist):
    def __init__(self, logits: Tensor) -> None:
        self.logits = logits

    def loss(self, x: Tensor) -> Tensor:
        t = x.argmax(dim=1)
        return F.cross_entropy(self.logits, t, reduction="none")

    def sample(self) -> Tensor:
        if self.logits.dim() > 1:
            shape = self.logits.shape
            logits = self.logits.view(shape[0], shape[1], -1).transpose(1, 2)
            sample = torch.distributions.Categorical(logits=logits).sample()
            return sample.view(shape[0], *shape[2:])
        else:
            return torch.distributions.Categorical(logits=logits).sample()


class GaussianDist(DecoderDist):
    """
    Thankfully referenced chainer implementation for nll loss:
    https://github.com/chainer/chainer/blob/v7.1.0/chainer/functions/loss/vae.py#L123
    """

    def __init__(self, mu: Tensor, logvar: Tensor) -> None:
        self.mu = mu
        self.logvar = logvar

    def loss(self, x: Tensor) -> Tensor:
        x_prec = torch.exp(-self.logvar)
        x_diff = x - self.mu
        x_power = x_diff.pow(2).mul_(x_prec).mul_(-0.5)
        return 0.5 * (self.logvar + np.log(2.0 * np.pi)) - x_power

    def sample(self) -> Tensor:
        return Normal(self.mu, torch.exp(0.5 * self.logvar)).sample()


class BernoulliHead(nn.Module):
    def __init__(self, net: nn.Module, init: net.Initializer) -> None:
        super().__init__()
        self.net = init(net)

    def forward(self, x: Tensor) -> BernoulliDist:
        return BernoulliDist(logits=self.net(x))


class CategoricalHead(BernoulliHead):
    def forward(self, x: Tensor) -> CategoricalDist:
        return CategoricalDist(logits=self.net(x))


class GaussianHead(nn.Module):
    def __init__(self, net: nn.Module, init: net.Initializer) -> None:
        super().__init__()
        self.mu = init(net)
        self.logvar = init(copy.deepcopy(net))

    def forward(self, x: Tensor) -> GaussianDist:
        mu = self.mu(x)
        logvar = self.logvar(x)
        return GaussianDist(mu, logvar)


class VaeOutPut(NamedTuple):
    decoder: DecoderDist
    mu: Tensor
    logvar: Tensor


class Vae(nn.Module, ABC):
    input_dim: Sequence[int]
    encoder: nn.Module
    decoder: nn.Module
    mu: nn.Module
    logvar: nn.Module

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x: Tensor) -> VaeOutPut:
        x = self.encoder(x)
        mu, logvar = self.mu(x), self.logvar(x)
        z = self.reparameterize(mu, logvar)
        return VaeOutPut(self.decoder(z), mu, logvar)


class FcVae(Vae):
    def __init__(
        self,
        input_dim: tuple,
        fc_dims: Sequence[int] = [64, 64],
        z_dim: int = 32,
        device: Device = Device(),
        decoder_kind: DecoderKind = DecoderKind.GAUSSIAN,
        init: net.Initializer = net.Initializer(),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        x_dim = np.prod(input_dim)
        dims = [x_dim] + list(fc_dims)
        encoders = []
        for i in range(len(fc_dims)):
            encoders.append(init(nn.Linear(dims[i], dims[i + 1])))
            encoders.append(nn.ReLU(inplace=True))

        decoders = [init(nn.Linear(z_dim, dims[-1])), nn.ReLU(inplace=True)]
        for i in reversed(range(1, len(fc_dims))):
            decoders.append(init(nn.Linear(dims[i + 1], dims[i])))
        decoders.append(decoder_kind.wrap(nn.Linear(dims[1], x_dim), init))
        self.encoder = nn.Sequential(*encoders)
        self.decoder = nn.Sequential(*decoders)
        self.mu = nn.Linear(dims[-1], z_dim)
        self.logvar = nn.Linear(dims[-1], z_dim)


class Flatten(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.view(x.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, shape: tuple) -> Tensor:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.view(x.size(0), *self.shape)


class CNNVae(Vae):
    def __init__(
        self,
        input_dim: tuple,
        fc_dim: int = 256,
        z_dim: int = 64,
        conv_channels: List[int] = [32, 64, 64],
        encoder_args: List[tuple] = [(8, 4), (4, 2), (3, 1)],
        decoder_args: List[tuple] = [(3, 1), (4, 2), (8, 4)],
        device: Device = Device(),
        decoder_kind: DecoderKind = DecoderKind.GAUSSIAN,
        cnn_init: net.Initializer = net.Initializer(
            weight_init=net.init.orthogonal(nonlinearity="relu"),
        ),
        init: net.Initializer = net.Initializer(),
    ) -> None:
        super().__init__()
        assert len(input_dim) == 3, "CNNVae assumes that len(input_dim) == 3"
        self.input_dim = input_dim
        in_channel, height, width = input_dim
        channels = [in_channel] + conv_channels
        conved_h, conved_w = net.calc_cnn_hidden(encoder_args, height, width)
        hidden = conved_h * conved_w * channels[-1]

        def _make_encoder() -> nn.Sequential:
            encoders = []
            for i in range(len(conv_channels)):
                encoders.append(
                    cnn_init(nn.Conv2d(channels[i], channels[i + 1], *encoder_args[i]))
                )
                encoders.append(nn.ReLU(inplace=True))
            return nn.Sequential(
                *encoders,
                Flatten(),
                init(nn.Linear(hidden, fc_dim)),
                nn.ReLU(inplace=True),
            )

        self.encoder = _make_encoder()
        self.mu = nn.Linear(fc_dim, z_dim)
        self.logvar = nn.Linear(fc_dim, z_dim)
        channels.reverse()

        def _make_decoder() -> nn.Sequential:
            decoders = [
                init(nn.Linear(z_dim, fc_dim)),
                nn.ReLU(inplace=True),
                init(nn.Linear(fc_dim, hidden)),
                Unflatten((channels[0], conved_w, conved_h)),
                nn.ReLU(inplace=True),
            ]

            for i in range(len(conv_channels) - 1):
                params = channels[i], channels[i + 1], *decoder_args[i]
                decoders.append(cnn_init(nn.ConvTranspose2d(*params)))
                decoders.append(nn.ReLU(inplace=True))
            decoders.append(
                decoder_kind.wrap(
                    nn.ConvTranspose2d(channels[-2], channels[-1], *decoder_args[-1]),
                    cnn_init,
                )
            )
            return nn.Sequential(*decoders)

        self.decoder = _make_decoder()


class VaeUnsupervisedBlock(UnsupervisedBlock):
    def __init__(self, vae: Vae) -> None:
        super().__init__()
        self.vae = vae

    def rewards(self, states: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        batch_size = states.size(0)
        decoder, *_ = self.vae(states)
        recons_loss = decoder.loss(states).div_(batch_size)
        return recons_loss, None

    def loss(self, states: Tensor, target: Optional[Tensor]) -> Tensor:
        decoder, mu, logvar = self.vae(states)
        recons_loss = decoder.loss(states).sum(dim=1)
        kl_loss = -0.5 * (1.0 + logvar - mu.pow(2.0) - logvar.exp()).sum(dim=1)
        return recons_loss + kl_loss

    @property
    def input_dim(self) -> Sequence[int]:
        return self.vae.input_dim


def normalize_vae(t: Tensor, rms: RunningMeanStdTorch) -> Tensor:
    t = t.reshape(-1, 1, *t.shape[-2:])
    t.sub_(rms.mean.float()).div_(rms.std().float())
    return t.clamp_(-5.0, 5.0).add_(5.0).div(10.0)


def irew_gen_cnn_vae(
    preprocess: PreProcessor = preprocess_default,
    state_normalizer: Normalizer = normalize_vae,
    reward_normalizer: Normalizer = normalize_r_default,
    **kwargs,
) -> Callable[[Config, Device], UnsupervisedIRewGen]:
    def _make_irew_gen(cfg: Config, device: Device) -> UnsupervisedIRewGen:
        input_dim = 1, *cfg.state_dim[1:]
        vae = CNNVae(input_dim, **kwargs)
        return UnsupervisedIRewGen(
            VaeUnsupervisedBlock(vae),
            cfg.int_discount_factor,
            cfg.nworkers,
            device,
            preprocess=preprocess,
            state_normalizer=state_normalizer,
            reward_normalizer=reward_normalizer,
        )

    return _make_irew_gen


def irew_gen_fc_vae(
    preprocess: PreProcessor = lambda x, _: x,
    state_normalizer: Normalizer = lambda x, _: x,
    reward_normalizer: Normalizer = normalize_r_default,
    **kwargs,
) -> Callable[[Config, Device], UnsupervisedIRewGen]:
    def _make_irew_gen(cfg: Config, device: Device) -> UnsupervisedIRewGen:
        vae = FcVae(cfg.state_dim, **kwargs)
        return UnsupervisedIRewGen(
            VaeUnsupervisedBlock(vae),
            cfg.int_discount_factor,
            cfg.nworkers,
            device,
            preprocess=preprocess,
            state_normalizer=state_normalizer,
            reward_normalizer=reward_normalizer,
            ob_rms_shape=cfg.state_dim,
        )

    return _make_irew_gen
