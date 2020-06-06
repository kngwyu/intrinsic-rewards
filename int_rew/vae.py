"""Variational Auto Encoder with CNN
Based on https://github.com/kngwyu/pytorch-autoencoders/
"""
from abc import ABC, abstractmethod
from itertools import chain
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Size, Tensor, nn

from rainy import Config
from rainy.net import Initializer, calc_cnn_hidden
from rainy.net.init import orthogonal
from rainy.utils import Device
from rainy.utils.rms import RunningMeanStdTorch

from .prelude import Normalizer, PreProcessor
from .unsupervised import (UnsupervisedBlock, UnsupervisedIRewGen,
                           normalize_r_default, preprocess_default)
from .utils import construct_body, sequential_body

flatten = chain.from_iterable


class VaeOutPut(NamedTuple):
    x: Tensor
    mu: Tensor
    logvar: Tensor


class Vae(ABC):
    input_dim: Sequence[int]

    @abstractmethod
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def decode(self, z: Tensor, old_shape: Size = None) -> Tensor:
        pass

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x: Tensor) -> VaeOutPut:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return VaeOutPut(self.decode(z, old_shape=x.shape), mu, logvar)


class FcVae(nn.Module, Vae):
     def __init__(
        self,
        input_dim: tuple,
        fc_dim: int = 400,
        z_dim: int = 20,
        noptions: int = 4,
        device: Device = Device(),
        decoder_kind: DecoderKind = DecoderKind.BERNOULLI,
        init: net.Initializer = net.Initializer(),
    ) -> None:
        super().__init__()
        self.device = device
        x_dim = np.prod(input_dim)
        self.noptions = noptions
        self.encoder = nn.Sequential(
            init(nn.Linear(x_dim, fc_dim)),
            nn.ReLU(inplace=True),
            init(nn.Linear(fc_dim, z_dim)),
        )
        self.decoder = nn.Sequential(
            init(nn.Linear(z_dim, fc_dim)),
            nn.ReLU(inplace=True),
            decoder_kind.wrap(nn.Linear(fc_dim, x_dim), init),
        )
        self.to(device.unwrapped)

class ConvVae(nn.Module, Vae):
    def __init__(
        self,
        input_dim: Sequence[int],
        conv_channels: List[int] = [32, 64, 32],
        encoder_args: List[tuple] = [(8, 4), (4, 2), (3, 1)],
        decoder_args: List[tuple] = [(3, 1), (4, 2), (8, 4)],
        fc_units: List[int] = [256],
        z_dim: int = 32,
        activator: nn.Module = nn.ReLU(True),
        initializer: Initializer = Initializer(
            weight_init=orthogonal(nonlinearity="relu")
        ),
    ) -> None:
        super().__init__()
        in_channel = input_dim[0] if len(input_dim) == 3 else 1
        channels = [in_channel] + conv_channels
        self.encoder_conv = sequential_body(
            lambda i: (
                nn.Conv2d(channels[i], channels[i + 1], *encoder_args[i]),
                activator,
            ),
            len(channels) - 1,
        )
        self.cnn_hidden = calc_cnn_hidden(encoder_args, *input_dim[-2:])
        hidden = self.cnn_hidden[0] * self.cnn_hidden[1] * channels[-1]
        encoder_units = [hidden] + fc_units
        self.encoder_fc = sequential_body(
            lambda i: (nn.Linear(encoder_units[i], encoder_units[i + 1]), activator),
            len(encoder_units) - 1,
        )
        self.mu_fc = nn.Linear(encoder_units[-1], z_dim)
        self.logvar_fc = nn.Linear(encoder_units[-1], z_dim)
        decoder_units = [z_dim] + list(reversed(fc_units[:-1])) + [hidden]
        self.decoder_fc = sequential_body(
            lambda i: (nn.Linear(decoder_units[i], decoder_units[i + 1]), activator),
            len(decoder_units) - 1,
        )
        channels = list(reversed(conv_channels))
        deconv = construct_body(
            lambda i: (
                nn.ConvTranspose2d(channels[i], channels[i + 1], *decoder_args[i]),
                activator,
            ),
            len(channels) - 1,
        )
        self.decoder_deconv = nn.Sequential(
            *deconv, nn.ConvTranspose2d(channels[-1], in_channel, *decoder_args[-1])
        )
        self.z_dim = z_dim
        self.input_dim = input_dim
        initializer(self)

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h1 = self.encoder_conv(x)
        h1 = h1.view(h1.size(0), -1)
        h2 = self.encoder_fc(h1)
        return self.mu_fc(h2), self.logvar_fc(h2)

    def decode(self, z: Tensor, old_shape: Size = None) -> Tensor:
        h3 = self.decoder_fc(z)
        h3 = h3.view(h3.size(0), -1, *self.cnn_hidden)
        return self.decoder_deconv(h3)


def bernoulli_recons(a: Tensor, b: Tensor) -> Tensor:
    return nn.functional.binary_cross_entropy_with_logits(a, b, reduction="none")


def categorical_gray(a: Tensor, b: Tensor) -> Tensor:
    assert b.size(1) == 1
    categ = a.size(1)
    value = (b * float(categ)).round().long()
    logits = a - a.logsumexp(dim=1, keepdim=True)
    log_probs = torch.gather(logits, 1, value)
    return -log_probs


def categorical_binary(a: Tensor, b: Tensor) -> Tensor:
    assert a.size(1) == b.size(1)
    value = b.max(dim=1, keepdim=True)[1]
    logits = a - a.logsumexp(dim=1, keepdim=True)
    log_probs = torch.gather(logits, 1, value)
    return -log_probs


def gaussian_recons(a: Tensor, b: Tensor) -> Tensor:
    return torch.sigmoid(a).sub(b).pow(2)


def _recons_fn(decoder_type: str = "bernoulli") -> Callable[[Tensor, Tensor], Tensor]:
    if decoder_type == "bernoulli":
        recons_loss = bernoulli_recons
    elif decoder_type == "gaussian":
        recons_loss = gaussian_recons
    elif decoder_type == "categorical_gray":
        recons_loss = categorical_gray
    elif decoder_type == "categorical_binary":
        recons_loss = categorical_binary
    else:
        raise ValueError(f"{decoder_type} is not supported")
    return recons_loss


class VaeLoss(ABC):
    @abstractmethod
    def recons_loss(self, output: Tensor, img: Tensor) -> Tensor:
        pass

    @abstractmethod
    def latent_loss(self, logvar: Tensor, mu: Tensor) -> Tensor:
        pass


class BetaVaeLoss(VaeLoss):
    def __init__(self, beta: float = 1.0, decoder_type: str = "gaussian") -> None:
        self._recons_fn = _recons_fn(decoder_type)
        self.beta = beta

    def recons_loss(self, output: Tensor, img: Tensor) -> Tensor:
        return self._recons_fn(output, img)

    def latent_loss(self, logvar: Tensor, mu: Tensor) -> Tensor:
        kld = -0.5 * torch.sum(1.0 + logvar - mu.pow(2.0) - logvar.exp())
        return kld.mul(self.beta)


class VaeUnsupervisedBlock(UnsupervisedBlock):
    def __init__(self, vae: ConvVae, loss_fn: VaeLoss = BetaVaeLoss(beta=1.0)) -> None:
        super().__init__()
        self.vae = vae
        self.loss_fn = loss_fn

    def rewards(self, states: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        batch_size = states.size(0)
        out = self.vae(states)
        res = self.loss_fn.recons_loss(out.x, states).div_(batch_size)
        return res, None

    def loss(self, states: Tensor, target: Optional[Tensor]) -> Tensor:
        batch_size = states.size(0)
        out = self.vae(states)
        recons_loss = self.loss_fn.recons_loss(out.x, states).div_(batch_size)
        latent_loss = self.loss_fn.latent_loss(out.logvar, out.mu).div_(batch_size)
        print(recons_loss.shape, latent_loss.shape)
        return recons_loss + latent_loss

    @property
    def input_dim(self) -> Sequence[int]:
        return self.vae.input_dim


def normalize_vae(t: Tensor, rms: RunningMeanStdTorch) -> Tensor:
    t = t.reshape(-1, 1, *t.shape[-2:])
    t.sub_(rms.mean.float()).div_(rms.std().float())
    return t.clamp_(-5.0, 5.0).add_(5.0).div(10.0)


def irew_gen_vae(
    vae_loss: VaeLoss = BetaVaeLoss(beta=1.0),
    preprocess: PreProcessor = preprocess_default,
    state_normalizer: Normalizer = normalize_vae,
    reward_normalizer: Normalizer = normalize_r_default,
    **kwargs,
) -> Callable[[Config, Device], UnsupervisedIRewGen]:
    def _make_irew_gen(cfg: Config, device: Device) -> UnsupervisedIRewGen:
        input_dim = 1, *cfg.state_dim[1:]
        vae = ConvVae(input_dim, **kwargs)
        loss_fn = vae_loss
        return UnsupervisedIRewGen(
            VaeUnsupervisedBlock(vae, loss_fn),
            cfg.int_discount_factor,
            cfg.nworkers,
            device,
            preprocess=preprocess,
            state_normalizer=state_normalizer,
            reward_normalizer=reward_normalizer,
        )

    return _make_irew_gen
