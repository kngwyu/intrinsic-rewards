"""Variational Auto Encoder with CNN
Based on https://github.com/kngwyu/pytorch-autoencoders/
"""
from abc import ABC, abstractmethod
from itertools.chain import from_iterable as flatten
import torch
from torch import nn, Size, Tensor
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple
from rainy.net import calc_cnn_hidden, Initializer
from rainy.utils import Device
from rainy.utils.rms import RunningMeanStdTorch

from .common import IntRewardBlock, IntRewardGenerator, normalize_r_default


class VaeOutPut(NamedTuple):
    x: Tensor
    mu: Tensor
    logvar: Tensor


class ConvVae(nn.Module):
    """VAE with CNN.
       Default network parameters are cited from https://arxiv.org/abns/1804.03599.
    """
    def __init__(
            self,
            input_dim: Sequence[int],
            conv_channels: List[int] = [32, 32, 32, 32],
            conv_args: List[tuple] = [(4, 2, 1), (4, 2, 1), (4, 2, 1), (4, 2, 1)],
            fc_units: List[int] = [256, 256],
            z_dim: int = 20,
            activator: nn.Module = nn.ReLU(True),
            initializer: Initializer = Initializer(nonlinearity='relu'),
    ) -> None:
        super().__init__()
        in_channel = input_dim[0] if len(input_dim) == 3 else 1
        channels = [in_channel] + conv_channels
        self.encoder_conv = nn.Sequential(*flatten([
            (nn.Conv2d(channels[i], channels[i + 1], *conv_args[i]), activator)
            for i in range(len(channels) - 1)
        ]))
        self.cnn_hidden = calc_cnn_hidden(conv_args, *input_dim[-2:])
        hidden = self.cnn_hidden[0] * self.cnn_hidden[1] * channels[-1]
        encoder_units = [hidden] + fc_units
        self.encoder_fc = nn.Sequential(*flatten([
            (nn.Linear(encoder_units[i], encoder_units[i + 1]), activator)
            for i in range(len(encoder_units) - 1)
        ]))
        self.mu_fc = nn.Linear(encoder_units[-1], z_dim)
        self.logvar_fc = nn.Linear(encoder_units[-1], z_dim)
        decoder_units = [z_dim] + list(reversed(fc_units[:-1])) + [hidden]
        self.decoder_fc = nn.Sequential(*flatten([
            (nn.Linear(decoder_units[i], decoder_units[i + 1]), activator)
            for i in range(len(decoder_units) - 1)
        ]))
        channels = list(reversed(conv_channels))
        deconv = flatten([(
            nn.ConvTranspose2d(channels[i], channels[i + 1], *conv_args[i]),
            activator
        ) for i in range(len(channels) - 1)])
        self.decoder_deconv = nn.Sequential(
            *deconv,
            nn.ConvTranspose2d(channels[-1], in_channel, *conv_args[-1])
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

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x: Tensor) -> VaeOutPut:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return VaeOutPut(self.decode(z, old_shape=x.shape), mu, logvar)


def bernoulli_recons(a: Tensor, b: Tensor) -> Tensor:
    return nn.functional.binary_cross_entropy_with_logits(a, b, reduction='sum')


def categorical_binary(a: Tensor, b: Tensor) -> Tensor:
    assert a.size(1) == b.size(1)
    value = b.max(dim=1, keepdim=True)[1]
    logits = a - a.logsumexp(dim=1, keepdim=True)
    log_probs = torch.gather(logits, 1, value)
    return -log_probs.sum()


def categorical_gray(a: Tensor, b: Tensor) -> Tensor:
    assert b.size(1) == 1
    categ = a.size(1)
    value = (b * float(categ)).round().long()
    logits = a - a.logsumexp(dim=1, keepdim=True)
    log_probs = torch.gather(logits, 1, value)
    return -log_probs.sum()


def gaussian_recons(a: Tensor, b: Tensor) -> Tensor:
    return nn.functional.mse_loss(torch.sigmoid(a), b, reduction='sum')


def _recons_fn(decoder_type: str = 'bernoulli') -> Callable[[Tensor, Tensor], Tensor]:
    if decoder_type == 'bernoulli':
        recons_loss = bernoulli_recons
    elif decoder_type == 'gaussian':
        recons_loss = gaussian_recons
    elif decoder_type == 'categorical_gray':
        recons_loss = categorical_gray
    elif decoder_type == 'categorical_binary':
        recons_loss = categorical_binary
    else:
        raise ValueError('Currently only bernoulli and gaussian are supported as decoder head')
    return recons_loss


class VaeLoss(ABC):
    @abstractmethod
    def recons_loss(self, output: Tensor, img: Tensor) -> Tensor:
        pass

    @abstractmethod
    def latent_loss(self, logvar: Tensor, mu: Tensor) -> Tensor:
        pass


class BetaVaeLoss(VaeLoss):
    def __init__(self, beta: float = 4.0, decoder_type: str = 'bernoulli') -> None:
        self._recons_fn = _recons_fn(decoder_type)
        self.beta = beta

    def recons_loss(self, output: Tensor, img: Tensor) -> Tensor:
        return self._recons_fn(output, img)

    def latent_loss(self, logvar: Tensor, mu: Tensor) -> Tensor:
        kld = -0.5 * \
            torch.sum(1.0 + logvar - mu.pow(2.0) - logvar.exp())
        return kld.mul(self.beta)


class VaeIntRewardBlock(IntRewardBlock):
    def __init__(self, vae: ConvVae, loss_fn: VaeLoss = BetaVaeLoss(beta=0.0)) -> None:
        super().__init__()
        self.vae = vae
        self.loss_fn = loss_fn

    def rewards(self, states: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        batch_size = states.size(0)
        out = self.vae(states)
        return self.loss_fn.recons_loss(out.x).div_(batch_size), None

    def loss(self, states: Tensor, target: Optional[Tensor]) -> Tensor:
        batch_size = states.size(0)
        out = self.vae(states)
        recons_loss = self.loss_fn.recons_loss(out.x).div_(batch_size)
        latent_loss = self.loss_fn.latent_loss(out.logvar, out.mu).div_(batch_size)
        return recons_loss + latent_loss

    def input_dim(self) -> Sequence[int]:
        return self.vae.input_dim


def preprocess_vae(t: Tensor, device: Device) -> Tensor:
    """Extract one channel and rescale to 0..255
    """
    return t.to(device.unwrapped)[:, -1]


def normalize_vae(t: Tensor, rms: RunningMeanStdTorch) -> Tensor:
    t = t.reshape(-1, 1, *t.shape[-2:])
    return t.sub_(rms.mean.float()).div_(rms.std().float())


def irew_gen_vae(
        preprocess: callable = preprocess_vae,
        state_normalizer: callable = normalize_vae,
        reward_normalizer: callable = normalize_r_default,
) -> Callable[['RndConfig', Device], IntRewardGenerator]:
    def _make_irew_gen(cfg: 'RndConfig', device: Device) -> IntRewardGenerator:
        input_dim = 1, *cfg.state_dim[1:]
        vae = ConvVae(
            input_dim,
            conv_channels=[32, 32, 64, 64],
            conv_args=[(4, 2, 1), (4, 2, 1), (4, 2, 1), (4, 2, 1)],
            fc_units=[256],
            z_dim=32,
        )
        loss_fn = BetaVaeLoss(beta=0.0, decoder_type='gaussian')
        return IntRewardGenerator(
            VaeIntRewardBlock(vae, loss_fn),
            cfg.int_discount_factor,
            cfg.nworkers,
            device,
            preprocess=preprocess,
            state_normalizer=state_normalizer,
            reward_normalizer=reward_normalizer,
        )
    return _make_irew_gen
