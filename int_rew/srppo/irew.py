import numpy as np
from rainy.net import calc_cnn_hidden, NetworkBlock
from rainy.net.init import Initializer, uniform
from rainy.net.prelude import Params
from rainy.utils import Device
from rainy.utils.log import ExpStats
from rainy.utils.rms import RunningMeanStdTorch
import torch
from torch import nn, Tensor
from typing import Callable, List, Optional, Sequence, Tuple, Union

from ..supervised import SupervisedBlock, SupervisedIRewGen


class SrBlock(SupervisedBlock):
    def __init__(
            self,
            input_dim: Sequence[int],
            action_dim: int,
            hidden_dim: int = 2048,
            channels: List[int] = [64, 64, 64],
            conv_args: List[tuple] = [(6, 2), (6, 2, 1), (6, 2, 1)],
            fc_additional: int = 1,
            activ1: nn.Module = nn.LeakyReLU(negative_slope=0.2, inplace=True),
            activ2: nn.Module = nn.ReLU(inplace=True),
            init: Initializer = Initializer(nonlinearity='relu'),
            init_enc: Initializer = Initializer(weight_init=uniform(min=-1.0, max=1.0)),
            init_act: Initializer = Initializer(weight_init=uniform(min=-0.1, max=0.1)),
    ):
        super().__init__()

        in_channel = input_dim[0]
        channels = [in_channel] + channels
        self.conv = init.make_list([
            nn.Conv2d(channels[i], channels[i + 1], *conv_args[i])
            for i in range(len(channels) - 1)
        ])

        conved_dim = np.prod(calc_cnn_hidden(conv_args, *input_dim[1:])) * channels[-1]

        self.fc_enc = init(nn.Linear(conved_dim, hidden_dim))

        self.w_enc = init_enc(nn.Linear(hidden_dim, hidden_dim))
        self.w_action = init_act(nn.Linear(action_dim, hidden_dim))

        self.fc_action_trans = nn.Linear(hidden_dim, hidden_dim)

        self.fc_dec = nn.Linear(hidden_dim, conved_dim)

        channels.reverse()
        conv_args.reverse()
        self.deconv = init.make_list([
            nn.ConvTranspose2d(channels[i], channels[i + 1], *conv_args[i])
            for i in range(len(channels) - 1)
        ])

    def rewards(self, state: Tensor, target: Tensor, action: Tensor):
        pass
