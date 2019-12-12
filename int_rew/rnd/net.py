import copy
from rainy.net.actor_critic import policy_init
from rainy.net.policy import CategoricalDist, Policy, PolicyDist
from rainy.net import (
    DummyRnn,
    DQNConv,
    FcBody,
    LinearHead,
    NetworkBlock,
    RnnBlock,
    RnnState,
    SharedACNet,
)
from rainy.prelude import ArrayLike
from rainy.utils import Device
from torch import Tensor
from typing import Callable, List, Optional, Tuple, Type


class RNDACNet(SharedACNet):
    def __init__(
        self,
        body: NetworkBlock,
        actor_head: NetworkBlock,
        critic_head: NetworkBlock,
        policy_dist: PolicyDist,
        recurrent_body: RnnBlock = DummyRnn(),
        device: Device = Device(),
        int_critic_head: Optional[NetworkBlock] = None,
    ) -> None:
        super().__init__(
            body, actor_head, critic_head, policy_dist, recurrent_body, device
        )
        self.int_critic_head = (
            copy.deepcopy(self.critic_head)
            if int_critic_head is None
            else int_critic_head
        )
        self.int_critic_head.to(device.unwrapped)

    def values(
        self,
        states: ArrayLike,
        rnns: Optional[RnnState] = None,
        masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        features = self._features(states, rnns, masks)[0]
        ext_v, int_v = self.critic_head(features), self.int_critic_head(features)
        return ext_v.squeeze(), int_v.squeeze()

    def forward(
        self,
        states: ArrayLike,
        rnns: Optional[RnnState] = None,
        masks: Optional[Tensor] = None,
    ) -> Tuple[Policy, Tensor, Tensor, RnnState]:
        features, rnn_next = self._features(states, rnns, masks)
        policy = self.actor_head(features)
        ext_value = self.critic_head(features).squeeze()
        int_value = self.int_critic_head(features).squeeze()
        return self.policy_dist(policy), ext_value, int_value, rnn_next


def rnd_ac_conv(
    policy: Type[PolicyDist] = CategoricalDist,
    hidden_channels: Tuple[int, int, int] = (32, 64, 32),
    output_dim: int = 256,
    rnn: Type[RnnBlock] = DummyRnn,
    **kwargs
) -> Callable[[Tuple[int, int, int], int, Device], RNDACNet]:
    def _net(
        state_dim: Tuple[int, int, int], action_dim: int, device: Device
    ) -> RNDACNet:
        body = DQNConv(
            state_dim, hidden_channels=hidden_channels, output_dim=output_dim, **kwargs
        )
        policy_dist = policy(action_dim, device)
        rnn_ = rnn(body.output_dim, body.output_dim)
        ac_head = LinearHead(body.output_dim, policy_dist.input_dim, policy_init())
        cr_head = LinearHead(body.output_dim, 1)
        return RNDACNet(
            body, ac_head, cr_head, policy_dist, recurrent_body=rnn_, device=device
        )

    return _net  # type: ignore


def rnd_ac_fc(
    policy: Type[PolicyDist] = CategoricalDist,
    units: List[int] = [64, 64],
    output_dim: int = 256,
    rnn: Type[RnnBlock] = DummyRnn,
    **kwargs
) -> Callable[[Tuple[int, int, int], int, Device], RNDACNet]:
    def _net(
        state_dim: Tuple[int, int, int], action_dim: int, device: Device
    ) -> RNDACNet:
        body = FcBody(state_dim[0], units=units, **kwargs)
        policy_dist = policy(action_dim, device)
        rnn_ = rnn(body.output_dim, body.output_dim)
        ac_head = LinearHead(body.output_dim, policy_dist.input_dim, policy_init())
        cr_head = LinearHead(body.output_dim, 1)
        return RNDACNet(
            body, ac_head, cr_head, policy_dist, recurrent_body=rnn_, device=device
        )

    return _net  # type: ignore
