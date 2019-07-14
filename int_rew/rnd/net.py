import copy
from rainy.net.actor_critic import policy_init
from rainy.net.policy import CategoricalDist, Policy, PolicyDist
from rainy.net import DummyRnn, DqnConv, LinearHead, RnnBlock, RnnState, SharedBodyACNet
from rainy.prelude import Array
from rainy.utils import Device
from torch import Tensor
from typing import Callable, Optional, Tuple, Union


class RndACNet(SharedBodyACNet):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.internal_critic_head = copy.deepcopy(self.critic_head)

    def values(
            self,
            states: Union[Array, Tensor],
            rnns: Optional[RnnState] = None,
            masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        features = self._features(states, rnns, masks)[0]
        ext_v, int_v = self.critic_head(features), self.internal_critic_head(features)
        return ext_v.squeeze(), int_v.squeeze()

    def forward(
            self,
            states: Union[Array, Tensor],
            rnns: Optional[RnnState] = None,
            masks: Optional[Tensor] = None,
    ) -> Tuple[Policy, Tensor, Tensor, RnnState]:
        features, rnn_next = self._features(states, rnns, masks)
        policy = self.actor_head(features)
        ext_value = self.critic_head(features).squeeze()
        int_value = self.internal_critic_head(features).squeeze()
        return self.policy_dist(policy), ext_value, int_value, rnn_next


def rnd_ac_conv(
        policy: Callable[[int, Device], PolicyDist] = CategoricalDist,
        hidden_channels: Tuple[int, int, int] = (32, 64, 32),
        output_dim: int = 256,
        rnn: Callable[[int, int], RnnBlock] = DummyRnn,
        **kwargs
) -> Callable[[Tuple[int, int, int], int, Device], RndACNet]:
    def _net(state_dim: Tuple[int, int, int], action_dim: int, device: Device) -> RndACNet:
        body = DqnConv(state_dim, hidden_channels=hidden_channels, output_dim=output_dim, **kwargs)
        policy_dist = policy(action_dim, device)
        rnn_ = rnn(body.output_dim, body.output_dim)
        ac_head = LinearHead(body.output_dim, policy_dist.input_dim, policy_init())
        cr_head = LinearHead(body.output_dim, 1)
        return RndACNet(body, ac_head, cr_head, policy_dist, recurrent_body=rnn_, device=device)
    return _net  # type: ignore
