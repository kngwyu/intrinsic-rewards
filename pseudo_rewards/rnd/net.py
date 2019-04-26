import copy
from rainy.net import RnnState, Policy, SharedBodyACNet
from rainy.prelude import Array
from torch import Tensor
from typing import Optional, Tuple, Union


class RndACNet(SharedBodyACNet):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.internal_critic_head = copy.deepcopy(self.critic_head)

    def values(
            self,
            states: Union[Array, Tensor],
            rnns: Optional[RnnState] = None,
            masks: Optional[Tensor] = None,
    ) -> Tensor:
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
        return self.policy_head(policy), ext_value, int_value, rnn_next
