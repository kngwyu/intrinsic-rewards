from itertools import chain
from rainy import Config
from rainy.agents import PpoAgent
from rainy.prelude import Array, State
import torch
from torch import Tensor
from torch import nn
from typing import Tuple
from .rollout import RndRolloutSampler, RndRolloutStorage


class RndPpoAgent(PpoAgent):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.net = config.net('actor-critic')
        rnd_device = config.device.split()
        self.storage = RndRolloutStorage(
            self.storage,
            config.nsteps,
            config.nworkers,
            config.discount_factor,
        )
        self.irew_gen = config.int_reward_gen(self.penv.state_dim, rnd_device)
        self.optimizer = config.optimizer(chain(self.net.parameters(), self.irew_gen.params()))
        self.lr_cooler = config.lr_cooler(self.optimizer.param_groups[0]['lr'])
        self.clip_cooler = config.clip_cooler()
        self.clip_eps = config.ppo_clip
        nbatchs = -(-self.config.nsteps * self.config.nworkers) // self.config.ppo_minibatch_size
        self.num_updates = self.config.ppo_epochs * nbatchs

    def members_to_save(self) -> Tuple[str, ...]:
        return 'net', 'clip_eps', 'clip_cooler', 'optimizer'

    def _one_step(self, states: Array[State]) -> Array[State]:
        net_in = self._network_in(states)
        with torch.no_grad():
            policy, value, pvalue, rnns = self.net(*net_in)
        next_states, rewards, done, info = self.penv.step(policy.action().squeeze().cpu().numpy())
        self.episode_length += 1
        self.rewards += rewards
        self.report_reward(done, info)
        self.storage.push(next_states, rewards, done, rnn_state=rnns, policy=policy, value=value)
        self.storage.push_int_rewards(self.irew_gen(net_in[0]), pvalue)
        return next_states

    @staticmethod
    def _rnd_value_loss(prediction: Tensor, target: Tensor) -> Tensor:
        return 0.5 * (prediction - target.to(prediction.device)).pow(2).mean()

    def nstep(self, states: Array[State]) -> Array[State]:
        for _ in range(self.config.nsteps):
            states = self._one_step(states)

        with torch.no_grad():
            next_value = self.net.value(*self._network_in(states))

        conf = self.config
        self.storage.calc_gae_returns(next_value, conf.discount_factor, conf.gae_lambda)
        self.storage.calc_int_returns(
            next_value,
            conf.int_discount_factor,
            conf.gae_lambda,
            conf.int_use_mask,
        )

        p, v, pv, e = (0.0,) * 4
        sampler = RndRolloutSampler(
            self.storage,
            self.penv,
            conf.ppo_minibatch_size,
            conf.adv_weight,
            conf.int_adv_weight,
            rnn=self.net.recurrent_body,
            adv_normalize_eps=self.config.adv_normalize_eps,
        )
        for _ in range(self.config.ppo_epochs):
            for batch in sampler:
                policy, value, int_value, _ = \
                    self.net(batch.states, batch.rnn_init, batch.masks)
                policy.set_action(batch.actions)
                policy_loss = self._policy_loss(policy, batch.advantages, batch.old_log_probs)
                value_loss = self._rnd_value_loss(value, batch.returns)
                int_value_loss = self._rnd_value_loss(int_value, batch.int_returns)
                entropy_loss = policy.entropy().mean()
                self.optimizer.zero_grad()
                (policy_loss
                 + self.config.value_loss_weight * value_loss
                 + self.config.int_value_loss_weight * int_value_loss
                 - self.config.entropy_weight * entropy_loss).backward()
                aux_loss = self.irew_gen.aux_loss(batch.states)
                aux_loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.config.grad_clip)
                self.optimizer.step()
                p, v, e = p + policy_loss.item(), v + value_loss.item(), e + entropy_loss.item()
                pv += int_value_loss.item()

        self.lr_cooler.lr_decay(self.optimizer)
        self.clip_eps = self.clip_cooler()
        self.storage.reset()

        p, v, pv, e = map(lambda x: x / float(self.num_updates), (p, v, pv, e))
        self.report_loss(policy_loss=p, value_loss=v, entropy_loss=e)
        return states
