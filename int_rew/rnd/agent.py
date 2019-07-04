from itertools import chain
import numpy as np
from rainy import Config
from rainy.agents import PpoAgent
from rainy.lib import mpi
from rainy.lib.rollout import RolloutSampler
from rainy.prelude import Array, State
from rainy.utils.log import ExpStats
import torch
from torch import Tensor
from typing import Tuple
from .rollout import RndRolloutSampler, RndRolloutStorage


class RndPpoAgent(PpoAgent):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.net = config.net('actor-critic')
        rnd_device = config.device.split()
        self.storage = RndRolloutStorage(
            config.nsteps,
            config.nworkers,
            config.device,
            config.discount_factor,
            rnd_device=rnd_device,
        )
        self.irew_gen = config.int_reward_gen(config, rnd_device)
        self.optimizer = config.optimizer(chain(self.net.parameters(), self.irew_gen.params()))
        self.lr_cooler = config.lr_cooler(self.optimizer.param_groups[0]['lr'])
        self.clip_cooler = config.clip_cooler()
        self.clip_eps = config.ppo_clip
        nbatchs = (self.config.nsteps * self.config.nworkers) // self.config.ppo_minibatch_size
        self.num_updates = self.config.ppo_epochs * nbatchs
        self.intrew_stats = ExpStats()
        mpi.setup_models(self.net, self.irew_gen.target, self.irew_gen.predictor)
        self.optimizer = mpi.setup_optimizer(self.optimizer)

    def members_to_save(self) -> Tuple[str, ...]:
        return 'net', 'clip_eps', 'clip_cooler', 'optimizer', 'irew_gen'

    def _one_step(self, states: Array[State]) -> Array[State]:
        with torch.no_grad():
            policy, value, pvalue, rnns = self.net(*self._network_in(states))
        next_states, rewards, done, info = self.penv.step(policy.action().squeeze().cpu().numpy())
        self.episode_length += 1
        self.rewards += rewards
        self.report_reward(done, info)
        self.storage.push(next_states, rewards, done, rnn_state=rnns, policy=policy, value=value)
        self.storage.push_int_value(pvalue)
        return next_states

    @staticmethod
    def _rnd_value_loss(prediction: Tensor, target: Tensor) -> Tensor:
        return 0.5 * (prediction - target.to(prediction.device)).pow(2).mean()

    def initialize_stats(self, t: int) -> None:
        for i in range(self.config.nsteps * t):
            actions = np.random.randint(self.penv.action_dim, size=self.config.nworkers)
            states, rewards, done, _ = self.penv.step(actions)
            self.storage.push(states, rewards, done)
            if (i + 1) % self.config.nsteps == 0:
                s = self.irew_gen.preprocess(self.storage.batch_states(self.penv))
                self.irew_gen.ob_rms.update(s.double().view(-1, *self.penv.state_dim[1:]))
                self.storage.reset()

    def nstep(self, states: Array[State]) -> Array[State]:
        if self.update_steps == 0 and self.config.initialize_stats is not None:
            self.initialize_stats(self.config.initialize_stats)
            states = self.storage.states[0]
        for _ in range(self.config.nsteps):
            states = self._one_step(states)

        with torch.no_grad():
            next_value, next_int_value = self.net.values(*self._network_in(states))

        cfg = self.config
        self.storage.calc_gae_returns(next_value, cfg.discount_factor, cfg.gae_lambda)
        normal_sampler = RolloutSampler(
            self.storage,
            self.penv,
            cfg.ppo_minibatch_size,
            rnn=self.net.recurrent_body,
        )

        int_rewards = self.irew_gen.gen_rewards(normal_sampler.states)
        self.storage.calc_int_returns(
            next_int_value,
            int_rewards,
            cfg.int_discount_factor,
            cfg.gae_lambda,
            cfg.int_use_mask,
        )

        self.intrew_stats.update({
            'intrew_mean': int_rewards.mean().item(),
            'intrew_max': int_rewards.max().item(),
            'intrew_min': int_rewards.min().item(),
        })
        if self.update_steps > 0 and self.update_steps % self.config.intrew_log_freq == 0:
            d = self.intrew_stats.report_and_reset()
            d['update-steps'] = self.update_steps
            self.logger.exp('intrew', d)

        p, v, iv, e = (0.0,) * 4
        sampler = RndRolloutSampler(
            normal_sampler,
            self.storage,
            self.irew_gen.cached_target,
            cfg.adv_weight,
            cfg.int_adv_weight,
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
                 + self.config.value_loss_weight * int_value_loss
                 - self.config.entropy_weight * entropy_loss).backward()
                aux_loss = self.irew_gen.aux_loss(
                    batch.states,
                    batch.targets,
                    cfg.auxloss_use_ratio
                )
                aux_loss.backward()
                mpi.clip_and_step(self.net, self.config.grad_clip, self.optimizer)
                p, v, e = p + policy_loss.item(), v + value_loss.item(), e + entropy_loss.item()
                iv += int_value_loss.item()

        self.lr_cooler.lr_decay(self.optimizer)
        self.clip_eps = self.clip_cooler()
        self.storage.reset()

        p, v, iv, e = map(lambda x: x / float(self.num_updates), (p, v, iv, e))
        self.report_loss(policy_loss=p, value_loss=v, int_value_loss=iv, entropy_loss=e)
        return states
