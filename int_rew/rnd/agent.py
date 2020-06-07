from itertools import chain
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from rainy.agents import PPOAgent
from rainy.lib import mpi
from rainy.lib.rollout import RolloutSampler
from rainy.prelude import Action, Array, State

from ..rollout import IntValueRolloutStorage
from .config import RNDConfig
from .rollout import RNDRolloutSampler


class RNDAgent(PPOAgent):
    SAVED_MEMBERS = "net", "clip_eps", "clip_cooler", "optimizer", "irew_gen"

    def __init__(self, config: RNDConfig) -> None:
        super().__init__(config)
        self.logger.summary_setting(
            "intrew",
            ["update_steps"],
            interval=self.config.intrew_log_freq,
            color="magenta",
        )
        self.net = config.net("actor-critic")
        self.another_device = config.device.split()
        self.storage = IntValueRolloutStorage(
            config.nsteps,
            config.nworkers,
            config.device,
            config.discount_factor,
            another_device=self.another_device,
        )
        self.irew_gen = config.int_reward_gen(self.another_device)
        self.optimizer = config.optimizer(
            chain(self.net.parameters(), self.irew_gen.block.parameters())
        )
        self.lr_cooler = config.lr_cooler(self.optimizer.param_groups[0]["lr"])
        self.clip_cooler = config.clip_cooler()
        self.clip_eps = config.ppo_clip
        batch_size = self.config.nsteps * self.config.nworkers
        nbatches = batch_size // self.config.ppo_minibatch_size
        self.num_updates = self.config.ppo_epochs * nbatches
        mpi.setup_models(self.net, self.irew_gen.block)
        self.optimizer = mpi.setup_optimizer(self.optimizer)
        if not self.config.normalize_int_reward:
            self.irew_gen.reward_normalizer = lambda intrew, _rms: intrew

        self.logger.summary_setting(
            "intrew",
            ["update_steps"],
            interval=config.network_log_freq,
            color="magenta",
        )

    @torch.no_grad()
    def actions(self, states: Array[State]) -> Tuple[Array[Action], dict]:
        policy, value, pvalue, rnns = self.net(*self._network_in(states))
        actions = policy.action().squeeze().cpu().numpy()
        return actions, dict(rnn_states=rnns, policy=policy, value=value, pvalue=pvalue)

    @staticmethod
    def _rnd_value_loss(prediction: Tensor, target: Tensor) -> Tensor:
        return (prediction - target.to(prediction.device)).pow_(2.0).mul_(0.5).mean()

    def _reset(self, initial_states: Array[State]) -> None:
        self.storage.set_initial_state(initial_states, self.rnn_init())
        if self.config.initialize_stats is not None:
            self.initialize_stats(self.config.initialize_stats)

    def initialize_stats(self, t: int) -> None:
        states = []
        for i in range(self.config.nsteps * t):
            actions = np.random.randint(self.penv.action_dim, size=self.config.nworkers)
            s, _, _, _ = self.penv.step(actions)
            states.append(self.another_device.tensor(self.penv.extract(s)))
            if len(states) == self.config.nsteps:
                processed = self.irew_gen.preprocess(torch.cat(states))
                self.irew_gen.ob_rms.update(
                    processed.double().view(-1, *self.penv.state_dim[1:])
                )
                states.clear()

    def _update_policy(self, sampler: RNDRolloutSampler) -> None:
        p, v, iv, e = (0.0,) * 4
        for _ in range(self.config.ppo_epochs):
            for batch in sampler:
                policy, value, int_value, _ = self.net(
                    batch.states, batch.rnn_init, batch.masks
                )
                policy.set_action(batch.actions)
                policy_loss = self._proximal_policy_loss(
                    policy, batch.advantages, batch.old_log_probs
                )
                value_loss = self._rnd_value_loss(value, batch.returns)
                int_value_loss = self._rnd_value_loss(int_value, batch.int_returns)
                entropy_loss = policy.entropy().mean()
                self.optimizer.zero_grad()
                (
                    policy_loss
                    + self.config.value_loss_weight * value_loss
                    + self.config.value_loss_weight * int_value_loss
                    - self.config.entropy_weight * entropy_loss
                ).backward()
                aux_loss = self.irew_gen.aux_loss(
                    batch.states, batch.targets, self.config.auxloss_use_ratio
                )
                aux_loss.backward()
                mpi.clip_and_step(
                    self.net.parameters(), self.config.grad_clip, self.optimizer
                )
                p, v, e = (
                    p + policy_loss.item(),
                    v + value_loss.item(),
                    e + entropy_loss.item(),
                )
                iv += int_value_loss.item()

        p, v, iv, e = (x / self.num_updates for x in (p, v, iv, e))
        self.network_log(policy_loss=p, value_loss=v, int_value_loss=iv, entropy_loss=e)

    def train(self, last_states: Array[State]) -> None:
        with torch.no_grad():
            next_value, next_int_value = self.net.values(*self._network_in(last_states))

        cfg = self.config
        self.storage.calc_gae_returns(next_value, cfg.discount_factor, cfg.gae_lambda)
        normal_sampler = RolloutSampler(
            self.storage, self.penv, cfg.ppo_minibatch_size,
        )

        int_rewards, stats = self.irew_gen.gen_rewards(normal_sampler.states)
        stats["update_steps"] = self.update_steps
        self.logger.submit("intrew", **stats)

        self.storage.calc_int_returns(
            next_int_value,
            int_rewards,
            cfg.int_discount_factor,
            cfg.gae_lambda,
            cfg.int_use_mask,
        )

        self._update_policy(
            RNDRolloutSampler(
                normal_sampler,
                self.storage,
                self.irew_gen.cached_target,
                cfg.adv_weight,
                cfg.int_adv_weight,
                adv_normalize_eps=self.config.adv_normalize_eps,
            )
        )

        self.lr_cooler.lr_decay(self.optimizer)
        self.clip_eps = self.clip_cooler()
        self.storage.reset()
