from itertools import chain
import numpy as np
from rainy.agents import PPOAgent
from rainy.lib import mpi
from rainy.lib.rollout import RolloutSampler
from rainy.prelude import Array, State
import torch
from torch import Tensor
from .config import RNDConfig
from .rollout import RNDRolloutSampler
from ..rollout import IntValueRolloutStorage


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
        another_device = config.device.split()
        self.storage = IntValueRolloutStorage(
            config.nsteps,
            config.nworkers,
            config.device,
            config.discount_factor,
            another_device=another_device,
        )
        self.irew_gen = config.int_reward_gen(another_device)
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

    def _one_step(self, states: Array[State]) -> Array[State]:
        with torch.no_grad():
            policy, value, pvalue, rnns = self.net(*self._network_in(states))
        next_states, rewards, done, info = self.penv.step(
            policy.action().squeeze().cpu().numpy()
        )
        self.episode_length += 1
        self.rewards += rewards
        self.report_reward(done, info)
        self.storage.push(
            next_states, rewards, done, rnn_state=rnns, policy=policy, value=value
        )
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
                self.irew_gen.ob_rms.update(
                    s.double().view(-1, *self.penv.state_dim[1:])
                )
                self.storage.reset()

    def _update_policy(self, sampler: RNDRolloutSampler) -> None:
        p, v, iv, e = (0.0,) * 4
        for _ in range(self.config.ppo_epochs):
            for batch in sampler:
                policy, value, int_value, _ = self.net(
                    batch.states, batch.rnn_init, batch.masks
                )
                policy.set_action(batch.actions)
                policy_loss = self._policy_loss(
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
        self.report_loss(policy_loss=p, value_loss=v, int_value_loss=iv, entropy_loss=e)

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

        int_rewards = self.irew_gen.gen_rewards(
            normal_sampler.states, reporter=self.logger
        )

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
        return states
