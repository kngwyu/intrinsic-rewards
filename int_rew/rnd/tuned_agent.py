from rainy.agents import PPOAgent
from rainy.lib import mpi

from ..rollout import IntValueRolloutStorage
from .agent import RNDAgent
from .config import RNDConfig
from .rollout import RNDRolloutSampler


class TunedRNDAgent(RNDAgent):
    def __init__(self, config: RNDConfig) -> None:
        PPOAgent.__init__(self, config)
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
        self.lr_cooler = config.lr_cooler(self.optimizer.param_groups[0]["lr"])
        self.clip_cooler = config.clip_cooler()
        self.clip_eps = config.ppo_clip
        batch_size = self.config.nsteps * self.config.nworkers
        nbatches = batch_size // self.config.ppo_minibatch_size
        self.num_updates = self.config.ppo_epochs * nbatches

        mpi.setup_models(self.net, self.irew_gen.block)
        self.optimizer = mpi.setup_optimizer(config.optimizer(self.net.parameters()))
        self.rnd_optimizer = mpi.setup_optimizer(
            config.optimizer(self.irew_gen.block.parameters(), key="rnd")
        )
        if not self.config.normalize_int_reward:
            self.irew_gen.reward_normalizer = lambda intrew, _rms: intrew

        self.logger.summary_setting(
            "intrew",
            ["update_steps"],
            interval=config.network_log_freq,
            color="magenta",
        )

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
                mpi.clip_and_step(
                    self.net.parameters(), self.config.grad_clip, self.optimizer
                )
                p, v, e = (
                    p + policy_loss.item(),
                    v + value_loss.item(),
                    e + entropy_loss.item(),
                )
                iv += int_value_loss.item()

                self.rnd_optimizer.zero_grad()
                aux_loss = self.irew_gen.aux_loss(
                    batch.states, batch.targets, self.config.auxloss_use_ratio
                )
                aux_loss.backward()
                mpi.clip_and_step(
                    self.irew_gen.block.parameters(),
                    self.config.grad_clip,
                    self.rnd_optimizer,
                )

        p, v, iv, e = (x / self.num_updates for x in (p, v, iv, e))
        self.network_log(policy_loss=p, value_loss=v, int_value_loss=iv, entropy_loss=e)
