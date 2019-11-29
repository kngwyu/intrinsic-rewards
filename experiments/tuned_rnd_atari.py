import click
import os
import rainy.utils.cli as cli
from rainy.envs import Atari, atari_parallel
from torch.optim import Adam
from int_rew import rnd


def config(game: str = "MontezumaRevenge", rnd_lr: float = 5.0e-5) -> rnd.RNDConfig:
    c = rnd.RNDConfig()
    c.set_env(lambda: Atari(game, cfg=rnd.atari_config(), frame_stack=False))
    c.set_optimizer(lambda params: Adam(params, lr=rnd_lr), key="rnd_separated")
    c.set_parallel_env(atari_parallel())
    c.max_steps = int(1e8) * 6
    c.grad_clip = 1.0
    # ppo params
    c.nworkers = 64
    c.nsteps = 128
    c.value_loss_weight = 0.5
    c.gae_lambda = 0.95
    c.ppo_minibatch_size = (c.nworkers * c.nsteps) // 4
    c.auxloss_use_ratio = min(1.0, 32.0 / c.nworkers)
    c.use_reward_monitor = True
    # eval settings
    c.eval_env = Atari(game, cfg=rnd.atari_config())
    c.episode_log_freq = 100
    c.eval_freq = None
    c.save_freq = int(1e8)
    return c


if __name__ == "__main__":
    options = [click.Option(["--rnd-lr"], type=float, default=5.0e-5)]
    cli.run_cli(
        config, rnd.TunedRNDAgent, os.path.realpath(__file__), options,
    )
