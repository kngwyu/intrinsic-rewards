"""RND Deepsea
"""
import click
import os
import rainy.utils.cli as cli
from rainy.envs import DeepSea, MultiProcEnv
from int_rew import rnd


def config(size: int = 20) -> rnd.RNDConfig:
    c = rnd.RNDConfig()
    c.set_env(lambda: DeepSea(size))
    c.set_parallel_env(MultiProcEnv)
    c.max_steps = int(1e5)
    c.grad_clip = 1.0
    # PPO params
    c.nworkers = 8
    c.nsteps = max(size, 8)
    c.entropy_weight = 0.01
    c.value_loss_weight = 0.5
    c.gae_lambda = 0.95
    c.ppo_minibatch_size = 32
    c.auxloss_use_ratio = min(1.0, 32.0 / c.nworkers)
    c.set_net_fn("actor-critic", rnd.net.rnd_ac_fc())
    c._int_reward_gen = rnd.irew_gen_fc()
    # eval settings
    c.eval_freq = 1000
    c.episode_log_freq = 100
    c.eval_times = 4
    c.eval_deterministic = False
    return c


if __name__ == "__main__":
    options = [click.Option(["--size"], type=int, default=20)]
    cli.run_cli(config, rnd.RNDAgent, os.path.realpath(__file__), options)
