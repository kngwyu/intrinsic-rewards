"""PPO Deepsea
"""
import click
import os
import rainy
import rainy.utils.cli as cli
from rainy.envs import DeepSea, MultiProcEnv
from torch.optim import Adam


def config(size: int = 20) -> rainy.Config:
    c = rainy.Config()
    c.set_env(lambda: DeepSea(size))
    c.set_parallel_env(MultiProcEnv)
    c.max_steps = int(1e5)
    c.nworkers = 8
    c.nsteps = max(size, 8)
    c.entropy_weight = 0.1
    c.set_parallel_env(MultiProcEnv)
    c.set_optimizer(lambda params: Adam(params, lr=2.5e-4, eps=1.0e-4))
    c.value_loss_weight = 0.2
    c.entropy_weight = 0.001
    c.grad_clip = 0.1
    c.gae_lambda = 0.95
    c.ppo_minibatch_size = 32
    c.use_gae = True
    c.ppo_clip = 0.2
    c.eval_freq = 1000
    c.eval_times = 4
    return c


if __name__ == "__main__":
    options = [click.Option(["--size"], type=int, default=20)]
    cli.run_cli(config, rainy.agents.PPOAgent, os.path.realpath(__file__), options)
