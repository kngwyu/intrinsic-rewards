"""PPO Atari
"""
import os
import rainy
import rainy.utils.cli as cli
from rainy.envs import Atari, atari_parallel
from rnd import atari_config
from torch.optim import Adam


def config() -> rainy.Config:
    c = rainy.Config()
    c.set_env(lambda: Atari('Venture', cfg=atari_config(), frame_stack=False))
    c.set_parallel_env(atari_parallel())
    c.set_optimizer(lambda params: Adam(params, lr=1.0e-4, eps=1.0e-8))
    c.max_steps = int(1e8)
    c.grad_clip = 1.0
    # ppo params
    c.discount_factor = 0.999
    c.entropy_weight = 0.001
    c.ppo_epochs = 4
    c.ppo_clip = 0.1
    c.use_gae = True
    c.nworkers = 64
    c.nsteps = 128
    c.value_loss_weight = 0.5 * 0.5
    c.gae_lambda = 0.95
    c.ppo_minibatch_size = (c.nworkers * c.nsteps) // 4
    c.use_reward_monitor = True
    # eval settings
    c.eval_env = Atari('Venture', cfg=atari_config())
    c.episode_log_freq = 100
    c.eval_freq = None
    c.save_freq = None
    return c


if __name__ == '__main__':
    cli.run_cli(config(), rainy.agents.PpoAgent, script_path=os.path.realpath(__file__))
