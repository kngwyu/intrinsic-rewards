import os
from rainy import Config
import rainy.utils.cli as cli
from rainy.envs import Atari, atari_parallel
from pseudo_rewards import rnd
from torch.optim import Adam


def config() -> Config:
    c = rnd.default_config()
    c.set_env(lambda: Atari('Breakout', frame_stack=False))
    c.set_net_fn('actor-critic', rnd.net.rnd_ac_conv())
    c.set_parallel_env(atari_parallel())
    c.set_optimizer(lambda params: Adam(params, lr=2.5e-4, eps=1.0e-4))
    c.max_steps = int(2e7)
    c.grad_clip = 0.5
    # ppo params
    c.nworkers = 8
    c.nsteps = 128
    c.value_loss_weight = 0.5
    c.gae_lambda = 0.95
    c.ppo_minibatch_size = 32 * 8
    c.ppo_clip = 0.1
    c.ppo_epochs = 3
    c.use_gae = True
    c.use_reward_monitor = True
    c.lr_min = None  # set 0.0 if you decrease ppo_clip
    # eval settings
    c.eval_env = Atari('Breakout')
    c.episode_log_freq = 100
    c.eval_freq = None
    c.save_freq = None
    return c


if __name__ == '__main__':
    cli.run_cli(config(), rnd.RndPpoAgent, script_path=os.path.realpath(__file__))
