import os
from rainy import Config
import rainy.utils.cli as cli
from rainy.envs import Atari, atari_parallel
from int_rew import rnd
from torch.optim import Adam


def config() -> Config:
    c = rnd.default_config()
    c.set_env(lambda: Atari('Venture', frame_stack=False))
    c.set_net_fn('actor-critic', rnd.net.rnd_ac_conv())
    c.set_parallel_env(atari_parallel())
    c.set_optimizer(lambda params: Adam(params, lr=1.0e-4, eps=1.0e-4))
    c.max_steps = int(2e7)
    c.grad_clip = 0.5
    # ppo params
    c.nworkers = 32
    c.nsteps = 128
    c.value_loss_weight = 0.5
    c.gae_lambda = 0.95
    c.ppo_minibatch_size = (32 * 128) // 4
    c.ppo_clip = 0.1
    c.ppo_epochs = 4
    c.use_gae = True
    c.use_reward_monitor = True
    # eval settings
    c.eval_env = Atari('Venture')
    c.episode_log_freq = 100
    c.eval_freq = None
    c.save_freq = None
    return c


if __name__ == '__main__':
    cli.run_cli(config(), rnd.RndPpoAgent, script_path=os.path.realpath(__file__))
