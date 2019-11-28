import os
import rainy.utils.cli as cli
from rainy.envs import Atari, atari_parallel
from int_rew import rnd


def config(game: str = "MontezumaRevenge") -> rnd.RNDConfig:
    c = rnd.RNDConfig()
    c.set_env(lambda: Atari(game, cfg=rnd.atari_config(), frame_stack=False))
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
    cli.run_cli(config, rnd.TunedRNDAgent, script_path=os.path.realpath(__file__))
