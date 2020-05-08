import os
import rainy
from rainy.envs import Atari, atari_parallel
from torch.optim import Adam
from int_rew import rnd


@rainy.main(rnd.RNDAgent, script_path=os.path.realpath(__file__))
def config(
    envname: str = "MontezumaRevenge",
    max_steps: int = int(1e8) * 6,
    nworkers: int = 128,
    rnd_lr: float = 5.0e-5,
) -> rnd.RNDConfig:
    c = rnd.RNDConfig()
    c.set_env(lambda: Atari(envname, cfg="rnd", frame_stack=False))
    c.set_optimizer(lambda params: Adam(params, lr=rnd_lr), key="rnd_separated")
    c.set_parallel_env(atari_parallel())
    c.max_steps = max_steps
    c.grad_clip = 1.0
    # ppo params
    c.nworkers = nworkers
    c.nsteps = 128
    c.value_loss_weight = 1.0
    c.gae_lambda = 0.95
    c.ppo_minibatch_size = (c.nworkers * c.nsteps) // 4
    c.auxloss_use_ratio = min(1.0, 32.0 / c.nworkers)
    c.use_reward_monitor = True
    # eval settings
    c.eval_env = Atari(envname, cfg="rnd")
    c.episode_log_freq = 100
    c.eval_freq = max_steps // 10
    c.save_freq = max_steps // 5
    return c


if __name__ == "__main__":
    main()
