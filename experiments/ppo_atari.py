"""PPO Atari
"""
import os
import rainy
from rainy.envs import Atari, atari_parallel
from torch.optim import Adam


@rainy.main(rainy.agents.PPOAgent, script_path=os.path.realpath(__file__))
def main(
    envname: str = "MontezumaRevenge",
    max_steps: int = int(1e8) * 6,
    nworkers: int = 128,
) -> rainy.Config:
    c = rainy.Config()
    c.set_env(lambda: Atari(envname, cfg="rnd", frame_stack=False))
    c.set_parallel_env(atari_parallel())
    c.set_net_fn("actor-critic", rainy.net.actor_critic.conv_shared())
    c.set_optimizer(lambda params: Adam(params, lr=1.0e-4, eps=1.0e-8))
    c.max_steps = max_steps
    c.grad_clip = 1.0
    # ppo params
    c.discount_factor = 0.999
    c.entropy_weight = 0.001
    c.ppo_epochs = 4
    c.ppo_clip = 0.1
    c.use_gae = True
    c.nworkers = nworkers
    c.nsteps = 128
    c.value_loss_weight = 1.0
    c.gae_lambda = 0.95
    c.ppo_minibatch_size = (c.nworkers * c.nsteps) // 4
    # eval settings
    c.eval_env = Atari(envname, cfg="rnd")
    c.episode_log_freq = 100
    c.eval_times = 12
    c.eval_freq = c.max_steps // 10
    c.save_freq = None
    return c


if __name__ == "__main__":
    main()
