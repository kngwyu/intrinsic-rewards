import os
from typing import Tuple, Union

from torch.optim import Adam

import rainy
from rogue_gym.envs import (
    DungeonType,
    ImageSetting,
    ParallelRogueEnv,
    RogueEnv,
    StatusFlag,
)
from rogue_gym_patch import ParallelRogueEnvExt, RogueEnvExt


def rogue_config(seed: Union[int, Tuple[int, int]]) -> dict:
    common = {
        "width": 32,
        "height": 16,
        "hide_dungeon": True,
        "dungeon": {"style": "rogue", "room_num_x": 2, "room_num_y": 2},
        "enemies": {"enemies": []},
    }
    if isinstance(seed, int):
        common["seed"] = seed
    else:
        common["seed_range"] = seed
    return common


EXPAND = ImageSetting(dungeon=DungeonType.GRAY, status=StatusFlag.EMPTY)


@rainy.main(rainy.agents.PPOAgent, script_path=os.path.realpath(__file__))
def main() -> rainy.Config:
    c = rainy.Config()
    c.set_parallel_env(
        lambda _env_gen, _num_w: ParallelRogueEnvExt(
            ParallelRogueEnv(
                [rogue_config(2)] * c.nworkers, max_steps=500, image_setting=EXPAND,
            )
        )
    )
    c.max_steps = int(2e7) * 2
    c.save_freq = None
    c.eval_freq = None
    c.eval_env = RogueEnvExt(
        RogueEnv(
            config_dict=rogue_config(2),
            mex_steps=500,
            stair_reward=50.0,
            image_setting=EXPAND,
        )
    )
    c.set_optimizer(lambda params: Adam(params, lr=1.0e-4, eps=1.0e-6))
    CNN_PARAM = [(8, 1), (4, 1), (3, 1)]
    c.set_net_fn(
        "actor-critic",
        rainy.net.actor_critic.conv_shared(cnn_params=CNN_PARAM, feature_dim=256),
    )
    c.nworkers = 32
    c.nsteps = 125
    c.value_loss_weight = 1.0
    c.gae_lambda = 0.95
    c.ppo_minibatch_size = (c.nworkers * c.nsteps) // 4
    return c


if __name__ == "__main__":
    main()
