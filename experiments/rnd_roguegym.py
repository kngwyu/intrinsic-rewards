import os
from typing import Tuple, Union

from torch.optim import Adam

import rainy
from int_rew import rnd
from rogue_gym.envs import (
    DungeonType,
    ImageSetting,
    ParallelRogueEnv,
    RogueEnv,
    StatusFlag,
)
from rogue_gym.rainy_impls import ParallelRogueEnvExt, RogueEnvExt


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


@rainy.main(rnd.RNDAgent, script_path=os.path.realpath(__file__))
def main() -> rnd.RNDConfig:
    c = rnd.RNDConfig()
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
    c.set_optimizer(lambda params: Adam(params, lr=1.0e-4, eps=1.0e-8))
    CNN_PARAM = [(8, 1), (4, 1), (3, 1)]
    c.set_net_fn(
        "actor-critic",
        rnd.net.rnd_ac_conv(kernel_and_strides=CNN_PARAM, output_dim=256,),
    )
    c._int_reward_gen = rnd.irew.irew_gen_default(
        cnn_params=CNN_PARAM,
        hidden_channels=(32, 64, 32),
        feature_dim=256,
        preprocess=lambda t, device: t.to(device.unwrapped),
        state_normalizer=lambda t, rms: t.reshape(-1, 1, *t.shape[-2:]),
    )
    c.nworkers = 32
    c.nsteps = 125
    c.value_loss_weight = 1.0
    c.gae_lambda = 0.95
    c.ppo_minibatch_size = (c.nworkers * c.nsteps) // 4
    c.auxloss_use_ratio = min(1.0, 32.0 / c.nworkers)
    return c


if __name__ == "__main__":
    main()
