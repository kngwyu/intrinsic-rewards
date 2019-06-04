import rainy
from .agent import RndPpoAgent
from .irew import irew_gen_deafult
from .net import rnd_ac_conv


def default_config() -> rainy.Config:
    config = rainy.Config()
    setattr(config, 'adv_weight', 2.0)
    setattr(config, 'int_adv_weight', 1.0)
    setattr(config, 'int_discount_factor', 0.99)
    setattr(config, 'int_use_mask', False)
    setattr(config, 'int_reward_gen', irew_gen_deafult())
    setattr(config, 'auxloss_use_ratio', 0.50)
    config.discount_factor = 0.999
    config.entropy_weight = 0.001
    config.adv_normalize_eps = None
    config.ppo_epochs = 4
    config.ppo_clip = 0.1
    config.use_gae = True
    config.set_net_fn('actor-critic', rnd_ac_conv())
    return config


def atari_config() -> rainy.envs.AtariConfig:
    c = rainy.envs.AtariConfig()
    c.override_timelimit = 4500 * 4
    c.noop_reset = False
    c.sticky_actions = True
    c.v4 = True
    c.frame_stack = False
    return c
