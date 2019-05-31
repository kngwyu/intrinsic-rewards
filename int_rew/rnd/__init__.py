from rainy import Config
from .agent import RndPpoAgent
from .irew import irew_gen_deafult

from . import net


def default_config() -> Config:
    config = Config()
    setattr(config, 'adv_weight', 2.0)
    setattr(config, 'int_adv_weight', 1.0)
    setattr(config, 'int_discount_factor', 0.0)
    setattr(config, 'int_use_mask', False)
    setattr(config, 'int_value_loss_weight', 0.0)
    setattr(config, 'int_reward_gen', irew_gen_deafult())
    config.entropy_weight = 0.001
    config.adv_normalize_eps = None
    return config
