from rainy import Config

from .prew import prew_gen_deafult


def default_config() -> Config:
    config = Config()
    setattr(config, 'adv_weight', 1.0)
    setattr(config, 'pseudo_adv_weight', 0.0)
    setattr(config, 'pseudo_discount_factor', 0.0)
    setattr(config, 'pseudo_value_loss_weight', 0.0)
    setattr(config, 'pseudo_reward_gen', prew_gen_deafult)
    config.entropy_weight = 0.001
    config.adv_normalize_eps = None
    return config
