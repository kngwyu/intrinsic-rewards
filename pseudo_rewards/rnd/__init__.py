from rainy import Config


def default_config() -> Config:
    config = Config()
    setattr(config, 'ext_adv_coeff', 1.0)
    setattr(config, 'int_adv_coeff', 0.0)
    config.adv_normalize_eps = None
    return config
