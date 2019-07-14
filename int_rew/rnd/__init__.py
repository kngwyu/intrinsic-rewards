import rainy
from .agent import RndPpoAgent
from .irew import irew_gen_deafult
from .net import rnd_ac_conv


class RndConfig(rainy.Config):
    def __init__(self) -> None:
        super().__init__()
        # Override hyper parameters
        self.discount_factor = 0.999
        self.entropy_weight = 0.001
        self.adv_normalize_eps = None
        self.ppo_epochs = 4
        self.ppo_clip = 0.1
        self.use_gae = True
        self.set_net_fn('actor-critic', rnd_ac_conv())
        # RND specific parameters
        self.adv_weight = 2.0
        self.int_adv_weight = 1.0
        self.int_discount_factor = 0.99
        self.int_use_mask = False
        self.int_reward_gen = irew_gen_deafult()
        self.auxloss_use_ratio = 0.5
        self.intrew_log_freq = 1000
        self.initialize_stats = 50


def atari_config() -> rainy.envs.AtariConfig:
    c = rainy.envs.AtariConfig()
    c.override_timelimit = 4500 * 4
    c.noop_reset = False
    c.sticky_actions = True
    c.v4 = True
    return c
