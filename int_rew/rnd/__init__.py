import rainy
from .agent import RNDAgent
from .config import RNDConfig
from .irew import irew_gen_default, irew_gen_fc, UnsupervisedIRewGen
from .tuned_agent import TunedRNDAgent


def atari_config() -> rainy.envs.AtariConfig:
    c = rainy.envs.AtariConfig()
    c.override_timelimit = 4500 * 4
    c.noop_reset = False
    c.sticky_actions = True
    c.v4 = True
    return c
