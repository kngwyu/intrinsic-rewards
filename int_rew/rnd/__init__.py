import rainy
from .agent import RndPpoAgent
from .config import RndConfig
from .irew import irew_gen_default, UnsupervisedIRewGen
from .net import RndACNet


def atari_config() -> rainy.envs.AtariConfig:
    c = rainy.envs.AtariConfig()
    c.override_timelimit = 4500 * 4
    c.noop_reset = False
    c.sticky_actions = True
    c.v4 = True
    return c
