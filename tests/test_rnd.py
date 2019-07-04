import rainy
from rainy.envs import Atari, DummyParallelEnv, atari_parallel
from rainy.envs.testing import DummyEnv
from rainy.lib.rollout import RolloutSampler
from rainy.net.policy import CategoricalDist
from rainy.utils import Device
import torch
from numpy.testing import assert_array_almost_equal

from int_rew import rnd


def config() -> rainy.Config:
    c = rnd.default_config()
    c.nworkers = 4
    c.nsteps = 4
    c.set_env(lambda: Atari('Venture', cfg=rnd.atari_config(), frame_stack=False))
    c.set_parallel_env(atari_parallel())
    return c


def test_save_and_load() -> None:
    agent = rnd.RndPpoAgent(config())
    agent.irew_gen.gen_rewards(torch.randn(4 * 4, 2, 84, 84))
    nonep = agent.irew_gen.rff_rms.mean.cpu().numpy()
    agent.save('agent.pth')
    agent.close()
    c = config()
    c.device = Device(use_cpu=True)
    agent = rnd.RndPpoAgent(c)
    agent.load('agent.pth')
    nonep_new = agent.irew_gen.rff_rms.mean.cpu().numpy()
    assert_array_almost_equal(nonep, nonep_new)
    agent.close()


def test_storage_and_irew() -> None:
    penv = DummyParallelEnv(lambda: DummyEnv(array_dim=(16, 16)), 6)
    NSTEPS = 4
    ACTION_DIM = 3
    NWORKERS = penv.num_envs
    states = penv.reset()
    storage = rnd.rollout.RndRolloutStorage(
        NSTEPS,
        NWORKERS,
        Device(),
        0.99,
        Device(use_cpu=True)
    )
    storage.set_initial_state(states)
    policy_head = CategoricalDist(ACTION_DIM)
    for _ in range(NSTEPS):
        state, reward, done, _ = penv.step([None] * NWORKERS)
        value = torch.randn(NWORKERS)
        pvalue = torch.randn(NWORKERS)
        policy = policy_head(torch.randn(NWORKERS).view(-1, 1))
        storage.push(state, reward, done, value=value, policy=policy)
        storage.push_int_value(pvalue)
    rewards = torch.randn(NWORKERS * NSTEPS)
    storage.calc_int_returns(torch.randn(NWORKERS), rewards, gamma=0.99, lambda_=0.95)
    batch = storage.batch_states(penv)
    batch_shape = torch.Size((NSTEPS * NWORKERS,))
    assert batch.shape == torch.Size((*batch_shape, 16, 16))
    MINIBATCH = 12
    sampler = rnd.rollout.RndRolloutSampler(
        RolloutSampler(storage, penv, MINIBATCH),
        storage,
        torch.randn(NSTEPS * NWORKERS),
        1.0,
        1.0
    )
    assert sampler.int_returns.shape == batch_shape
    assert sampler.int_values.shape == batch_shape
    assert sampler.advantages.shape == batch_shape
    for batch in sampler:
        assert len(batch.states) == MINIBATCH
    penv.close()
