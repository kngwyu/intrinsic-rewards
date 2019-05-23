from rainy.envs import DummyParallelEnv
from rainy.envs.testing import DummyEnv
from rainy.lib.rollout import RolloutStorage
from rainy.net.policy import CategoricalHead
from rainy.utils import Device
import torch

from pseudo_rewards import rnd


def test_storage() -> None:
    penv = DummyParallelEnv(lambda: DummyEnv(array_dim=(16, 16)), 6)
    NSTEP = 4
    ACTION_DIM = 3
    NWORKERS = penv.num_envs
    states = penv.reset()
    dev = Device()
    storage = rnd.rollout.RndRolloutStorage(
        RolloutStorage(NSTEP, NWORKERS, dev),
        NSTEP,
        NWORKERS,
        0.99,
        Device(use_cpu=True)
    )
    storage.set_initial_state(states)
    policy_head = CategoricalHead(ACTION_DIM)
    for _ in range(NSTEP):
        state, reward, done, _ = penv.step([None] * NWORKERS)
        value = torch.randn(NWORKERS)
        prew = torch.randn(NWORKERS)
        pvalue = torch.randn(NWORKERS)
        policy = policy_head(torch.randn(NWORKERS).view(-1, 1))
        storage.push(state, reward, done, value=value, policy=policy)
        storage.push_pseudo_rewards(pvalue, prew)
    storage.calc_pseudo_returns(torch.randn(NWORKERS), gamma=0.99, lambda_=0.95)
    batch = storage.batch_states(penv)
    batch_shape = torch.Size((NSTEP * NWORKERS,))
    assert batch.shape == torch.Size((*batch_shape, 16, 16))
    MINIBATCH = 12
    sampler = rnd.rollout.RndRolloutSampler(storage, penv, MINIBATCH, 1.0, 1.0)
    assert sampler.pseudo_returns.shape == batch_shape
    assert sampler.pseudo_values.shape == batch_shape
    assert sampler.advantages.shape == batch_shape
    for batch in sampler:
        assert len(batch.states) == MINIBATCH
    penv.close()
