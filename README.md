# Intrinsic-Rewards
A collection of deep reinforcement learning algorithms with intrinsic rewards,
based on [Rainy](https://github.com/kngwyu/Rainy) and [PyTorch](https://pytorch.org/).

## Setup
First, install [pipenv](https://pipenv.readthedocs.io/en/latest/).
E.g. you can install it via
``` bash
pip install pipenv --user
```

Then you can create a virtual environment for isolated installing of related packages.
```bash
pipenv --site-packages --three install
```

## Implemented Algorithms

### Random Network Distillation
- https://arxiv.org/abs/1810.12894
- command: `pipenv run python experiments/rnd_atari.py`

## Results
Commit hash: aa4ebf0c3e9090d11fbd88a5de44aa2189f1d232

- RND
  - with 128 parallel enviroments + CNN policy(NO LSTM)
  - All parameters are the same as the paper
- PPO
  - with the same setting
  - All parameters are in [ppo_atari.py](experiments/ppo_atari.py)

![Venture](./pictures/venture.png)

![Montezuma's Revenge](./pictures/montezuma.png)
