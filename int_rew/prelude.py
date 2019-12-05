"""Aliases for common types
"""
from rainy.utils import Device, RunningMeanStdTorch
from torch import Tensor
from typing import Callable


Normalizer = Callable[[Tensor, RunningMeanStdTorch], Tensor]
PreProcessor = Callable[[Tensor, Device], Tensor]
