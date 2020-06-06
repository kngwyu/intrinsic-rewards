"""Aliases for common types
"""
from typing import Callable

from torch import Tensor

from rainy.utils import Device, RunningMeanStdTorch

Normalizer = Callable[[Tensor, RunningMeanStdTorch], Tensor]
PreProcessor = Callable[[Tensor, Device], Tensor]
