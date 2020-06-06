from typing import Callable, List, Tuple

from torch import nn


def construct_body(
    net_fn: Callable[[int], Tuple[nn.Module, nn.Module]], i_max: int
) -> List[nn.Module]:
    """Get Net1 + actitivator + Net2 + activator + ...
    """
    res = []
    for i in range(i_max):
        net, activ = net_fn(i)
        res.append(net)
        res.append(activ)
    return res


def sequential_body(
    net: Callable[[int], Tuple[nn.Module, nn.Module]], i_max: int
) -> nn.Sequential:
    modules = construct_body(net, i_max)
    return nn.Sequential(*modules)
