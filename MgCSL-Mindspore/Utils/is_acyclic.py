import numpy as np
from mindspore import Tensor

def is_acyclic(adjacency, device=None):
    """
    Whether the adjacency matrix is a acyclic graph.
    """
    prod = np.eye(adjacency.shape[0])
    adjacency, prod = transfer_to_device(adjacency, prod, device=device)
    for _ in range(1, adjacency.shape[0] + 1):
        prod = Tensor.matmul(adjacency, prod)
        if Tensor.trace(prod) != 0:
            return False
    return True

def transfer_to_device(*args, device=None):
    """
    Transfer `*args` to `device`

    Parameters
    ----------
    args: np.ndarray, torch.Tensor
        variables that need to transfer to `device`
    device: str
        'cpu' or 'gpu', if None, default='cpu

    Returns
    -------
    out: args
    """

    out = []
    for each in args:
        if isinstance(each, np.ndarray):
            each = Tensor(each)
        elif isinstance(each, Tensor):
            each = each
        else:
            raise TypeError(f"Expected type of the args is np.ndarray "
                            f"or torch.Tensor, but got `{type(each)}`.")
        out.append(each)
    if len(out) > 1:
        return tuple(out)
    else:
        return out[0]