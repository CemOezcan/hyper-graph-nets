import numpy as np
import torch

def detach(tensor: torch.Tensor) -> np.array:

    if tensor.is_cuda:
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()
