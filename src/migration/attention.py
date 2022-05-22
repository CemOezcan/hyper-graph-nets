from torch import nn
from torch_scatter import scatter_softmax

from src.util import device


class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_layer = nn.LazyLinear(1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.to(device)

    def forward(self, x, index):
        latent = self.linear_layer(x)
        latent = self.leaky_relu(latent)

        result = scatter_softmax(latent.float(), index, dim=0)
        result = result.type(result.dtype)
        return result
