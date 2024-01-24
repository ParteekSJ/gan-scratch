import sys

sys.path.append("../")
import torch
from torch import nn
import ipdb
from torchinfo import summary
from constants import *


class Generator(nn.Module):
    def __init__(self, z_dim: int = Z_DIM, im_dim: int = IM_DIM, hidden_dim: int = HIDDEN_DIM):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            self.get_generator_block(in_features=z_dim, out_features=hidden_dim),
            self.get_generator_block(in_features=hidden_dim, out_features=hidden_dim * 2),
            self.get_generator_block(in_features=hidden_dim * 2, out_features=hidden_dim * 4),
            self.get_generator_block(in_features=hidden_dim * 4, out_features=hidden_dim * 8),
            nn.Linear(in_features=hidden_dim * 8, out_features=im_dim),
            nn.Sigmoid(),
        )

    def get_generator_block(self, in_features: int, out_features: int):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, noise):
        return self.gen(noise)


def create_noise(batch_size: int, z_dim: int, device: str = "cpu"):
    return torch.randn(batch_size, z_dim).to(device)


if __name__ == "__main__":
    model = Generator()
    print(model)
    ipdb.set_trace()
    noise = create_noise(batch_size=16, z_dim=64)
    output = model(noise)
    print(summary(model, input_data=noise))

    """
    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    Generator                                [16, 784]                 --
    ├─Sequential: 1-1                        [16, 784]                 --
    │    └─Sequential: 2-1                   [16, 128]                 --
    │    │    └─Linear: 3-1                  [16, 128]                 8,320
    │    │    └─BatchNorm1d: 3-2             [16, 128]                 256
    │    │    └─ReLU: 3-3                    [16, 128]                 --
    │    └─Sequential: 2-2                   [16, 256]                 --
    │    │    └─Linear: 3-4                  [16, 256]                 33,024
    │    │    └─BatchNorm1d: 3-5             [16, 256]                 512
    │    │    └─ReLU: 3-6                    [16, 256]                 --
    │    └─Sequential: 2-3                   [16, 512]                 --
    │    │    └─Linear: 3-7                  [16, 512]                 131,584
    │    │    └─BatchNorm1d: 3-8             [16, 512]                 1,024
    │    │    └─ReLU: 3-9                    [16, 512]                 --
    │    └─Sequential: 2-4                   [16, 1024]                --
    │    │    └─Linear: 3-10                 [16, 1024]                525,312
    │    │    └─BatchNorm1d: 3-11            [16, 1024]                2,048
    │    │    └─ReLU: 3-12                   [16, 1024]                --
    │    └─Linear: 2-5                       [16, 784]                 803,600
    │    └─Sigmoid: 2-6                      [16, 784]                 --
    ==========================================================================================
    """
