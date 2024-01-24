import sys

sys.path.append("../")
import torch
from torch import nn
from torchinfo import summary
from constants import *


class Discriminator(nn.Module):
    def __init__(self, im_dim: int = IM_DIM, hidden_dim: int = HIDDEN_DIM):
        super(Discriminator, self).__init__()

        # Generated/Real Image gets fed to the Discrimintor, i.e., 784(Image Dim) -> Discriminator
        self.disc = nn.Sequential(
            self.get_discriminator_block(in_features=im_dim, out_features=hidden_dim * 4),
            self.get_discriminator_block(in_features=hidden_dim * 4, out_features=hidden_dim * 2),
            self.get_discriminator_block(in_features=hidden_dim * 2, out_features=hidden_dim),
            nn.Linear(in_features=hidden_dim, out_features=1),
        )

    def get_discriminator_block(self, in_features: int, out_features: int):
        return nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image):
        return self.disc(image)


if __name__ == "__main__":
    model = Discriminator()
    print(model)
    input = torch.randn(16, 1, 28, 28)
    output = model(nn.Flatten(start_dim=1, end_dim=-1)(input))  # [16, 1]
    print(output.shape)
    print(summary(model, input_data=nn.Flatten(start_dim=1, end_dim=-1)(input)))

    """
    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    Discriminator                            [16, 1]                   --
    ├─Sequential: 1-1                        [16, 1]                   --
    │    └─Sequential: 2-1                   [16, 512]                 --
    │    │    └─Linear: 3-1                  [16, 512]                 401,920
    │    │    └─LeakyReLU: 3-2               [16, 512]                 --
    │    └─Sequential: 2-2                   [16, 256]                 --
    │    │    └─Linear: 3-3                  [16, 256]                 131,328
    │    │    └─LeakyReLU: 3-4               [16, 256]                 --
    │    └─Sequential: 2-3                   [16, 128]                 --
    │    │    └─Linear: 3-5                  [16, 128]                 32,896
    │    │    └─LeakyReLU: 3-6               [16, 128]                 --
    │    └─Linear: 2-4                       [16, 1]                   129
    ==========================================================================================
    """
