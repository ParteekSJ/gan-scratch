import torch
from model.generator import Generator, create_noise
from constants import *
import ipdb
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Initializing empty generator
    generator = Generator().to(DEVICE)

    # Load generator checkpoint
    checkpoint = torch.load(PRETRAINED_PATH, map_location=torch.device("cpu"))
    generator.load_state_dict(checkpoint["gen_state_dict"])
    generator.eval()

    # ipdb.set_trace()

    # Generating noise
    noise = create_noise(batch_size=50, z_dim=Z_DIM)
    with torch.inference_mode():
        fake_samples = generator(noise)  # generating some fake samples

    fake_samples = fake_samples.reshape(-1, *IMG_SIZE)  # [B, 784] -> [B, 1, 28, 28]
    gen_img_grid = make_grid(fake_samples, nrow=10)

    # Create a figure and axis
    fig, ax = plt.subplots()

    ax.imshow(gen_img_grid.permute(1, 2, 0).squeeze())
    plt.title("Generated Images")
    plt.savefig(f"./gen_images.png")
    plt.close(fig)
