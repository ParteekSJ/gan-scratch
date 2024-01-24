import torch
from model.generator import Generator, create_noise
from constants import *
import ipdb

if __name__ == "__main__":
    # Initializing empty generator
    generator = Generator().to(DEVICE)

    # Load generator checkpoint
    checkpoint = torch.load(PRETRAINED_PATH, map_location=torch.device("cpu"))
    generator.load_state_dict(checkpoint["gen_state_dict"])
    generator.eval()

    # Generating noise
    noise = create_noise(batch_size=25, z_dim=Z_DIM)
    with torch.inference_mode():
        fake_samples = generator(noise)  # generating some fake samples
