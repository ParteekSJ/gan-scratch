from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import sys
import ipdb

sys.path.append("../")
from constants import *


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), show=False, plot_name=""):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    """

    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)

    # Create a figure and axis
    fig, ax = plt.subplots()

    ax.imshow(image_grid.permute(1, 2, 0).squeeze())
    if show:
        plt.show()

    plt.savefig(f"{plot_name}")
    plt.close(fig)


def plot_loss_curves(gen_epoch_losses, disc_epoch_losses, directory):
    plt.plot(gen_epoch_losses, label="Generator Loss")
    plt.plot(disc_epoch_losses, label="Discriminator Loss")
    plt.title("Generator & Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{directory}/loss.svg", format="svg")


def init_setting():
    timestr = str(datetime.datetime.now().strftime("%Y-%m%d_%H%M"))
    experiment_dir = Path(LOG_PATH)
    experiment_dir.mkdir(exist_ok=True)  # directory for saving experimental results
    experiment_dir = experiment_dir.joinpath(timestr)
    experiment_dir.mkdir(exist_ok=True)  # root directory of each experiment

    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_dir = checkpoint_dir.joinpath(timestr)
    checkpoint_dir.mkdir(exist_ok=True)  # root directory of each checkpoint

    image_dir = Path(IMAGE_DIR)
    image_dir.mkdir(exist_ok=True)
    image_dir = image_dir.joinpath(timestr)
    image_dir.mkdir(exist_ok=True)  # root directory of each image (generated and real)

    # returns several directory paths
    return experiment_dir, checkpoint_dir, image_dir
