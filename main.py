import torch
from constants import *
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from model.generator import Generator, create_noise
from model.discriminator import Discriminator
from utils.utils import show_tensor_images
from utils.utils import init_setting, plot_loss_curves
from logger.logger import setup_logging, get_logger
from torch import nn


if __name__ == "__main__":
    # Creating empty directories
    experiment_dir, checkpoint_dir, image_dir = init_setting()
    setup_logging(save_dir=experiment_dir)

    # Initializing the logger
    logger = get_logger(name="train")  # log message printing

    # Loading the MNIST dataset
    dataloader = DataLoader(
        dataset=MNIST(root="./dataset/", transform=transforms.ToTensor()),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # Defining Models and Optimizers
    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    gen_opt = torch.optim.Adam(params=generator.parameters(), lr=LR)
    disc_opt = torch.optim.Adam(params=discriminator.parameters(), lr=LR)

    # Initializing empty arrays to store G and D losses
    gen_epoch_losses = []
    disc_epoch_losses = []

    for epoch in range(N_EPOCHS):
        # Setting the models in train mode
        generator.train()
        discriminator.train()

        # for storing batch lossess
        gen_losses = []
        disc_losses = []

        for idx, (images, label) in enumerate(dataloader):
            curr_batch_size = images.shape[0]

            # Flatten the batch of real images
            images = nn.Flatten(start_dim=1, end_dim=-1)(images).to(DEVICE)

            ## Update Discriminator
            disc_opt.zero_grad()

            noise = create_noise(batch_size=curr_batch_size, z_dim=Z_DIM, device=DEVICE)
            fake_images = generator(noise)  # generated samples
            disc_fake_pred = discriminator(fake_images.detach())  # probability that the fake image is real
            disc_real_pred = discriminator(images)  # probability that the real image is real

            fake_loss = CRITERION(disc_fake_pred, torch.zeros_like(disc_fake_pred))
            real_loss = CRITERION(disc_real_pred, torch.ones_like(disc_real_pred))
            disc_loss = (fake_loss + real_loss) / 2

            disc_loss.backward(retain_graph=True)
            disc_losses.append(disc_loss.item())
            disc_opt.step()

            ## Update Generator
            gen_opt.zero_grad()
            fake_images_2 = generator(noise)

            disc_fake_pred_2 = discriminator(fake_images_2)
            gen_loss = CRITERION(disc_fake_pred_2, torch.ones_like(disc_fake_pred_2))

            gen_loss.backward(retain_graph=True)
            gen_losses.append(gen_loss.item())
            gen_opt.step()

            if idx % DISPLAY_STEP == 0 and idx > 0:
                avg_gen_loss = sum(gen_losses[-DISPLAY_STEP:]) / DISPLAY_STEP
                avg_disc_loss = sum(disc_losses[-DISPLAY_STEP:]) / DISPLAY_STEP

                print(f"EPOCH: {epoch} | STEP: {idx} | G_LOSS: {avg_gen_loss} | D_LOSS: {avg_disc_loss}")

        epoch_gen_loss = sum(gen_losses[:]) / len(dataloader)
        gen_epoch_losses.append(epoch_gen_loss)

        epoch_disc_loss = sum(disc_losses[:]) / len(dataloader)
        disc_epoch_losses.append(epoch_disc_loss)

        # Saving model and optimizer state
        checkpoint = {
            "epoch": epoch,
            "gen_state_dict": generator.state_dict(),
            "disc_state_dict": discriminator.state_dict(),
            "gen_optimizer": gen_opt.state_dict(),
            "disc_optimizer": disc_opt.state_dict(),
        }
        torch.save(checkpoint, f"./{checkpoint_dir}/gan_mnist.pth")
        logger.info(f"\nEPOCH: {epoch} | G_LOSS: {epoch_gen_loss} | D_LOSS: {epoch_disc_loss}\n")

        # Plotting a grid of images every epoch
        with torch.inference_mode():
            fake = generator(noise)
        show_tensor_images(fake, show=False, plot_name=f"{image_dir}/E-{epoch}-F.png")
        show_tensor_images(images, show=False, plot_name=f"{image_dir}/E-{epoch}-R.png")

    # Plotting the GAN loss curves
    plot_loss_curves(gen_epoch_losses, disc_epoch_losses, checkpoint_dir)
