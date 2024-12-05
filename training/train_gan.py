import os

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from losses.bce_loss import CustomBCELoss
from models.gan import Generator, Discriminator


def train_gan(
        train_loader,
        device,
        log_dir,
        checkpoint_dir,
        num_epochs,
        learning_rate,
        log_interval=5,
        latent_dim=128,
        img_shape=(3, 32, 32),
        fixed_noise_samples=16
):
    writer = SummaryWriter(log_dir=log_dir)
    generator = Generator(latent_dim=latent_dim, img_shape=img_shape).to(device)
    discriminator = Discriminator(img_shape=img_shape).to(device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    criterion = CustomBCELoss()

    fixed_z = torch.randn((fixed_noise_samples, latent_dim), device=device)

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        g_loss_avg, d_loss_avg = 0.0, 0.0

        for real_imgs, _ in tqdm(train_loader, desc=f"GAN Epoch {epoch + 1}", leave=False):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            valid = torch.ones((batch_size, 1), device=device)
            fake = torch.zeros((batch_size, 1), device=device)

            # Train Discriminator
            z = torch.randn((batch_size, latent_dim), device=device)
            fake_imgs = generator(z)

            d_optimizer.zero_grad()
            real_loss = criterion(discriminator(real_imgs), valid)
            fake_loss = criterion(discriminator(fake_imgs.detach()), fake)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()
            d_loss_avg += d_loss.item()

            # Train Generator
            g_optimizer.zero_grad()
            g_loss = criterion(discriminator(fake_imgs), valid)
            g_loss.backward()
            g_optimizer.step()
            g_loss_avg += g_loss.item()

        writer.add_scalar("GAN/Loss/Discriminator", d_loss_avg / len(train_loader), epoch + 1)
        writer.add_scalar("GAN/Loss/Generator", g_loss_avg / len(train_loader), epoch + 1)

        print(f"Epoch {epoch + 1}/{num_epochs}: G Loss = {g_loss_avg:.4f}, D Loss = {d_loss_avg:.4f}")

        if (epoch + 1) % log_interval == 0:
            generator.eval()
            with torch.no_grad():
                generated_imgs = generator(fixed_z).cpu()
            grid = torchvision.utils.make_grid(generated_imgs, nrow=4, normalize=True)
            writer.add_image(f"GAN/Generated Images", grid, epoch + 1)

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(generator.state_dict(), f"{checkpoint_dir}/generator_epoch_{epoch + 1}.pth")
            torch.save(discriminator.state_dict(), f"{checkpoint_dir}/discriminator_epoch_{epoch + 1}.pth")

        writer.close()
