import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        x_reconstructed = torch.sigmoid(self.fc2(z))
        return x_reconstructed


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, beta=1.0):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.beta = beta

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, logvar

    def loss_function(self, x, x_reconstructed, mu, logvar):
        reconstruction_loss = F.binary_cross_entropy(x_reconstructed, x, reduction="sum")
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + self.beta * kl_divergence, reconstruction_loss, kl_divergence


def load_mnist(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_vae(model, train_loader, epochs, learning_rate, device):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    losses, recon_losses, kl_divs = [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, recon_loss, kl_loss = 0, 0, 0

        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            x = x.to(device)
            x_reconstructed, mu, logvar = model(x)
            loss, recon, kl = model.loss_function(x, x_reconstructed, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            recon_loss += recon.item()
            kl_loss += kl.item()

        losses.append(train_loss / len(train_loader.dataset))
        recon_losses.append(recon_loss / len(train_loader.dataset))
        kl_divs.append(kl_loss / len(train_loader.dataset))

        print(f"Epoch {epoch}: Loss={losses[-1]:.4f}, Recon={recon_losses[-1]:.4f}, KL={kl_divs[-1]:.4f}")

    return losses, recon_losses, kl_divs


def traverse_latent_space(model, device, latent_dim, n_samples=10, range_val=5):
    model.eval()
    z = torch.zeros(latent_dim).to(device)
    images = []

    for dim in range(latent_dim):
        traverse_images = []
        for val in torch.linspace(-range_val, range_val, steps=n_samples):
            z[dim] = val
            reconstructed = model.decoder(z).view(28, 28).cpu().detach().numpy()
            traverse_images.append(reconstructed)
        images.append(traverse_images)
        z[dim] = 0

    return images


# Visualization: Comparison of Loss and Latent Space Traversal
def visualize_comparison(beta_vae_data, vae_data, beta_vae_images, vae_images, latent_dim):
    epochs = np.arange(1, len(beta_vae_data[0]) + 1)

    plt.figure(figsize=(12, 6))

    # Reconstruction Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, beta_vae_data[1], label="Beta-VAE Reconstruction", marker="o")
    plt.plot(epochs, vae_data[1], label="VAE Reconstruction", marker="x")
    plt.title("Reconstruction Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # KL Divergence
    plt.subplot(1, 2, 2)
    plt.plot(epochs, beta_vae_data[2], label="Beta-VAE KL Divergence", marker="o")
    plt.plot(epochs, vae_data[2], label="VAE KL Divergence", marker="x")
    plt.title("KL Divergence")
    plt.xlabel("Epochs")
    plt.ylabel("KL Divergence")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Latent Space Traversal
    fig, axs = plt.subplots(2, latent_dim, figsize=(15, 6))
    fig.suptitle("Latent Space Traversal: Beta-VAE vs. VAE", fontsize=16)

    for dim in range(latent_dim):
        for idx, img in enumerate(beta_vae_images[dim]):
            axs[0, dim].imshow(img, cmap="gray")
            axs[0, dim].axis("off")
        axs[0, dim].set_title(f"Latent {dim + 1} (Beta-VAE)")

        for idx, img in enumerate(vae_images[dim]):
            axs[1, dim].imshow(img, cmap="gray")
            axs[1, dim].axis("off")
        axs[1, dim].set_title(f"Latent {dim + 1} (VAE)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    input_dim = 28 * 28
    hidden_dim = 500
    latent_dim = 10
    beta = 5.0
    batch_size = 100
    epochs = 20
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _ = load_mnist(batch_size)

    # Train Beta-VAE
    beta_vae = VAE(input_dim, hidden_dim, latent_dim, beta=beta)
    beta_vae_data = train_vae(beta_vae, train_loader, epochs, learning_rate, device)
    beta_vae_images = traverse_latent_space(beta_vae, device, latent_dim)

    # Train Standard VAE
    vae = VAE(input_dim, hidden_dim, latent_dim, beta=1.0)
    vae_data = train_vae(vae, train_loader, epochs, learning_rate, device)
    vae_images = traverse_latent_space(vae, device, latent_dim)

    # Visualize Comparison
    visualize_comparison(beta_vae_data, vae_data, beta_vae_images, vae_images, latent_dim)
