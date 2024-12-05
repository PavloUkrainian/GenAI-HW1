import torch


class BaseConfig:
    num_epochs = 10
    batch_size = 64
    learning_rate = 1e-3

    log_dir = "./logs"
    checkpoint_dir = "./checkpoints"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    latent_dim = 128  # For VAE and GAN
    input_dim = 3 * 32 * 32  # For Normalizing Flow
    hidden_dim = 256  # For Normalizing Flow


class AutoencoderConfig(BaseConfig):
    log_dir = f"{BaseConfig.log_dir}/autoencoder"
    checkpoint_dir = f"{BaseConfig.checkpoint_dir}/autoencoder"


class VAEConfig(BaseConfig):
    log_dir = f"{BaseConfig.log_dir}/vae"
    checkpoint_dir = f"{BaseConfig.checkpoint_dir}/vae"
    latent_dim = 128  # Specific to VAE


class GANConfig(BaseConfig):
    log_dir = f"{BaseConfig.log_dir}/gan"
    checkpoint_dir = f"{BaseConfig.checkpoint_dir}/gan"
    latent_dim = 128  # Latent space dimension


class NormalizingFlowConfig(BaseConfig):
    log_dir = f"{BaseConfig.log_dir}/normalizing_flow"
    checkpoint_dir = f"{BaseConfig.checkpoint_dir}/normalizing_flow"
    input_dim = 3 * 32 * 32  # Input dimension
    hidden_dim = 256  # Hidden dimension

class BetaVAEConfig(BaseConfig):
    beta = 5.0  # Specific for Beta-VAE

