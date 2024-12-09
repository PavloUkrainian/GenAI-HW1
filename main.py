from configs.default_config import AutoencoderConfig, GANConfig, NormalizingFlowConfig, VAEConfig
from training.train_autoencoder import train_autoencoder
from training.train_gan import train_gan
from training.train_normalizing_flow import train_normalizing_flow
from training.train_vae import train_vae
from utils.dataset import get_cifar10_loaders

MODEL_CONFIG_MAP = {
    "autoencoder": (AutoencoderConfig, train_autoencoder),
    "vae": (VAEConfig, train_vae),
    "gan": (GANConfig, train_gan),
    "normalizing_flow": (NormalizingFlowConfig, train_normalizing_flow),
}


def main():
    config = AutoencoderConfig()
    train_loader, val_loader, _ = get_cifar10_loaders(batch_size=config.batch_size)

    for model_name, (config_class, train_function) in MODEL_CONFIG_MAP.items():
        config = config_class()

        print(f"Training {model_name.capitalize()}...")
        if model_name == "gan":
            # For GAN
            train_function(
                train_loader=train_loader,
                device=config.device,
                log_dir=config.log_dir,
                checkpoint_dir=config.checkpoint_dir,
                num_epochs=config.num_epochs,
                learning_rate=config.learning_rate,
            )
        else:
            train_function(
                train_loader=train_loader,
                val_loader=val_loader,
                device=config.device,
                log_dir=config.log_dir,
                checkpoint_dir=config.checkpoint_dir,
                num_epochs=config.num_epochs,
                learning_rate=config.learning_rate,
            )
        print(f"Training for {model_name.capitalize()} completed.\n")


if __name__ == "__main__":
    main()
