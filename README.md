
# GenAI HW 1

This repo provides a framework in the scope of HW1 (GenAI) for developing, training, and testing  generative models with custom loss functions and metrics. 

## Features

### Loss Functions
Custom implementations of loss functions:
- **Binary Cross Entropy Loss** (`bce_loss.py`)
- **Mean Squared Error Loss** (`custom_mse_loss.py`)
- **Variational Autoencoder Loss** (`vae_loss.py`)
- **Negative Log-Likelihood Loss** (`nll_loss.py`)

### Metrics
Utility functions for evaluating model performance:
- Custom metric implementations (`custom_metrics.py`)

### Models
Predefined models to streamline workflow:
- **Variational Autoencoder (VAE)** (`vae.py`)
- **Autoencoder** (`autoencoder.py`)
- **Generative Adversarial Network (GAN)** (`gan.py`)
- **Normalizing Flows** (`normalizing_flows.py`)

### Training Pipelines
Training scripts and utilities:
- **Training VAE** (`train_vae.py`)
- **Training GAN** (`train_gan.py`)
- **Training Normalizing Flow Models** (`train_normalizing_flow.py`)
- **Training Autoencoder** (`train_autoencoder.py`)
- Unified training utility (`trainer.py`)

### Utilities
- **Dataset Management** (`dataset.py`): Tools for loading and preprocessing datasets.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/your_repo_name.git
   cd your_repo_name
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   The main dependencies include:
   - PyTorch and torchvision
   - NumPy
   - tqdm
   - matplotlib
   - piq (Perceptual Image Quality metrics)

## Usage

### Training a Model
Each model has a corresponding training script. For example, to train a Variational Autoencoder:
```bash
python train_vae.py
```

You can customize the training configurations by modifying the `main.py` file or passing arguments in the command line.

### Testing
Test scripts are provided to evaluate model performance. For example:
```bash
python test_autoencoder.py
```

### Customization
You can extend the repository by adding new loss functions, metrics, or models:
- Implement a new loss function in the style of `bce_loss.py`.
- Add a new model architecture under the `models` folder.
- Create a new training script or integrate with the unified `trainer.py`.

## Repository Structure
```
├── losses
│   ├── bce_loss.py
│   ├── custom_mse_loss.py
│   ├── vae_loss.py
│   └── nll_loss.py
├── metrics
│   └── custom_metrics.py
├── models
│   ├── autoencoder.py
│   ├── gan.py
│   ├── vae.py
│   └── normalizing_flows.py
├── training
│   ├── train_vae.py
│   ├── train_gan.py
│   ├── train_normalizing_flow.py
│   ├── train_autoencoder.py
│   └── trainer.py
├── data
│   └── dataset.py
├── testing
│   ├── test_autoencoder.py
│   ├── test_custom_loss.py
│   ├── test_dataset.py
│   ├── test_gan.py
│   ├── test_metrics.py
│   ├── test_vae.py
│   └── test_vae_loss.py
├── main.py
├── requirements.txt
└── README.md
```
