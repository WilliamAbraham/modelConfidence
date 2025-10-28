# MNIST Model Benchmarking

This repository contains a comprehensive benchmarking suite for different neural network architectures on the MNIST dataset.

## Available Models

The following models are implemented and ready for benchmarking:

1. **CNN** - Simple Convolutional Neural Network (baseline)
2. **MLP** - Multi-Layer Perceptron (fully connected)
3. **LeNet5** - Classic LeNet-5 architecture
4. **ResNet** - Residual Network with skip connections
5. **VGG** - VGG-style deep convolutional network
6. **DenseNet** - Densely connected convolutional network
7. **EfficientNet** - Simplified EfficientNet architecture
8. **Transformer** - Vision Transformer (ViT) for image classification

## Usage

### Benchmark a Single Model
```bash
python main.py --model cnn --epochs 10
```

### Benchmark All Models
```bash
python main.py --model all --epochs 10
```

### Command Line Options
- `--model`: Model to benchmark (default: all)
- `--device`: Device to use - cpu or cuda (default: cpu)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 1e-3)
- `--batch_size`: Batch size (default: 64)

### Examples
```bash
# Quick test with 2 epochs
python main.py --model resnet --epochs 2

# Full benchmark on GPU
python main.py --model all --device cuda --epochs 20

# Custom hyperparameters
python main.py --model transformer --epochs 15 --lr 5e-4 --batch_size 32
```

## Output

The benchmarking script generates:
- **Model files**: `mnist_{model_name}.pth` - Trained model weights
- **Results files**: `results_{model_name}.json` - Detailed training metrics
- **Summary file**: `benchmark_summary.json` - Comparison of all models

## Model Comparison

Each model is evaluated on:
- **Parameters**: Number of trainable parameters
- **Accuracy**: Final test accuracy
- **Training Time**: Total training time
- **Time per Epoch**: Average time per training epoch

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- numpy

## Installation

```bash
# Activate your conda environment
conda activate py310

# Install dependencies (if not already installed)
pip install torch torchvision
```
