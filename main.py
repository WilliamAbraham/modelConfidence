import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import time
import json
from pathlib import Path
import os

from model import get_model, MODELS

# Default hyperparameters
lr = 1e-3
log_interval = 10
epochs = 500
batch_size = 64
data_root = '/scratch/jc14407/datasets'

def get_dataset_config(dataset_name):
    """
    Get configuration for different torchvision datasets.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'MNIST', 'CIFAR10', 'CIFAR100')
    
    Returns:
        dict with keys: input_channels, num_classes, image_size, dataset_class
    """
    dataset_name = dataset_name.upper()
    
    configs = {
        'MNIST': {
            'input_channels': 1,
            'num_classes': 10,
            'image_size': (28, 28),
            'dataset_class': datasets.MNIST
        },
        'CIFAR10': {
            'input_channels': 3,
            'num_classes': 10,
            'image_size': (32, 32),
            'dataset_class': datasets.CIFAR10
        },
        'CIFAR100': {
            'input_channels': 3,
            'num_classes': 100,
            'image_size': (32, 32),
            'dataset_class': datasets.CIFAR100
        },
        'FASHIONMNIST': {
            'input_channels': 1,
            'num_classes': 10,
            'image_size': (28, 28),
            'dataset_class': datasets.FashionMNIST
        }
    }
    
    if dataset_name not in configs:
        raise ValueError(f"Dataset {dataset_name} not supported. Available: {list(configs.keys())}")
    
    return configs[dataset_name] 

def train(model, device, train_loader, optimizer, epoch, verbose=True):
    model.train()
    total_class_loss = 0
    total_epi_error = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, epi_error = model(data)
        class_loss = F.cross_entropy(output, target)
        var_loss = torch.mean(epi_error)
        loss = class_loss + var_loss
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total_class_loss += class_loss.item()
        total_epi_error += var_loss.item()
        
        if verbose and batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    accuracy = 100. * correct / len(train_loader.dataset)
    total_class_loss /= len(train_loader.dataset)
    total_epi_error /= len(train_loader)
    
    return total_class_loss,accuracy, total_epi_error
    
def test(model, device, test_loader, verbose=True):
    model.eval()
    total_class_loss = 0
    correct = 0
    total_epi_error = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, epi_error = model(data)
            total_class_loss += F.cross_entropy(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_epi_error += torch.mean(epi_error).item()
    
    total_class_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    total_epi_error /= len(test_loader)
    
    if verbose:
        print(f'\nTest set: Average loss: {total_class_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
              f' ({accuracy:.0f}%)\n')
    
    return total_class_loss, accuracy, total_epi_error

def benchmark_model(model_name, device='cpu', epochs=10, lr=1e-3, batch_size=64, save_results=False, dataset='MNIST', file_path):
    """Benchmark a single model with fixed number of epochs
    
    Args:
        model_name: Name of the model to benchmark
        device: Device to use ('cpu' or 'cuda')
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        save_results: Whether to save results to JSON
        dataset: Dataset name (default: 'MNIST'). Must be a torchvision dataset.
                 Supported: 'MNIST', 'CIFAR10', 'CIFAR100', 'FashionMNIST'
    """
    print(f"\n{'='*50}")
    print(f"Benchmarking {model_name.upper()} on {dataset.upper()}")
    print(f"{'='*50}")
    
    # Get dataset configuration
    dataset_config = get_dataset_config(dataset)
    dataset_class = dataset_config['dataset_class']
    input_channels = dataset_config['input_channels']
    num_classes = dataset_config['num_classes']
    image_size = dataset_config['image_size']
    
    # Load data
    train_dataset = dataset_class(root=data_root, train=True, download=True,
                                  transform=transforms.ToTensor())
    train_dataset = torch.utils.data.Subset(train_dataset, range(100))  # Use a subset for faster benchmarking
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = dataset_class(root=data_root, train=False, transform=transforms.ToTensor())
    #test_dataset = torch.utils.data.Subset(test_dataset, range(1024))  # Use a subset for faster benchmarking
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model - pass image_size for EAFNO/EACNN models
    model_kwargs = {}
    if model_name in ['eafno', 'eacnn']:
        model_kwargs['image_size'] = image_size
    
    model = get_model(model_name, input_channels=input_channels, output_channels=num_classes, **model_kwargs).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    
    print(f"Model: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    
    # Training with fixed number of epochs
    start_time = time.time()
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_accuracy, train_epi_error = train(model, device, train_loader, optimizer, epoch, verbose=False)
        test_loss, test_acc, test_epi_error = test(model, device, test_loader, verbose=False)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch:2d}/{epochs}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Train Epi Error: {train_epi_error:.6f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Test Epi Error: {test_epi_error:.6f}, Time: {epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    
    # Final results
    final_test_acc = test_accuracies[-1]
    print(f"\nFinal Results:")
    print(f"Test Accuracy: {final_test_acc:.2f}%")
    print(f"Training Time: {total_time:.2f}s")
    print(f"Time per Epoch: {total_time/epochs:.2f}s")
    
    # Save model
    #models_dir = Path("/scratch/jc14407/modelConfidence/checkpoints")
    models_dir = file_path
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    # Include dataset name in model path
    model_path = os.path.join(models_dir, f"{dataset.lower()}_{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    # torch.save(model, model_path)
    
    # Prepare results
    results = {
        'model_name': model_name,
        'dataset': dataset,
        'parameters': sum(p.numel() for p in model.parameters()),
        'epochs': epochs,
        'learning_rate': lr,
        'batch_size': batch_size,
        'device': device,
        'training_time': total_time,
        'time_per_epoch': total_time / epochs if epochs > 0 else 0,
        'final_test_accuracy': final_test_acc,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'model_path': str(model_path)
    }
    
    if save_results:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        results_path = results_dir / f"results_{dataset.lower()}_{model_name}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")
    
    return results

def benchmark_all_models(device='cpu', epochs=10, lr=1e-3, batch_size=64, dataset='MNIST', file_path):
    """Benchmark all available models with fixed number of epochs
    
    Args:
        device: Device to use ('cpu' or 'cuda')
        epochs: Number of training epochs per model
        lr: Learning rate
        batch_size: Batch size for training
        dataset: Dataset name (default: 'MNIST')
    """
    print(f"{dataset.upper()} Model Benchmarking")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Epochs per model: {epochs}")
    print(f"Learning Rate: {lr}")
    print(f"Batch Size: {batch_size}")
    print(f"Dataset: {dataset}")
    print(f"Available Models: {list(MODELS.keys())}")
    
    all_results = {}
    
    for model_name in MODELS.keys():
        try:
            results = benchmark_model(model_name, device, epochs, lr, batch_size, save_results=False, dataset=dataset, file_path=file_path)
            all_results[model_name] = results
        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            all_results[model_name] = {'error': str(e)}
    
    # Summary
    print(f"\n{'='*50}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*50}")
    print(f"{'Model':<15} {'Parameters':<12} {'Accuracy':<10} {'Time (s)':<10}")
    print("-" * 50)
    
    for model_name, results in all_results.items():
        if 'error' not in results:
            print(f"{model_name:<15} {results['parameters']:<12,} {results['final_test_accuracy']:<10.2f} {results['training_time']:<10.2f}")
        else:
            print(f"{model_name:<15} {'ERROR':<12} {'N/A':<10} {'N/A':<10}")
    
    # Save summary
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    summary_path = results_dir / "benchmark_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")
    
    return all_results

# Global configuration variables - modify these as needed
model = 'eacnn'  # Model to benchmark: 'all' or one of ['cnn', 'mlp', 'lenet5', 'resnet', 'vgg', 'densenet', 'efficientnet', 'transformer','eafno', 'eacnn']
device = 'cuda'  # Device to use: 'cpu' or 'cuda'
file_path = '/scratch/wja6857/modelConfidence/checkpoints'
# Note: epochs is already defined above in the default hyperparameters section
# Note: dataset can be changed in the main() function below (default: 'MNIST', also supports 'CIFAR10', 'CIFAR100', 'FashionMNIST')

def main():
    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        actual_device = 'cpu'
    else:
        actual_device = device
    
    # Default dataset - can be changed here or passed as parameter
    dataset = 'CIFAR10'
    
    if model == 'all':
        benchmark_all_models(actual_device, epochs, lr, batch_size, dataset=dataset, file_path=file_path)
    else:
        benchmark_model(model, actual_device, epochs, lr, batch_size, dataset=dataset, file_path=file_path)

if __name__ == '__main__':
    main()