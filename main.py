import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import time
import json
from pathlib import Path

from model import get_model, MODELS

# Default hyperparameters
lr = 1e-3
log_interval = 10
epochs = 50
batch_size = 64

def train(model, device, train_loader, optimizer, epoch, verbose=True):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if verbose and batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    return total_loss / len(train_loader)
    
def test(model, device, test_loader, verbose=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    if verbose:
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
              f' ({accuracy:.0f}%)\n')
    
    return test_loss, accuracy

def benchmark_model(model_name, device='cpu', epochs=10, lr=1e-3, batch_size=64, save_results=True):
    """Benchmark a single model with fixed number of epochs"""
    print(f"\n{'='*50}")
    print(f"Benchmarking {model_name.upper()}")
    print(f"{'='*50}")
    
    # Load data
    train_dataset = datasets.MNIST('datasets', train=True, download=True,
                                  transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = datasets.MNIST('datasets', train=False, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = get_model(model_name, input_channels=1, output_channels=10).to(device)
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
        
        train_loss = train(model, device, train_loader, optimizer, epoch, verbose=False)
        test_loss, test_acc = test(model, device, test_loader, verbose=False)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch:2d}/{epochs}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Time: {epoch_time:.1f}s")
    
    total_time = time.time() - start_time
    
    # Final results
    final_test_acc = test_accuracies[-1]
    print(f"\nFinal Results:")
    print(f"Test Accuracy: {final_test_acc:.2f}%")
    print(f"Training Time: {total_time:.2f}s")
    print(f"Time per Epoch: {total_time/epochs:.2f}s")
    
    # Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / f"mnist_{model_name}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    # Prepare results
    results = {
        'model_name': model_name,
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
        results_path = results_dir / f"results_{model_name}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")
    
    return results

def benchmark_all_models(device='cpu', epochs=10, lr=1e-3, batch_size=64):
    """Benchmark all available models with fixed number of epochs"""
    print("MNIST Model Benchmarking")
    print("=" * 50)
    print(f"Device: {device}")
    print(f"Epochs per model: {epochs}")
    print(f"Learning Rate: {lr}")
    print(f"Batch Size: {batch_size}")
    print(f"Available Models: {list(MODELS.keys())}")
    
    all_results = {}
    
    for model_name in MODELS.keys():
        try:
            results = benchmark_model(model_name, device, epochs, lr, batch_size)
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
model = 'all'  # Model to benchmark: 'all' or one of ['cnn', 'mlp', 'lenet5', 'resnet', 'vgg', 'densenet', 'efficientnet', 'transformer']
device = 'cuda'  # Device to use: 'cpu' or 'cuda'
# Note: epochs is already defined above in the default hyperparameters section

def main():
    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        actual_device = 'cpu'
    else:
        actual_device = device
    
    if model == 'all':
        benchmark_all_models(actual_device, epochs, lr, batch_size)
    else:
        benchmark_model(model, actual_device, epochs, lr, batch_size)

if __name__ == '__main__':
    main()