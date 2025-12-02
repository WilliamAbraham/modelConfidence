import os

import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import datasets, transforms

from ABNN.bnl import BNL
from ABNN.train import train_model
from ABNN.test_and_eval import test_model_with_metrics
from model import get_model

def makeBNN(model_name, device):
    """Create a BNL model with the specified architecture."""
    if model_name not in ['cnn', 'resnet', 'vgg', 'densenet', 'lenet5', 'mlp', 'transformer', 'efficientnet']:
        raise ValueError(f"Invalid model name: {model_name}. Available: ['cnn', 'resnet', 'vgg', 'densenet', 'lenet5', 'mlp', 'transformer', 'efficientnet']")
    
    model = get_model(model_name, input_channels=1, output_channels=10, norm_layer=BNL).to(device)
    return model

def create_data_loaders(batch_size=64, num_workers=1, random_seed=42):
    """Create train, validation, and test data loaders with 80/15/5 split."""
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('datasets', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('datasets', train=False, download=True, transform=transform)
    full_dataset = ConcatDataset([train_dataset, test_dataset])

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_subset, valid_subset, test_subset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers) # 80%
    validloader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers) # 15%
    testloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers) # 5%
    
    return trainloader, validloader, testloader

def trainBNN(model_name, batch_size=64, num_workers=1, epochs=200, 
             learning_rate=0.001, gamma_lr=0.2, milestones=[60, 120, 160],
             weight_decay=5e-4, momentum=0.9, optimizer_type='SGD'):
    """Train a BNL model with the specified architecture."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = makeBNN(model_name, device)
    
    # Create data loaders
    trainloader, validloader, testloader = create_data_loaders(batch_size, num_workers)
    
    os.makedirs('modelsBNL', exist_ok=True)

    print(f"DataLoaders created")
    print(f"Training BNL model: {model_name}")

    train_losses, val_losses = train_model(
        model=model, 
        train_loader=trainloader, 
        val_loader=validloader,
        epochs=epochs, 
        learning_rate=learning_rate, 
        gamma_lr=gamma_lr,
        milestones=milestones, 
        save_path=f'modelsBNL/{model_name}_bnl.pth', 
        Weight_decay=weight_decay,
        Momentum=momentum, 
        Optimizer_type=optimizer_type,  
        Loss_fn='CustomMAPLoss',
        Num_classes=10,
        BNL_enable=True,
        BNL_load_path=f'models/mnist_{model_name}.pth'
    )

    print("Training complete")
    return train_losses, val_losses

def testBNN(model_name, batch_size=64, num_workers=1, num_samples=10):
    """Test a trained BNL model and compute evaluation metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_path = f'modelsBNL/{model_name}_bnl.pth'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please train the model first.")

    # Create model and load weights
    model = makeBNN(model_name, device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    
    # Create test data loader
    _, _, testloader = create_data_loaders(batch_size, num_workers)

    print(f"Testing BNL model: {model_name}")
    
    # Testing the model with metrics
    test_model_with_metrics(
        loss_fn='CustomMAPLoss', 
        model=model, 
        test_loader=testloader, 
        load_path=model_path,
        calculate_uncert=True, 
        calculate_nll_loss=True, 
        calculate_ece_error=True,
        calculate_auprc=True, 
        calculate_auc_roc=True, 
        calculate_fpr_95=True, 
        count_params=True,
        plot_uncert=False, 
        predict_uncert=False, 
        model_class=None, 
        models=None,
        num_samples=num_samples, 
        num_classes=10,
        Weight_decay=5e-4
    )

    print("Testing complete")

if __name__ == '__main__':
    # Example usage
    model_name = 'cnn'
    
    # Train the model
    trainBNN(model_name)
    
    # Test the model
    testBNN(model_name)