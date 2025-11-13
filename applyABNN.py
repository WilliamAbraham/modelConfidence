import os

import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import datasets, transforms

from ABNN.bnl import BNL
from ABNN.train import train_model
from ABNN.test_and_eval import test_model_with_metrics
from model import get_model

batch_size = 64
num_workers = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

mnist_cnn_bnl = get_model('cnn', input_channels=1, output_channels=10, norm_layer=BNL).to(device)

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
    generator=torch.Generator().manual_seed(42)
)

# Create DataLoaders
trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
validloader = DataLoader(valid_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

testloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

os.makedirs('modelsBNL', exist_ok=True)

print("DataLoaders created")
print("Training BNL model")

train_losses, val_losses = train_model(
    model=mnist_cnn_bnl, 
    train_loader=trainloader, 
    val_loader=validloader,
    epochs=1, 
    learning_rate=0.1, 
    gamma_lr=0.2,
    milestones=[60, 120, 160], 
    save_path='modelsBNL/mnist_cnn_bnl.pth', 
    Weight_decay=5e-4,
    Momentum=0.9, 
    Optimizer_type='SGD',  
    Loss_fn='CustomMAPLoss',
    Num_classes=10,
    BNL_enable=True,
    BNL_load_path='models/mnist_cnn.pth'
)

print("Training complete")

# Testing the model with metrics
test_model_with_metrics(
    loss_fn='CustomMAPLoss', 
    model=mnist_cnn_bnl, 
    test_loader=testloader, 
    load_path='modelsBNL/mnist_cnn_bnl.pth',
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
    models=[torch.load('modelsBNL/mnist_cnn_bnl.pth')],
    num_samples=10, 
    num_classes=10,
    Weight_decay=5e-4
)

print("Testing complete")