import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import datasets, transforms
from torch.nn.functional import softmax

from ABNN.bnl import BNL
from model import get_model

# Configuration
batch_size = 64
num_workers = 1
model_path = 'modelsBNL/mnist_cnn_bnl.pth'
num_samples = 10  # Number of Monte Carlo samples for uncertainty estimation
num_test_samples = 5  # Number of random test images to evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the BNL model
print("Loading BNL model...")
model = get_model('cnn', input_channels=1, output_channels=10, norm_layer=BNL).to(device)

# Check if model exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}. Please train the model first.")

# Load model weights
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()
print(f"Model loaded from {model_path}")

# Prepare test data (using same split as training)
transform = transforms.ToTensor()
train_dataset = datasets.MNIST('datasets', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('datasets', train=False, download=True, transform=transform)
full_dataset = ConcatDataset([train_dataset, test_dataset])

total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

# Use same random seed as training to get same split
_, _, test_subset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# Create test DataLoader
testloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Get random samples from test set
print(f"\nSampling {num_test_samples} random images from test set...")
indices = torch.randperm(len(test_subset))[:num_test_samples]
sample_images = []
sample_labels = []

for idx in indices:
    image, label = test_subset[idx]
    sample_images.append(image)
    sample_labels.append(label)

sample_images = torch.stack(sample_images).to(device)
sample_labels = torch.tensor(sample_labels).to(device)

print("\n" + "="*70)
print("BNL Model Predictions with Uncertainty Estimates")
print("="*70)

# Evaluate each sample
with torch.no_grad():
    for i in range(num_test_samples):
        image = sample_images[i:i+1]  # Keep batch dimension
        true_label = sample_labels[i].item()
        
        # Monte Carlo sampling for uncertainty estimation
        mc_outputs = []
        for _ in range(num_samples):
            output = model(image)
            mc_outputs.append(output)
        
        # Stack all outputs
        mc_outputs = torch.stack(mc_outputs)  # Shape: [num_samples, 1, 10]
        
        # Get mean prediction
        mean_output = mc_outputs.mean(dim=0)  # Shape: [1, 10]
        mean_probs = softmax(mean_output, dim=1)[0]  # Shape: [10]
        
        # Get prediction
        predicted_label = mean_probs.argmax().item()
        confidence = mean_probs[predicted_label].item()
        
        # Calculate uncertainty (variance in predictions)
        probs_samples = softmax(mc_outputs, dim=-1)  # Shape: [num_samples, 1, 10]
        probs_samples = probs_samples.squeeze(1)  # Shape: [num_samples, 10]
        
        # Uncertainty as variance of predicted class probability
        predicted_class_probs = probs_samples[:, predicted_label]
        uncertainty = predicted_class_probs.var().item()
        
        # Alternative: entropy-based uncertainty
        mean_entropy = -(mean_probs * torch.log(mean_probs + 1e-10)).sum().item()
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(mean_probs, 3)
        
        print(f"\nSample {i+1}:")
        print(f"  True Label: {true_label}")
        print(f"  Predicted Label: {predicted_label} {'✓' if predicted_label == true_label else '✗'}")
        print(f"  Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"  Uncertainty (Variance): {uncertainty:.6f}")
        print(f"  Uncertainty (Entropy): {mean_entropy:.4f}")
        print(f"  Top 3 Predictions:")
        for j, (prob, idx) in enumerate(zip(top3_probs, top3_indices)):
            print(f"    {j+1}. Class {idx.item()}: {prob.item():.4f} ({prob.item()*100:.2f}%)")
        
        # Show probability distribution
        print(f"  Full Probability Distribution:")
        for class_idx in range(10):
            prob = mean_probs[class_idx].item()
            bar = '█' * int(prob * 50)  # Visual bar
            print(f"    Class {class_idx}: {prob:.4f} {bar}")

print("\n" + "="*70)
print("Evaluation complete!")
print("="*70)

