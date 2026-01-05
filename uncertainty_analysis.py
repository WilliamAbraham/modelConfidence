### analyze the relationshop between uncertainty predicted by the model and the loss value
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import time
import json
from pathlib import Path
import os
import os.path as osp
import csv
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from model import get_model, MODELS

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ### load dataset
    batch_size = 64
    data_root = '/scratch/jc14407/datasets' 
    dataset_name = 'cifar10'
    if dataset_name == 'mnist':
        train_dataset = datasets.MNIST(root = data_root, train=True, download=True,
                                    transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(root =data_root, train=False, transform=transforms.ToTensor())
    elif dataset_name == 'cifar10':
        im_x, im_y = 32, 32
        hidden_dim = 64
        epi_channels = 10
        input_channels = 3
        output_channels = 10
        ea = True
        train_dataset = datasets.CIFAR10(root = data_root, train=True, download=True,
                                    transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR10(root =data_root, train=False, transform=transforms.ToTensor())
    #train_dataset = torch.utils.data.Subset(train_dataset, range(1000))  # Use a subset for faster testing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #test_dataset = torch.utils.data.Subset(test_dataset, range(1000))  # Use a subset for faster testing
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    ### load model
    model_name = 'earesnet50'

    ### check if predicted results already exist
    results_file = f"results/uncertainty_loss_analysis_{dataset_name}_{model_name}.npz"
    if os.path.exists(results_file):
        print(f"Results file {results_file} already exists. Loading results.")
        data = np.load(results_file)
        df = pd.DataFrame({
            'epi_uncertainty': data['epi_uncertainty'],
            'entropy_uncertainty': data['entropy_uncertainty'],
            'loss': data['loss'],
            'correct': data['correct'],
            'set': data['set']
        })
    else:
        print(f"Results file {results_file} does not exist. Evaluating model and generating results.")
        models_dir = Path("/home/jc14407/codes/modelConfidence/checkpoints")
        model_path = os.path.join(models_dir, dataset_name+"_"+model_name+".pth")
        model = get_model(model_name,im_x=im_x, im_y=im_y, hidden_dim=hidden_dim, epi_channels=epi_channels, input_channels=input_channels, output_channels=output_channels, ea=ea).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        ### store the uncertainty and loss values from the training and testing sets in the same array
        results = {'epi_uncertainty': [], 'entropy_uncertainty': [], 'loss': [], 'correct': [], 'set': []}  # 'set' indicates whether it's from train or test set
        ### evaluate on training set
        with torch.no_grad():
            for images, target in train_loader:
                images, target = images.to(device), target.to(device)
                output, epi_error = model(images)
                epi_uncertainty = torch.mean(epi_error, dim=1).flatten()
                probs = F.softmax(output, dim=1)
                entropy_uncertainty = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # Entropy as uncertainty measure
                loss = F.cross_entropy(output, target, reduction='none')
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).flatten()
                results['epi_uncertainty'].extend(epi_uncertainty.cpu().numpy().tolist())
                results['entropy_uncertainty'].extend(entropy_uncertainty.cpu().numpy().tolist())
                results['loss'].extend(loss.cpu().numpy().tolist())
                results['correct'].extend(correct.cpu().numpy().flatten().tolist())
                results['set'].extend(['train'] * len(target))
        ### evaluate on testing set
        with torch.no_grad():
            for images, target in test_loader:
                images, target = images.to(device), target.to(device)
                output, epi_error = model(images)
                epi_uncertainty = torch.mean(epi_error, dim=1).flatten()
                probs = F.softmax(output, dim=1)
                entropy_uncertainty = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)  # Entropy as uncertainty measure
                loss = F.cross_entropy(output, target, reduction='none')
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).flatten()
                results['epi_uncertainty'].extend(epi_uncertainty.cpu().numpy().tolist())
                results['entropy_uncertainty'].extend(entropy_uncertainty.cpu().numpy().tolist())
                results['loss'].extend(loss.cpu().numpy().tolist())
                results['correct'].extend(correct.cpu().numpy().flatten().tolist())
                results['set'].extend(['test'] * len(target))
        
        ### save results to a numpy file
        np.savez(f"results/uncertainty_loss_analysis_{dataset_name}_{model_name}.npz", **results)
        df = pd.DataFrame({
            'epi_uncertainty': results['epi_uncertainty'],
            'entropy_uncertainty': results['entropy_uncertainty'],
            'loss': results['loss'],
            'correct': results['correct'],
            'set': results['set']
        })

    df_train = df[df['set']=='train']
    df_test = df[df['set']=='test']
    avg_train_epi_uncertainty = np.mean(df_train['epi_uncertainty'].to_numpy())
    avg_train_entropy_uncertainty = np.mean(df_train['entropy_uncertainty'].to_numpy())
    ### calculate the avarage loss and correctness for training data
    avg_train_loss = np.mean(df_train['loss'].to_numpy())
    avg_train_correctness = np.mean(df_train['correct'].to_numpy())
    ### calculate the minimum confidence that make the avarage
    ### calculate the avarage loss and correctness for testing data
    avg_test_loss = np.mean(df_test['loss'].to_numpy())
    avg_test_correctness = np.mean(df_test['correct'].to_numpy())
    df_test_low_epi_uncertainty = df_test[df_test['epi_uncertainty'] < avg_train_epi_uncertainty*0.1]
    avg_test_low_epi_uncertainty_loss = np.mean(df_test_low_epi_uncertainty['loss'].to_numpy())
    avg_test_low_epi_uncertainty_correctness = np.mean(df_test_low_epi_uncertainty['correct'].to_numpy())
    num_low_epi_uncertainty_samples = len(df_test_low_epi_uncertainty)
    df_test_low_entropy = df_test[df_test['entropy_uncertainty'].to_numpy() < avg_train_entropy_uncertainty*0.1]
    avg_test_low_entropy_loss = np.mean(df_test_low_entropy['loss'].to_numpy())
    avg_test_low_entropy_correctness = np.mean(df_test_low_entropy['correct'].to_numpy())
    num_low_entropy_samples = len(df_test_low_entropy)
    print(f"Average training  epistemic uncdertainty: {avg_train_epi_uncertainty:.4f}")
    print(f"Average training loss: {avg_train_loss:.4f}")
    print(f"Average training correctness: {avg_train_correctness:.4f}")
    print(f"Average testing loss: {avg_test_loss:.4f}")
    print(f"Average testing correctness: {avg_test_correctness:.4f}")
    print(f"Average testing loss for samples with epistemic uncertianty lower than training average: {avg_test_low_epi_uncertainty_loss:.4f}")
    print(f"Average testing correctness for samples with epistemic uncertianty lower than training average: {avg_test_low_epi_uncertainty_correctness:.4f}")
    print(f"Number of low epistemic uncertainty samples in testing set: {num_low_epi_uncertainty_samples}")
    print(f"Number of low entropy samples in testing set: {num_low_entropy_samples}")
    print(f"Average testing loss for samples with entropy uncertainty lower than training average: {avg_test_low_entropy_loss:.4f}")
    print(f"Average testing correctness for samples with entropy uncertainty lower than training average: {avg_test_low_entropy_correctness:.4f}")
    ### select  data
    df = df[df['set']=='test'] 
    ### plot epi uncertainty vs loss
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='epi_uncertainty', y='loss', hue='set', alpha=0.4)
    plt.title(f'Epi Uncertainty vs Loss for {model_name} on {dataset_name}')
    plt.xlabel('Epi Uncertainty')
    plt.ylabel('Loss')
    plt.legend()
    # ### set x and y limits
    plt.xlim(0, df['epi_uncertainty'].max()*0.01)
    plt.ylim(0, df['loss'].max()*0.01)
    max_uncertainty = np.max(df['epi_uncertainty'].to_numpy())
    plt.xticks(np.linspace(0, max_uncertainty*0.01, num=10))
    plt.savefig(f'results/epi_uncertainty_vs_loss_{dataset_name}_{model_name}.png')
    plt.close()
    ### plot confidence sorce vs loss
    confidence_score = np.exp(-df['epi_uncertainty'].to_numpy())
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=confidence_score, y=df['loss'], hue=df['set'], alpha=0.4)
    plt.title(f'Epi Confidence Score vs Loss for {model_name} on {dataset_name}')
    plt.xlabel('Epi Confidence Score')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'results/epi_confidence_vs_loss_{dataset_name}_{model_name}.png')
    plt.close()
    ### plot confidence source vs correctness
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=confidence_score, y=df['correct'], hue=df['set'], alpha=0.4)
    plt.title(f'Epi Confidence Score vs Correctness for {model_name} on {dataset_name}')
    plt.xlabel('Epi Confidence Score')
    plt.ylabel('Correctness')
    plt.legend()
    plt.savefig(f'results/epi_confidence_vs_correct_{dataset_name}_{model_name}.png')
    plt.close()
    ### plot entropy uncertainty vs loss
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='entropy_uncertainty', y='loss', hue='set', alpha=0.4)
    plt.title(f'Entropy Uncertainty vs Loss for {model_name} on {dataset_name}')
    plt.xlabel('Entropy Uncertainty')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'results/entropy_uncertainty_vs_loss_{dataset_name}_{model_name}.png')
    plt.close() 
    ### plot epistemic uncertainty vs loss using line approaximation with standard deviation
    sorted_indices = np.argsort(df['epi_uncertainty'].to_numpy())
    sorted_uncertainty = df['epi_uncertainty'].to_numpy()[sorted_indices]
    sorted_loss = df['loss'].to_numpy()[sorted_indices]
    window_size = 1000
    averaged_uncertainty = np.convolve(sorted_uncertainty, np.ones(window_size)/window_size, mode='valid')
    averaged_loss = np.convolve(sorted_loss, np.ones(window_size)/window_size, mode='valid')
    std_loss = np.array([np.std(sorted_loss[max(0, i - window_size // 2):min(len(sorted_loss), i + window_size // 2)]) for i in range(len(averaged_loss))])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(averaged_uncertainty, averaged_loss, color='blue', label='Averaged Loss')
    ax.fill_between(averaged_uncertainty, averaged_loss - std_loss, averaged_loss + std_loss, color='blue', alpha=0.2, label='Standard Deviation')
    ax.set_xlabel('Epi Uncertainty')
    ax.set_ylabel('Loss')
    ax.set_title(f'Epi Uncertainty vs Loss for {model_name} on {dataset_name}')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, which="both", ls="--")
    ax.legend()
    plt.savefig(f'results/epi_uncertainty_vs_loss_line_{dataset_name}_{model_name}.png')
    plt.close()

    ### plot confidence vs loss using line approaximation with standard variance
    confidence_score = np.exp(-df['epi_uncertainty'].to_numpy())
    sorted_indices = np.argsort(confidence_score)
    sorted_confidence = confidence_score[sorted_indices]
    sorted_loss = df['loss'].to_numpy()[sorted_indices]
    window_size = 1000
    averaged_confidence = np.convolve(sorted_confidence, np.ones(window_size)/window_size, mode='valid')
    averaged_loss = np.convolve(sorted_loss, np.ones(window_size)/window_size, mode='valid')
    std_loss = np.array([np.std(sorted_loss[max(0, i - window_size // 2):min(len(sorted_loss), i + window_size // 2)]) for i in range(len(averaged_loss))])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(averaged_confidence, averaged_loss, color='blue', label='Averaged Loss')
    ax.fill_between(averaged_confidence, averaged_loss - std_loss, averaged_loss + std_loss, color='blue', alpha=0.2, label='Standard Deviation')
    ax.set_xlabel('Epi Confidence Score')
    ax.set_ylabel('Loss')
    #plt.xlim(0.96, 1)
    ax.set_title(f'Epi Confidence Score vs Loss for {model_name} on {dataset_name}')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, which="both", ls="--")
    ax.legend()
    plt.savefig(f'results/epi_confidence_vs_loss_line_{dataset_name}_{model_name}.png')
    plt.close() 

    ### plot entropy uncertainty vs loss using line approaximation with standard variance
    sorted_indices = np.argsort(df['entropy_uncertainty'].to_numpy())
    sorted_uncertainty = df['entropy_uncertainty'].to_numpy()[sorted_indices]
    sorted_loss = df['loss'].to_numpy()[sorted_indices]
    window_size = 1000
    averaged_uncertainty = np.convolve(sorted_uncertainty, np.ones(window_size)/window_size, mode='valid')
    averaged_loss = np.convolve(sorted_loss, np.ones(window_size)/window_size, mode='valid')
    std_loss = np.array([np.std(sorted_loss[max(0, i - window_size // 2):min(len(sorted_loss), i + window_size // 2)]) for i in range(len(averaged_loss))])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(averaged_uncertainty, averaged_loss, color='blue', label='Averaged Loss')
    ax.fill_between(averaged_uncertainty, averaged_loss - std_loss, averaged_loss + std_loss, color='blue', alpha=0.2, label='Standard Deviation')
    ax.set_xlabel('Entropy Uncertainty')
    ax.set_ylabel('Loss')
    ax.set_title(f'Entropy Uncertainty vs Loss for {model_name} on {dataset_name}')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, which="both", ls="--")
    ax.legend()
    plt.savefig(f'results/entropy_uncertainty_vs_loss_line_{dataset_name}_{model_name}.png')
    plt.close()
    ### plot epistemic uncertainty vs correct using line approaximation with standard variance
    sorted_indices = np.argsort(df['epi_uncertainty'].to_numpy())
    sorted_uncertainty = df['epi_uncertainty'].to_numpy()[sorted_indices]
    sorted_correct = df['correct'].to_numpy()[sorted_indices]
    window_size = 1000
    averaged_uncertainty = np.convolve(sorted_uncertainty, np.ones(window_size)/window_size, mode='valid')
    averaged_correct = np.convolve(sorted_correct, np.ones(window_size)/window_size, mode='valid')
    #std_correct = np.array([np.std(sorted_correct[max(0, i - window_size // 2):min(len(sorted_correct), i + window_size // 2)]) for i in range(len(averaged_correct))])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(averaged_uncertainty, averaged_correct, color='blue', label='Averaged Correctness')
    #ax.fill_between(averaged_uncertainty, averaged_correct - std_correct, averaged_correct + std_correct, color='blue', alpha=0.2, label='Standard Deviation')
    ax.set_xlabel('Epi Uncertainty')
    ax.set_ylabel('Correctness')
    ax.set_title(f'Epi Uncertainty vs Correctness for {model_name} on {dataset_name}')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, which="both", ls="--")
    ax.legend()
    plt.savefig(f'results/epi_uncertainty_vs_correct_line_{dataset_name}_{model_name}.png')
    plt.close()

    ### plot confidence source vs correct using line approaximation with standard variance
    confidence_score = np.exp(-df['epi_uncertainty'].to_numpy())
    sorted_indices = np.argsort(confidence_score)
    sorted_confidence = confidence_score[sorted_indices]
    sorted_correct = df['correct'].to_numpy()[sorted_indices]
    window_size = 1000
    averaged_confidence = np.convolve(sorted_confidence, np.ones(window_size)/window_size, mode='valid')
    averaged_correct = np.convolve(sorted_correct, np.ones(window_size)/window_size, mode='valid')
    #std_correct = np.array([np.std(sorted_correct[max(0, i - window_size // 2):min(len(sorted_correct), i + window_size // 2)]) for i in range(len(averaged_correct))])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(averaged_confidence, averaged_correct, color='blue', label='Averaged Correctness')
    #ax.fill_between(averaged_confidence, averaged_correct - std_correct, averaged_correct + std_correct, color='blue', alpha=0.2, label='Standard Deviation')
    ax.set_xlabel('Epi Confidence Score')
    ax.set_ylabel('Correctness')
    ax.set_title(f'Epi Confidence Score vs Correctness for {model_name} on {dataset_name}')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, which="both", ls="--")
    ax.legend()
    plt.savefig(f'results/epi_confidence_vs_correct_line_{dataset_name}_{model_name}.png')
    plt.close() 

    ### plot entropy uncertainty vs correct using line approaximation with standard variance
    sorted_indices = np.argsort(df['entropy_uncertainty'].to_numpy())
    sorted_uncertainty = df['entropy_uncertainty'].to_numpy()[sorted_indices]
    sorted_correct = df['correct'].to_numpy()[sorted_indices]
    window_size = 1000
    averaged_uncertainty = np.convolve(sorted_uncertainty, np.ones(window_size)/window_size, mode='valid')
    averaged_correct = np.convolve(sorted_correct, np.ones(window_size)/window_size, mode='valid')
    #std_correct = np.array([np.std(sorted_correct[max(0, i - window_size // 2):min(len(sorted_correct), i + window_size // 2)]) for i in range(len(averaged_correct))])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(averaged_uncertainty, averaged_correct, color='blue', label='Averaged Correctness')
    #ax.fill_between(averaged_uncertainty, averaged_correct - std_correct, averaged_correct + std_correct, color='blue', alpha=0.2, label='Standard Deviation')
    ax.set_xlabel('Entropy Uncertainty')
    ax.set_ylabel('Correctness')
    ax.set_title(f'Entropy Uncertainty vs Correctness for {model_name} on {dataset_name}')
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, which="both", ls="--")
    ax.legend()
    plt.savefig(f'results/entropy_uncertainty_vs_correct_line_{dataset_name}_{model_name}.png')
    plt.close()
