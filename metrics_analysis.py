import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import seaborn as sns
from collections import defaultdict
import pandas as pd

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_metrics_folder():
    """Create the metrics folder structure"""
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(exist_ok=True)
    
    # Create subfolders for each model
    models = ['cnn', 'mlp', 'lenet5', 'resnet', 'vgg', 'densenet', 'efficientnet', 'transformer']
    for model in models:
        model_dir = metrics_dir / model
        model_dir.mkdir(exist_ok=True)
    
    return metrics_dir

def load_results():
    """Load all result files"""
    results = {}
    results_dir = Path("results")
    
    # Check if results directory exists
    if not results_dir.exists():
        print(f"Results directory '{results_dir}' not found.")
        return results
    
    # Load all JSON files from results directory
    for result_file in results_dir.glob("*.json"):
        # Extract model name from filename
        # Handles both "results_modelname.json" and "benchmark_summary.json"
        if result_file.stem == "benchmark_summary":
            # Load summary file and extract individual model results
            try:
                with open(result_file, 'r') as f:
                    summary_data = json.load(f)
                    # If it's a dict with model names as keys, use it directly
                    if isinstance(summary_data, dict):
                        results.update(summary_data)
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
        else:
            model_name = result_file.stem.replace("results_", "")
            try:
                with open(result_file, 'r') as f:
                    results[model_name] = json.load(f)
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
    
    return results

def plot_training_curves(results, metrics_dir):
    """Plot training and validation curves for each model"""
    print("Generating training curves...")
    
    for model_name, data in results.items():
        if 'error' in data:
            continue
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        epochs = range(1, len(data['train_losses']) + 1)
        
        # Loss curves
        ax1.plot(epochs, data['train_losses'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, data['test_losses'], 'r-', label='Test Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{model_name.upper()} - Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curve
        ax2.plot(epochs, data['test_accuracies'], 'g-', label='Test Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title(f'{model_name.upper()} - Accuracy Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(metrics_dir / model_name / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {model_name} training curves saved")

def plot_model_comparison(results, metrics_dir):
    """Create comparison plots across all models"""
    print("Generating model comparison plots...")
    
    # Filter out models with errors
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_results:
        print("No valid results to compare")
        return
    
    # Prepare data for comparison
    model_names = list(valid_results.keys())
    accuracies = [valid_results[m]['final_test_accuracy'] for m in model_names]
    parameters = [valid_results[m]['parameters'] for m in model_names]
    training_times = [valid_results[m]['training_time'] for m in model_names]
    time_per_epoch = [valid_results[m]['time_per_epoch'] for m in model_names]
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy comparison
    bars1 = ax1.bar(model_names, accuracies, color='skyblue', alpha=0.7)
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # Parameters comparison
    bars2 = ax2.bar(model_names, parameters, color='lightcoral', alpha=0.7)
    ax2.set_ylabel('Number of Parameters')
    ax2.set_title('Model Size Comparison')
    ax2.set_yscale('log')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, param in zip(bars2, parameters):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
                f'{param:,}', ha='center', va='bottom', fontsize=8)
    
    # Training time comparison
    bars3 = ax3.bar(model_names, training_times, color='lightgreen', alpha=0.7)
    ax3.set_ylabel('Total Training Time (seconds)')
    ax3.set_title('Training Time Comparison')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, time in zip(bars3, training_times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(training_times)*0.01, 
                f'{time:.1f}s', ha='center', va='bottom')
    
    # Efficiency plot (Accuracy vs Parameters)
    scatter = ax4.scatter(parameters, accuracies, s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
    ax4.set_xlabel('Number of Parameters')
    ax4.set_ylabel('Test Accuracy (%)')
    ax4.set_title('Efficiency: Accuracy vs Model Size')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)
    
    # Add model labels
    for i, model in enumerate(model_names):
        ax4.annotate(model, (parameters[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(metrics_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Model comparison plots saved")

def plot_learning_efficiency(results, metrics_dir):
    """Plot learning efficiency metrics"""
    print("Generating learning efficiency plots...")
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_results:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy vs Training Time
    model_names = list(valid_results.keys())
    accuracies = [valid_results[m]['final_test_accuracy'] for m in model_names]
    training_times = [valid_results[m]['training_time'] for m in model_names]
    
    scatter1 = ax1.scatter(training_times, accuracies, s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
    ax1.set_xlabel('Training Time (seconds)')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Accuracy vs Training Time')
    ax1.grid(True, alpha=0.3)
    
    for i, model in enumerate(model_names):
        ax1.annotate(model, (training_times[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Accuracy vs Time per Epoch
    time_per_epoch = [valid_results[m]['time_per_epoch'] for m in model_names]
    
    scatter2 = ax2.scatter(time_per_epoch, accuracies, s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
    ax2.set_xlabel('Time per Epoch (seconds)')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Accuracy vs Time per Epoch')
    ax2.grid(True, alpha=0.3)
    
    for i, model in enumerate(model_names):
        ax2.annotate(model, (time_per_epoch[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(metrics_dir / 'learning_efficiency.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ Learning efficiency plots saved")

def create_summary_table(results, metrics_dir):
    """Create a summary table of all results"""
    print("Creating summary table...")
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_results:
        return
    
    # Create DataFrame
    data = []
    for model_name, result in valid_results.items():
        data.append({
            'Model': model_name.upper(),
            'Parameters': f"{result['parameters']:,}",
            'Final Accuracy (%)': f"{result['final_test_accuracy']:.2f}",
            'Training Time (s)': f"{result['training_time']:.2f}",
            'Time/Epoch (s)': f"{result['time_per_epoch']:.2f}",
            'Epochs': result['epochs'],
            'Learning Rate': result['learning_rate'],
            'Batch Size': result['batch_size']
        })
    
    df = pd.DataFrame(data)
    
    # Create table plot
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f1f1f2')
    
    plt.title('MNIST Model Benchmarking Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(metrics_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save as CSV
    df.to_csv(metrics_dir / 'summary_table.csv', index=False)
    
    print("  ✓ Summary table saved")

def generate_individual_model_analysis(results, metrics_dir):
    """Generate detailed analysis for each model"""
    print("Generating individual model analysis...")
    
    for model_name, data in results.items():
        if 'error' in data:
            continue
        
        model_dir = metrics_dir / model_name
        
        # Create detailed analysis plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = range(1, len(data['train_losses']) + 1)
        
        # Training curves
        ax1.plot(epochs, data['train_losses'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, data['test_losses'], 'r-', label='Test Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{model_name.upper()} - Loss Curves')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy progression
        ax2.plot(epochs, data['test_accuracies'], 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Test Accuracy (%)')
        ax2.set_title(f'{model_name.upper()} - Accuracy Progression')
        ax2.grid(True, alpha=0.3)
        
        # Loss distribution (box plot style)
        ax3.boxplot([data['train_losses'], data['test_losses']], 
                   labels=['Training Loss', 'Test Loss'])
        ax3.set_ylabel('Loss')
        ax3.set_title(f'{model_name.upper()} - Loss Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Model statistics
        stats_text = f"""
Model: {model_name.upper()}
Parameters: {data['parameters']:,}
Final Accuracy: {data['final_test_accuracy']:.2f}%
Training Time: {data['training_time']:.2f}s
Time per Epoch: {data['time_per_epoch']:.2f}s
Epochs: {data['epochs']}
Learning Rate: {data['learning_rate']}
Batch Size: {data['batch_size']}
        """
        
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        ax4.set_title(f'{model_name.upper()} - Model Statistics')
        
        plt.tight_layout()
        plt.savefig(model_dir / 'detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {model_name} detailed analysis saved")

def main():
    """Main function to generate all metrics and visualizations"""
    print("Starting MNIST Model Metrics Analysis...")
    print("=" * 50)
    
    # Create folder structure
    metrics_dir = create_metrics_folder()
    print(f"Created metrics folder structure in: {metrics_dir}")
    
    # Load results
    results = load_results()
    if not results:
        print("No result files found. Please run benchmarks first.")
        return
    
    print(f"Loaded results for {len(results)} models: {list(results.keys())}")
    
    # Generate all visualizations
    plot_training_curves(results, metrics_dir)
    plot_model_comparison(results, metrics_dir)
    plot_learning_efficiency(results, metrics_dir)
    create_summary_table(results, metrics_dir)
    generate_individual_model_analysis(results, metrics_dir)
    
    print("\n" + "=" * 50)
    print("Metrics analysis complete!")
    print(f"All visualizations saved to: {metrics_dir}")
    print("=" * 50)

if __name__ == "__main__":
    main()
