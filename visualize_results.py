import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
import os
import sys
import traceback

# Set style for better visualizations
plt.style.use('default')  # Use default style instead of seaborn
sns.set_theme(style="whitegrid")  # Set seaborn theme properly
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = {
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'pandas': 'pandas',
        'scikit-learn': 'sklearn',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:", ", ".join(missing_packages))
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    return True

def check_data_files():
    """Check if all required data files exist."""
    required_files = [
        'MedicalData/results/supervised_results.csv',
        'MedicalData/results/inductive_results.csv',
        'MedicalData/results/transductive_results.csv',
        'MedicalData/results/aucp_results.csv'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("Missing required data files:", ", ".join(missing_files))
        print("Please ensure all result files are present in the MedicalData/results directory")
        return False
    return True

def load_results():
    """Load all result files."""
    results_dir = 'MedicalData/results'
    results = {}
    
    # Load supervised results
    supervised_df = pd.read_csv(os.path.join(results_dir, 'supervised_results.csv'))
    results['supervised'] = supervised_df
    
    # Load GAN results
    inductive_df = pd.read_csv(os.path.join(results_dir, 'inductive_results.csv'))
    transductive_df = pd.read_csv(os.path.join(results_dir, 'transductive_results.csv'))
    aucp_df = pd.read_csv(os.path.join(results_dir, 'aucp_results.csv'))
    results['gan'] = {
        'inductive': inductive_df,
        'transductive': transductive_df,
        'aucp': aucp_df
    }
    
    return results

def plot_roc_curves(results):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    # Plot supervised model ROC
    fpr_sup, tpr_sup, _ = roc_curve(results['supervised']['label'], results['supervised']['score'])
    auc_sup = auc(fpr_sup, tpr_sup)
    plt.plot(fpr_sup, tpr_sup, label=f'Supervised CNN (AUC = {auc_sup:.3f})', linewidth=2)
    
    # Plot GAN model ROC
    fpr_gan, tpr_gan, _ = roc_curve(results['gan']['inductive']['label'], results['gan']['inductive']['score'])
    auc_gan = auc(fpr_gan, tpr_gan)
    plt.plot(fpr_gan, tpr_gan, label=f'GAN (AUC = {auc_gan:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: Supervised CNN vs GAN')
    plt.legend()
    plt.grid(True)
    plt.savefig('MedicalData/results/roc_curves.png')
    plt.close()

def plot_score_distributions(results):
    """Plot score distributions for both models."""
    plt.figure(figsize=(12, 6))
    
    # Supervised model scores
    sns.kdeplot(data=results['supervised'], x='score', hue='label', 
                label='Supervised CNN', common_norm=False)
    
    # GAN model scores
    sns.kdeplot(data=results['gan']['inductive'], x='score', hue='label',
                label='GAN', common_norm=False)
    
    plt.title('Score Distributions by Model and Class')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('MedicalData/results/score_distributions.png')
    plt.close()

def plot_model_comparison():
    """Create a bar plot comparing model performances."""
    models = ['Supervised CNN', 'GAN']
    auc_scores = [0.8751, 0.6794]  # From results summary
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(models, auc_scores)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.title('Model Performance Comparison')
    plt.ylabel('AUC Score')
    plt.ylim(0, 1)
    plt.grid(True, axis='y')
    plt.savefig('MedicalData/results/model_comparison.png')
    plt.close()

def plot_training_metrics():
    """Plot training metrics over time."""
    # Training metrics from results summary
    epochs = range(1, 23)  # Fix: Changed from 24 to 23 to match data length
    supervised_acc = [54.22, 62.66, 65.26, 63.31, 64.94, 65.58, 68.51, 
                     71.75, 70.78, 69.48, 73.05, 73.70, 72.08, 73.38, 
                     74.68, 75.65, 76.95, 80.19, 78.90, 80.19, 78.25, 
                     79.87]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, supervised_acc, marker='o', label='Training Accuracy')
    plt.axhline(y=86.25, color='r', linestyle='--', label='Best Validation Accuracy')
    
    plt.title('Supervised CNN Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    plt.savefig('MedicalData/results/training_progress.png')
    plt.close()

def main():
    """Generate all visualizations with error handling."""
    try:
        print("Checking dependencies...")
        if not check_dependencies():
            return
        
        print("Checking data files...")
        if not check_data_files():
            return
        
        print("Creating results directory...")
        os.makedirs('MedicalData/results', exist_ok=True)
        
        print("Loading results...")
        results = load_results()
        
        print("Generating visualizations...")
        plot_roc_curves(results)
        print("- Generated ROC curves")
        
        plot_score_distributions(results)
        print("- Generated score distributions")
        
        plot_model_comparison()
        print("- Generated model comparison")
        
        plot_training_metrics()
        print("- Generated training progress")
        
        print("\nVisualizations have been saved to MedicalData/results/")
        print("Generated files:")
        print("- roc_curves.png")
        print("- score_distributions.png")
        print("- model_comparison.png")
        print("- training_progress.png")
        
    except Exception as e:
        print("\nError occurred while generating visualizations:")
        print(str(e))
        print("\nTraceback:")
        traceback.print_exc()
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    if not success:
        print("\nVisualization generation failed. Please check the error messages above.")
    input("\nPress Enter to exit...") 