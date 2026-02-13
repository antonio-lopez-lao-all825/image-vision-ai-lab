"""
MNIST model evaluation on test set.
MNIST test set contains 10,000 images separate from training.
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNISTNet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    f1_score, precision_score, recall_score
)
import seaborn as sns


def load_test_data(batch_size=64):
    """Loads the MNIST test set"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False,  # TEST set
        download=True, 
        transform=transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader, test_dataset


def evaluate_model(model, test_loader, device):
    """Evaluates the model and returns predictions and metrics"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss = criterion(output, target)
            total_loss += loss.item()
            
            probs = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return {
        'predictions': np.array(all_preds),
        'targets': np.array(all_targets),
        'probabilities': np.array(all_probs),
        'avg_loss': total_loss / len(test_loader)
    }


def print_metrics(targets, predictions):
    """Prints evaluation metrics"""
    accuracy = 100 * np.mean(predictions == targets)
    f1 = f1_score(targets, predictions, average='weighted')
    precision = precision_score(targets, predictions, average='weighted')
    recall = recall_score(targets, predictions, average='weighted')
    
    print("="*60)
    print("EVALUATION METRICS ON TEST SET (10,000 images)")
    print("="*60)
    print(f"\nAccuracy:  {accuracy:.2f}%")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT PER DIGIT")
    print("="*60)
    print(classification_report(targets, predictions, 
                                target_names=[str(i) for i in range(10)]))
    
    return accuracy, f1, precision, recall


def plot_confusion_matrix(targets, predictions, save_path='mnist/test_confusion_matrix.png'):
    """Generates confusion matrix"""
    cm = confusion_matrix(targets, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Prediction')
    plt.ylabel('True Value')
    plt.title('Confusion Matrix - MNIST Test Set')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_pred_vs_true(targets, predictions, save_path='mnist/test_pred_vs_true.png'):
    """Scatter plot of predictions vs true values"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter with jitter
    ax1 = axes[0]
    jitter_t = targets + np.random.normal(0, 0.15, len(targets))
    jitter_p = predictions + np.random.normal(0, 0.15, len(predictions))
    colors = ['green' if t == p else 'red' for t, p in zip(targets, predictions)]
    
    ax1.scatter(jitter_t, jitter_p, c=colors, alpha=0.3, s=10)
    ax1.plot([-0.5, 9.5], [-0.5, 9.5], 'b--', linewidth=2, label='Ideal line')
    ax1.set_xlabel('True Value')
    ax1.set_ylabel('Prediction')
    ax1.set_title('Prediction vs True (Green=Correct, Red=Incorrect)')
    ax1.set_xlim(-0.5, 9.5)
    ax1.set_ylim(-0.5, 9.5)
    ax1.set_xticks(range(10))
    ax1.set_yticks(range(10))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy per class
    ax2 = axes[1]
    class_acc = []
    for digit in range(10):
        mask = targets == digit
        acc = 100 * np.mean(predictions[mask] == digit) if mask.sum() > 0 else 0
        class_acc.append(acc)
    
    bars = ax2.bar(range(10), class_acc, color='steelblue', edgecolor='black')
    ax2.axhline(y=np.mean(class_acc), color='r', linestyle='--', 
                label=f'Mean: {np.mean(class_acc):.1f}%')
    ax2.set_xlabel('Digit')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy per Class')
    ax2.set_xticks(range(10))
    ax2.set_ylim(0, 105)
    ax2.legend()
    
    for bar, acc in zip(bars, class_acc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Scatter plot saved to {save_path}")


def plot_error_analysis(targets, predictions, probabilities, test_dataset, 
                        save_path='mnist/test_error_analysis.png'):
    """Shows examples of model errors"""
    errors_idx = np.where(targets != predictions)[0]
    
    if len(errors_idx) == 0:
        print("No errors to analyze!")
        return
    
    # Select errors with highest confidence (worst errors)
    error_confidence = probabilities[errors_idx].max(axis=1)
    worst_errors = errors_idx[np.argsort(error_confidence)[-16:]]
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('Error Analysis - Incorrect Predictions with High Confidence', 
                 fontsize=14)
    
    for i, idx in enumerate(worst_errors):
        ax = axes[i // 4, i % 4]
        image = test_dataset.data[idx].numpy()
        
        ax.imshow(image, cmap='gray')
        confidence = probabilities[idx].max() * 100
        ax.set_title(f'True: {targets[idx]}\nPred: {predictions[idx]} ({confidence:.1f}%)',
                    fontsize=10, color='red')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Error analysis saved to {save_path}")
    print(f"Total errors: {len(errors_idx)} of {len(targets)} ({100*len(errors_idx)/len(targets):.2f}%)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = MNISTNet().to(device)
    try:
        model.load_state_dict(torch.load('mnist/mnist_model.pth', map_location=device))
        print("Model loaded from mnist/mnist_model.pth")
    except FileNotFoundError:
        print("ERROR: mnist/mnist_model.pth not found")
        print("First run: python train.py")
        return
    
    # Load test data
    test_loader, test_dataset = load_test_data()
    print(f"Test set: {len(test_dataset)} images")
    
    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(model, test_loader, device)
    
    # Metrics
    print_metrics(results['targets'], results['predictions'])
    
    # Plots
    print("\nGenerating visualizations...")
    plot_confusion_matrix(results['targets'], results['predictions'])
    plot_pred_vs_true(results['targets'], results['predictions'])
    plot_error_analysis(results['targets'], results['predictions'], 
                       results['probabilities'], test_dataset)
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print("="*60)


if __name__ == "__main__":
    main()
