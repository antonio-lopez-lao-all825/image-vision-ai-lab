import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNISTNet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm
import seaborn as sns


def evaluate(model, data_loader, criterion, device):
    """Evaluates the model and returns metrics"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    accuracy = 100 * np.sum(all_preds == all_targets) / len(all_targets)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'predictions': all_preds,
        'targets': all_targets
    }


def plot_training_metrics(history, save_path='mnist/training_plots/training_metrics.png'):
    """Generates training metrics plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-o', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss per Epoch')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['val_accuracy'], 'g-o', label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy per Epoch')
    ax2.legend()
    ax2.grid(True)
    
    # F1, Precision, Recall
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['val_f1'], 'b-o', label='F1 Score')
    ax3.plot(epochs, history['val_precision'], 'g-s', label='Precision')
    ax3.plot(epochs, history['val_recall'], 'r-^', label='Recall')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Score')
    ax3.set_title('F1, Precision and Recall per Epoch')
    ax3.legend()
    ax3.grid(True)
    
    # Train vs Val Loss comparison
    ax4 = axes[1, 1]
    ax4.fill_between(epochs, history['train_loss'], alpha=0.3, label='Train Loss')
    ax4.fill_between(epochs, history['val_loss'], alpha=0.3, label='Val Loss')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.set_title('Train vs Val Loss Comparison')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training metrics saved to {save_path}")


def plot_confusion_matrix(targets, predictions, save_path='mnist/training_plots/confusion_matrix.png'):
    """Generates confusion matrix"""
    cm = confusion_matrix(targets, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Prediction')
    plt.ylabel('True Value')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_pred_vs_true_scatter(targets, predictions, save_path='mnist/training_plots/pred_vs_true.png'):
    """Generates scatter plot of predictions vs true values with jitter"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot with jitter
    ax1 = axes[0]
    jitter_targets = targets + np.random.normal(0, 0.15, len(targets))
    jitter_preds = predictions + np.random.normal(0, 0.15, len(predictions))
    
    colors = ['green' if t == p else 'red' for t, p in zip(targets, predictions)]
    ax1.scatter(jitter_targets, jitter_preds, c=colors, alpha=0.3, s=10)
    ax1.plot([-0.5, 9.5], [-0.5, 9.5], 'b--', linewidth=2, label='Ideal line')
    ax1.set_xlabel('True Value')
    ax1.set_ylabel('Prediction')
    ax1.set_title('Prediction vs True Value (with jitter)\nGreen=Correct, Red=Incorrect')
    ax1.set_xlim(-0.5, 9.5)
    ax1.set_ylim(-0.5, 9.5)
    ax1.set_xticks(range(10))
    ax1.set_yticks(range(10))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy per class
    ax2 = axes[1]
    class_accuracy = []
    for digit in range(10):
        mask = targets == digit
        if np.sum(mask) > 0:
            acc = np.sum(predictions[mask] == digit) / np.sum(mask) * 100
        else:
            acc = 0
        class_accuracy.append(acc)
    
    bars = ax2.bar(range(10), class_accuracy, color='steelblue', edgecolor='black')
    ax2.axhline(y=np.mean(class_accuracy), color='r', linestyle='--', label=f'Mean: {np.mean(class_accuracy):.1f}%')
    ax2.set_xlabel('Digit')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy per Class')
    ax2.set_xticks(range(10))
    ax2.set_ylim(0, 105)
    ax2.legend()
    
    # Add values above bars
    for bar, acc in zip(bars, class_accuracy):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Scatter plot pred vs true saved to {save_path}")


def train():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    epochs = 10
    
    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model, loss and optimizer
    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Metrics history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': []
    }
    
    # Training
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Progress bar with tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", 
                    unit='batch', leave=True)
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Evaluation
        val_metrics = evaluate(model, test_loader, criterion, device)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Accuracy: {val_metrics['accuracy']:.2f}% | "
              f"F1: {val_metrics['f1']:.4f}")
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    final_metrics = evaluate(model, test_loader, criterion, device)
    print(f"\nAccuracy: {final_metrics['accuracy']:.2f}%")
    print(f"F1 Score: {final_metrics['f1']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(final_metrics['targets'], final_metrics['predictions'], 
                                target_names=[str(i) for i in range(10)]))
    
    # Generate plots
    print("\nGenerating visualizations...")
    plot_training_metrics(history)
    plot_confusion_matrix(final_metrics['targets'], final_metrics['predictions'])
    plot_pred_vs_true_scatter(final_metrics['targets'], final_metrics['predictions'])
    
    # Save model
    torch.save(model.state_dict(), 'mnist/mnist_model.pth')
    print("\nModel saved to mnist/mnist_model.pth")


if __name__ == "__main__":
    train()
