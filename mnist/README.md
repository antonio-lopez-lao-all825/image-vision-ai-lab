# MNIST Digit Recognition with PyTorch

Handwritten digit recognition using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## Description

This project implements a digit classifier (0-9) using PyTorch. The model is a CNN that achieves ~99% accuracy on the MNIST test set.

## Project Structure

```
mnist/
├── model.py          # Neural network architecture (MNISTNet)
├── train.py          # Training script
├── evaluate.py       # Full evaluation with metrics and visualizations
├── predict.py        # Prediction on new images
├── images/           # Test images
├── data/             # MNIST dataset (downloaded automatically)
└── mnist_model.pth   # Trained model (generated after training)
```

## Model Architecture

```
MNISTNet (CNN)
├── Conv2d(1, 32, kernel=3, padding=1) + ReLU + MaxPool2d(2)   → 28x28 → 14x14
├── Conv2d(32, 64, kernel=3, padding=1) + ReLU + MaxPool2d(2)  → 14x14 → 7x7
├── Dropout(0.25)
├── Flatten                                                     → 64*7*7 = 3136
├── Linear(3136, 128) + ReLU
├── Dropout(0.5)
└── Linear(128, 10)                                            → 10 classes
```

## Installation

```bash
# From the project root
pip install torch torchvision pillow matplotlib seaborn scikit-learn tqdm numpy
```

## Usage

### 1. Train the model

```bash
python mnist/train.py
```

Training:
- Automatically downloads MNIST (60,000 training images)
- Trains for 10 epochs
- Saves model to `mnist/mnist_model.pth`

### 2. Evaluate the model

```bash
python mnist/evaluate.py
```

Generates:
- Detailed metrics (Accuracy, F1, Precision, Recall)
- `test_confusion_matrix.png` - Confusion matrix
- `test_pred_vs_true.png` - Predictions vs actual scatter plot
- `test_error_analysis.png` - Worst errors analysis

### 3. Predict on an image

```bash
python mnist/predict.py <image_path>
```

Example:
```bash
python mnist/predict.py mnist/images/two.jpeg
# Output: Predicted digit: 2
```

## MNIST Dataset

| Set | Images | Purpose |
|-----|--------|--------|
| Train | 60,000 | Model training |
| Test | 10,000 | Evaluation (never seen during training) |

Each image is 28x28 pixels in grayscale.

## Results

After 10 epochs of training:

| Metric | Value |
|--------|-------|
| Accuracy | ~99% |
| F1 Score | ~0.99 |
| Precision | ~0.99 |
| Recall | ~0.99 |

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 64 |
| Learning rate | 0.001 |
| Epochs | 10 |
| Optimizer | AdamW |
| Loss | CrossEntropyLoss |

## Generated Visualizations

After running `evaluate.py`:

- **Confusion Matrix**: Shows errors between classes
- **Scatter Plot**: Prediction vs actual value with accuracy per digit
- **Error Analysis**: The 16 incorrect predictions with highest confidence

## Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.2.0
- numpy >= 1.24.0
- Pillow >= 9.0.0
- tqdm >= 4.65.0
