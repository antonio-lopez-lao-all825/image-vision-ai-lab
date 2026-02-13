# Image Vision AI Lab

A deep learning laboratory for image recognition and computer vision experiments using PyTorch and YOLO. This project provides a modular framework for training, evaluating, and deploying neural networks on various computer vision tasks.

## Overview

This repository contains implementations of image classification and object detection models. Each module has its own directory with dedicated scripts.

## Project Structure

```
image-vision-ai-lab/
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── .gitignore
├── data/                    # Shared data directory
├── mnist/                   # MNIST digit recognition
│   ├── README.md
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── object_detection/        # Real-time object detection
│   ├── README.md
│   ├── detect_webcam.py     # Webcam detection
│   └── detect_image.py      # Image detection
└── [future_modules]/        # Additional experiments (coming soon)
```

## Modules

### Image Classification

| Dataset | Description | Classes | Status |
|---------|-------------|---------|--------|
| [MNIST](mnist/) | Handwritten digit recognition | 10 | ✅ Complete |
| CIFAR-10 | Object recognition | 10 | 🔜 Coming soon |
| Fashion-MNIST | Clothing classification | 10 | 🔜 Coming soon |

### Object Detection

| Module | Description | Model | Status |
|--------|-------------|-------|--------|
| [Object Detection](object_detection/) | Real-time webcam detection | YOLOv8 | ✅ Complete |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/image-vision-ai-lab.git
cd image-vision-ai-lab
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### MNIST Digit Recognition

```bash
python mnist/train.py       # Train model
python mnist/evaluate.py    # Evaluate on test set
```

### Real-time Object Detection (Webcam)

```bash
python object_detection/detect_webcam.py
```

Controls: `q` quit, `s` screenshot, `+/-` adjust confidence

### Train a model

```bash
python mnist/train.py
```

### Evaluate a model

```bash
python mnist/evaluate.py
```

### Make predictions

```bash
python mnist/predict.py <image_path>
```

## Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- CUDA (optional, for GPU acceleration)

### Core Dependencies

- **torch** - Deep learning framework
- **torchvision** - Computer vision utilities
- **numpy** - Numerical computing
- **matplotlib** - Visualization
- **seaborn** - Statistical visualization
- **scikit-learn** - ML metrics and utilities
- **tqdm** - Progress bars
- **Pillow** - Image processing

## Hardware Support

- **CPU**: Fully supported
- **CUDA GPU**: Automatically detected and used when available

## Adding New Datasets

To add a new dataset, create a new directory with the following structure:

```
new_dataset/
├── README.md       # Documentation
├── model.py        # Model architecture
├── train.py        # Training script
├── evaluate.py     # Evaluation script
└── predict.py      # Prediction script
```

Each dataset module should follow the same interface pattern for consistency.

## Results Summary

| Dataset | Model | Accuracy | F1 Score |
|---------|-------|----------|----------|
| MNIST | CNN | ~99% | ~0.99 |

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request
