# Real-time Object Detection with YOLOv8

Real-time object detection using your laptop's webcam with YOLOv8. Supports both pretrained COCO models (80 classes) and custom-trained models.

## Features

- Real-time webcam detection
- 80 pretrained object classes (person, car, dog, chair, cell phone, etc.)
- Custom object training support
- FPS counter display
- Configurable confidence threshold
- Screenshot capture functionality

## Pretrained Classes (COCO)

The pretrained model can detect 80 different objects including:

**People & Animals**: person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Vehicles**: bicycle, car, motorcycle, airplane, bus, train, truck, boat

**Indoor Objects**: chair, couch, bed, dining table, toilet, tv, laptop, mouse, keyboard, **cell phone**, book

**Food**: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

**Sports**: frisbee, skis, snowboard, sports ball, kite, baseball bat, tennis racket

## Installation

```bash
pip install ultralytics opencv-python labelImg
```

## Usage

### Basic webcam detection (pretrained)

```bash
python object_detection/detect_webcam.py
```

### With custom settings

```bash
python object_detection/detect_webcam.py --confidence 0.5 --camera 0
```

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot |
| `+` | Increase confidence threshold |
| `-` | Decrease confidence threshold |

---

## Custom Object Training

Train YOLOv8 to detect your own objects.

### Step 1: Capture Images

```bash
python object_detection/capture_dataset.py my_object
```

- Press `s` to save frames
- Capture 50-300+ images with varied:
  - Distances (close, medium, far)
  - Angles (front, side, tilted)
  - Lighting conditions
  - Backgrounds

### Step 2: Label Images

Option A - LabelImg (local):
```bash
pip install labelImg
labelImg object_detection/custom_dataset/images/my_object
```
- Select YOLO format
- Draw bounding boxes around your object
- Save labels

Option B - Roboflow (web):
1. Go to https://roboflow.com
2. Upload images
3. Label with web interface
4. Export in YOLOv8 format

### Step 3: Organize Dataset

```
custom_dataset/
├── dataset.yaml
├── images/
│   ├── train/    # 80% of images
│   └── val/      # 20% of images
└── labels/
    ├── train/    # Corresponding .txt files
    └── val/
```

### Step 4: Create dataset.yaml

```yaml
path: /full/path/to/custom_dataset
train: images/train
val: images/val

names:
  0: my_object
```

### Step 5: Train

```bash
python object_detection/train_custom.py --dataset path/to/dataset.yaml --epochs 100
```

### Step 6: Use Your Model

```bash
python object_detection/detect_webcam.py --model object_detection/runs/custom_detector/train/weights/best.pt
```

---

## Files

| File | Description |
|------|-------------|
| `detect_webcam.py` | Real-time webcam detection |
| `detect_image.py` | Detection on static images |
| `capture_dataset.py` | Capture images for custom dataset |
| `train_custom.py` | Train custom YOLOv8 model |

## Requirements

- Python 3.8+
- Webcam
- ultralytics >= 8.0.0
- opencv-python >= 4.8.0
- labelImg (for labeling)
