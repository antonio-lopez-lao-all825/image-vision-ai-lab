"""
Train YOLOv8 on a custom dataset.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def create_dataset_yaml(dataset_dir: str, class_names: list) -> str:
    """
    Create dataset.yaml configuration file for YOLOv8 training.
    
    Args:
        dataset_dir: Path to dataset directory
        class_names: List of class names
    
    Returns:
        Path to created yaml file
    """
    dataset_path = Path(dataset_dir).resolve()
    yaml_path = dataset_path / "dataset.yaml"
    
    yaml_content = f"""# Custom Dataset Configuration
path: {dataset_path}
train: images/train
val: images/val

# Classes
names:
"""
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Created dataset config: {yaml_path}")
    return str(yaml_path)


def train_custom_model(
    dataset_yaml: str,
    model_name: str = "yolov8n.pt",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    project_name: str = "custom_detector"
):
    """
    Train YOLOv8 on custom dataset.
    
    Args:
        dataset_yaml: Path to dataset.yaml file
        model_name: Base model to fine-tune
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image size for training
        project_name: Name for the training run
    """
    print(f"\n{'='*50}")
    print("CUSTOM YOLOv8 TRAINING")
    print(f"{'='*50}")
    print(f"Dataset: {dataset_yaml}")
    print(f"Base model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size}")
    print(f"{'='*50}\n")
    
    # Load pretrained model
    model = YOLO(model_name)
    
    # Train
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=f"object_detection/runs/{project_name}",
        name="train",
        patience=20,  # Early stopping patience
        save=True,
        plots=True,
        verbose=True
    )
    
    print(f"\n{'='*50}")
    print("TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"Best model saved to: object_detection/runs/{project_name}/train/weights/best.pt")
    print(f"\nTo use your trained model:")
    print(f"  python object_detection/detect_webcam.py --model object_detection/runs/{project_name}/train/weights/best.pt")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on custom dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset.yaml file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="Base model to fine-tune (default: yolov8n.pt)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image size (default: 640)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="custom_detector",
        help="Project name for this training run"
    )
    
    args = parser.parse_args()
    
    train_custom_model(
        dataset_yaml=args.dataset,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        project_name=args.name
    )


if __name__ == "__main__":
    main()
