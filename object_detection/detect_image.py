"""
Object detection on static images using YOLOv8.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def detect_image(
    image_path: str,
    model_name: str = "yolov8n.pt",
    confidence: float = 0.5,
    save_output: bool = True
):
    """
    Run object detection on a static image.
    
    Args:
        image_path: Path to input image
        model_name: YOLOv8 model variant
        confidence: Minimum confidence threshold
        save_output: Whether to save the annotated image
    """
    # Validate input
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return None
    
    # Load model
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)
    
    # Run detection
    print(f"Detecting objects in: {image_path}")
    results = model(str(image_path), conf=confidence)
    
    # Print results
    print(f"\nDetected {len(results[0].boxes)} objects:")
    
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        conf = float(box.conf[0])
        print(f"  - {class_name}: {conf:.2f}")
    
    # Save output
    if save_output:
        output_dir = Path("object_detection/output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"detected_{image_path.name}"
        
        # Save annotated image
        results[0].save(str(output_path))
        print(f"\nResult saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Object detection on images with YOLOv8")
    parser.add_argument(
        "image",
        type=str,
        help="Path to input image"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="YOLOv8 model variant"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output image"
    )
    
    args = parser.parse_args()
    
    detect_image(
        image_path=args.image,
        model_name=args.model,
        confidence=args.confidence,
        save_output=not args.no_save
    )


if __name__ == "__main__":
    main()
