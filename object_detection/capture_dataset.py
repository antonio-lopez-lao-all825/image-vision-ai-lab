"""
Capture images from webcam to build a custom dataset.
Press 's' to save a frame, 'q' to quit.
"""
import cv2
import argparse
import time
from pathlib import Path


def capture_dataset(
    object_name: str,
    camera_id: int = 0,
    output_dir: str = "object_detection/custom_dataset"
):
    """
    Capture images from webcam for dataset creation.
    
    Args:
        object_name: Name of the object to detect
        camera_id: Camera device ID
        output_dir: Base directory for the dataset
    """
    # Create directory structure
    base_dir = Path(output_dir)
    images_dir = base_dir / "images" / object_name
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {camera_id}")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Count existing images
    existing = list(images_dir.glob("*.jpg"))
    count = len(existing)
    
    print(f"\n{'='*50}")
    print(f"DATASET CAPTURE: {object_name}")
    print(f"{'='*50}")
    print(f"Output directory: {images_dir}")
    print(f"Existing images: {count}")
    print(f"\nControls:")
    print(f"  s - Save current frame")
    print(f"  q - Quit")
    print(f"\nTips for good dataset:")
    print(f"  - Vary distance (close, medium, far)")
    print(f"  - Vary angles (front, side, tilted)")
    print(f"  - Vary lighting (bright, dim)")
    print(f"  - Vary backgrounds")
    print(f"  - Include partial occlusions")
    print(f"{'='*50}\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display info on frame
        display_frame = frame.copy()
        cv2.putText(
            display_frame,
            f"Object: {object_name} | Captured: {count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        cv2.putText(
            display_frame,
            "Press 's' to capture, 'q' to quit",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )
        
        cv2.imshow(f"Dataset Capture - {object_name}", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{object_name}_{timestamp}_{count:04d}.jpg"
            filepath = images_dir / filename
            cv2.imwrite(str(filepath), frame)
            count += 1
            print(f"Saved: {filename} (Total: {count})")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nCapture complete!")
    print(f"Total images: {count}")
    print(f"Location: {images_dir}")
    print(f"\nNext step: Label images using one of these tools:")
    print(f"  1. LabelImg: pip install labelImg && labelImg")
    print(f"  2. Roboflow: https://roboflow.com (free tier available)")
    print(f"  3. CVAT: https://cvat.ai")


def main():
    parser = argparse.ArgumentParser(description="Capture images for custom dataset")
    parser.add_argument(
        "object_name",
        type=str,
        help="Name of the object to detect (e.g., 'phone', 'cup', 'keys')"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="object_detection/custom_dataset",
        help="Output directory for dataset"
    )
    
    args = parser.parse_args()
    
    capture_dataset(
        object_name=args.object_name,
        camera_id=args.camera,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
