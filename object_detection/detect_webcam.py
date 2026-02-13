"""
Real-time object detection using webcam with YOLOv8.
Detects 80 classes from COCO dataset.
"""
import cv2
import argparse
import time
from pathlib import Path
from ultralytics import YOLO


def detect_webcam(
    model_name: str = "yolov8n.pt",
    camera_id: int = 0,
    confidence: float = 0.5,
    show_fps: bool = True
):
    """
    Run real-time object detection on webcam feed.
    
    Args:
        model_name: YOLOv8 model variant (yolov8n/s/m/l/x.pt)
        camera_id: Camera device ID (0 for default webcam)
        confidence: Minimum confidence threshold for detections
        show_fps: Whether to display FPS counter
    """
    # Load model (downloads automatically if not present)
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)
    
    # Open webcam
    print(f"Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {camera_id}")
        print("Try: python detect_webcam.py --camera 1")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    
    # Create screenshots directory
    screenshots_dir = Path("object_detection/screenshots")
    screenshots_dir.mkdir(exist_ok=True)
    
    # FPS calculation
    prev_time = time.time()
    fps = 0
    
    print("\nControls:")
    print("  q - Quit")
    print("  s - Save screenshot")
    print("  + - Increase confidence threshold")
    print("  - - Decrease confidence threshold")
    print(f"\nCurrent confidence threshold: {confidence}")
    print("Starting detection...\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame")
            break
        
        # Run YOLOv8 detection
        results = model(frame, conf=confidence, verbose=False)
        
        # Draw results on frame
        annotated_frame = results[0].plot()
        
        # Calculate and display FPS
        if show_fps:
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            cv2.putText(
                annotated_frame, 
                f"FPS: {fps:.1f}", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
        
        # Display confidence threshold
        cv2.putText(
            annotated_frame, 
            f"Conf: {confidence:.2f}", 
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        # Count detections
        num_detections = len(results[0].boxes)
        cv2.putText(
            annotated_frame, 
            f"Objects: {num_detections}", 
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        # Show frame
        cv2.imshow("YOLOv8 Object Detection - Press 'q' to quit", annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = screenshots_dir / f"detection_{timestamp}.jpg"
            cv2.imwrite(str(filepath), annotated_frame)
            print(f"Screenshot saved: {filepath}")
        elif key == ord('+') or key == ord('='):
            confidence = min(0.95, confidence + 0.05)
            print(f"Confidence threshold: {confidence:.2f}")
        elif key == ord('-'):
            confidence = max(0.05, confidence - 0.05)
            print(f"Confidence threshold: {confidence:.2f}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")


def main():
    parser = argparse.ArgumentParser(description="Real-time object detection with YOLOv8")
    parser.add_argument(
        "--model", 
        type=str, 
        default="yolov8n.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="YOLOv8 model variant (n=nano, s=small, m=medium, l=large, x=extra large)"
    )
    parser.add_argument(
        "--camera", 
        type=int, 
        default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--confidence", 
        type=float, 
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--no-fps", 
        action="store_true",
        help="Hide FPS counter"
    )
    
    args = parser.parse_args()
    
    detect_webcam(
        model_name=args.model,
        camera_id=args.camera,
        confidence=args.confidence,
        show_fps=not args.no_fps
    )


if __name__ == "__main__":
    main()
