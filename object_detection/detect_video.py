"""
Object detection on video files using YOLOv8.
Processes video and saves annotated output.
"""
import cv2
import argparse
import time
from pathlib import Path
from ultralytics import YOLO


def detect_video(
    video_path: str,
    model_name: str = "yolov8n.pt",
    confidence: float = 0.5,
    output_dir: str = "object_detection/output",
    show_preview: bool = True,
    save_output: bool = True
):
    """
    Run object detection on a video file.
    
    Args:
        video_path: Path to input video
        model_name: YOLOv8 model variant or custom model path
        confidence: Minimum confidence threshold
        output_dir: Directory to save output video
        show_preview: Whether to show preview while processing
        save_output: Whether to save the annotated video
    """
    # Validate input
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}")
        return
    
    # Load model
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"\nVideo: {video_path.name}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Duration: {duration:.1f}s ({total_frames} frames)")
    print(f"Confidence threshold: {confidence}")
    
    # Setup output video writer
    output_path = None
    writer = None
    if save_output:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"detected_{video_path.name}"
        
        # Use mp4v codec for .mp4 files
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"\nProcessing video...")
    if show_preview:
        print("Press 'q' to stop, 'p' to pause/resume")
    
    frame_count = 0
    start_time = time.time()
    paused = False
    detection_counts = {}
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            results = model(frame, conf=confidence, verbose=False)
            
            # Count detections
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
            
            # Draw results
            annotated_frame = results[0].plot()
            
            # Add progress info
            progress = (frame_count / total_frames) * 100
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_count}/{total_frames} ({progress:.1f}%)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Save frame
            if writer is not None:
                writer.write(annotated_frame)
            
            # Show preview
            if show_preview:
                # Resize for display if too large
                display_frame = annotated_frame
                if width > 1280:
                    scale = 1280 / width
                    display_frame = cv2.resize(annotated_frame, None, fx=scale, fy=scale)
                
                cv2.imshow("Video Detection - Press 'q' to stop", display_frame)
        
        # Handle key presses
        if show_preview:
            key = cv2.waitKey(1 if not paused else 100) & 0xFF
            if key == ord('q'):
                print("\nStopped by user")
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
    
    # Cleanup
    elapsed_time = time.time() - start_time
    processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print(f"\n{'='*50}")
    print("DETECTION SUMMARY")
    print(f"{'='*50}")
    print(f"Processed: {frame_count}/{total_frames} frames")
    print(f"Processing time: {elapsed_time:.1f}s")
    print(f"Processing speed: {processing_fps:.1f} FPS")
    
    if detection_counts:
        print(f"\nObjects detected (total counts across all frames):")
        for obj, count in sorted(detection_counts.items(), key=lambda x: -x[1]):
            print(f"  {obj}: {count}")
    
    if output_path:
        print(f"\nOutput saved to: {output_path}")
    
    return detection_counts


def main():
    parser = argparse.ArgumentParser(description="Object detection on video files")
    parser.add_argument(
        "video",
        type=str,
        help="Path to input video file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLOv8 model (yolov8n/s/m/l/x.pt) or path to custom model"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="object_detection/output",
        help="Output directory for processed video"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable preview window (faster processing)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save output video"
    )
    
    args = parser.parse_args()
    
    detect_video(
        video_path=args.video,
        model_name=args.model,
        confidence=args.confidence,
        output_dir=args.output,
        show_preview=not args.no_preview,
        save_output=not args.no_save
    )


if __name__ == "__main__":
    main()
