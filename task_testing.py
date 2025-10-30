# Importing required libraries
import os
import cv2
import argparse
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import Counter
from time import time


# Load YOLOv8 model with the given weights
def load_model(weights_path='best.pt'):
    """Load YOLOv8 model from custom weights."""
    model = YOLO(weights_path)
    return model


# Perform object detection on a single frame
def detect_objects(model, frame, conf_threshold=0.4):
    """
    Run detection on frame; returns list of detection dicts each containing:
    bounding box coordinates, confidence score, and class name.
    """
    results = model(frame)[0]
    detections = []
    class_names = model.names
    for box, conf, cls in zip(results.boxes.xyxy.cpu().numpy(),
                              results.boxes.conf.cpu().numpy(),
                              results.boxes.cls.cpu().numpy()):
        if conf >= conf_threshold:
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[int(cls)] if int(cls) in class_names else str(int(cls))
            detections.append({'bbox': (x1, y1, x2, y2),
                               'conf': float(conf),
                               'class_name': class_name})
    return detections


# Draw bounding boxes and labels on frame for detected objects
def draw_detections(frame, detections):
    """
    For each detected object, draws a rectangle and label on the frame
    indicating class and confidence.
    """
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['conf']
        class_name = det['class_name']

        # Draw bounding box rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {conf:.2f}"

        # Calculate text size and background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (255, 255, 255), -1)

        # Put class label text
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return frame


# Convert detections to DeepSORT input format
def convert_detections_to_deepsort_format(detections):
    """
    Convert list of detection dicts into format expected by DeepSORT: 
    a list of tuples containing bounding box [x, y, w, h] and confidence score.
    Also returns list of class names matching detections.
    """
    det_for_tracker = []
    classes_for_tracker = []
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['conf']
        class_name = det['class_name']
        w = x2 - x1
        h = y2 - y1
        if w > 0 and h > 0:
            det_for_tracker.append(([x1, y1, w, h], conf))
            classes_for_tracker.append(class_name)
    return det_for_tracker, classes_for_tracker


# Save the analytics report to a text file
def save_analytics_report(report_path, total_frames, class_counts, avg_time_per_frame, min_conf, max_conf):
    """
    Write analytics summary with frames processed, class counts, 
    processing time and confidence stats into a text file.
    """
    with open(report_path, 'w') as f:
        f.write("=== Analytics Report ===\n")
        f.write(f"Total frames processed: {total_frames}\n")
        f.write(f"Complete list of object classes detected: {list(class_counts.keys())}\n")
        f.write("Count of each unique object type detected:\n")
        for cls, cnt in class_counts.items():
            f.write(f"  {cls}: {cnt}\n")
        f.write(f"Average processing time per frame: {avg_time_per_frame:.4f} seconds\n")
        if min_conf is not None and max_conf is not None:
            f.write(f"Minimum confidence score observed: {min_conf:.4f}\n")
            f.write(f"Maximum confidence score observed: {max_conf:.4f}\n")
        else:
            f.write("No confidence scores recorded.\n")


# Main function for running detection, tracking, output saving, and analytics reporting
def run_object_detection_and_tracking(video_path, weights_path='best.pt', conf_threshold=0.4):
    """
    Processes an input video for object detection and tracking, saves output video and analytics.
    
    Arguments:
    - video_path: path to input video file
    - weights_path: path to YOLOv8 weights file
    - conf_threshold: minimum confidence threshold for detections
    
    Outputs:
    - Output video saved with bounding boxes annotated
    - Analytics report saved as a text file
    """
    output_dir = "D:\\Object detection Task\\Outputs"  # Path where detected output will be stored
    output_video_path = os.path.join(output_dir, "output_tracked.mp4")
    analytics_report_path = os.path.join(output_dir, "analytics_report.txt")

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Load detection model and tracker
    model = load_model(weights_path)
    tracker = DeepSort()

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")

    # Setup video writer for output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Initialize variables for analytics
    total_frames = 0
    unique_track_ids = set()
    track_id_to_class = dict()
    confidence_scores = []

    start_time = time()

    # Process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        # Detect objects on frame
        detections = detect_objects(model, frame, conf_threshold=conf_threshold)
        # Convert detections for tracking
        det_for_tracker, classes_for_tracker = convert_detections_to_deepsort_format(detections)

        # Update tracker with current detections
        tracks = tracker.update_tracks(det_for_tracker, frame=frame)

        # Assign classes to tracks and keep unique track IDs
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            unique_track_ids.add(track_id)

            # Determine which detection box is associated with track by IoU
            track_class = 'Unknown'
            for i, (bbox_det, conf) in enumerate(det_for_tracker):
                dx1, dy1, dw, dh = bbox_det
                dx2 = dx1 + dw
                dy2 = dy1 + dh
                bbox = track.to_tlbr()
                iou_x1 = max(bbox[0], dx1)
                iou_y1 = max(bbox[1], dy1)
                iou_x2 = min(bbox[2], dx2)
                iou_y2 = min(bbox[3], dy2)
                inter_area = max(0, iou_x2 - iou_x1) * max(0, iou_y2 - iou_y1)
                box1_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                box2_area = dw * dh
                iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
                if iou > 0.5:
                    track_class = classes_for_tracker[i]
                    break
            # Record the class per unique track ID once
            if track_id not in track_id_to_class:
                track_id_to_class[track_id] = track_class

        # Collect confidence scores for analytics
        for det in detections:
            confidence_scores.append(det['conf'])

        # Draw detection boxes on frame
        if detections:
            frame = draw_detections(frame, detections)

        # Write processed frame to output video file
        out.write(frame)

    # Release video resources
    cap.release()
    out.release()

    # Timing and analytics calculations
    total_time = time() - start_time
    avg_time_per_frame = total_time / total_frames if total_frames else 0

    # Count unique object types detected
    class_counts = Counter(track_id_to_class.values())

    # Print output video path
    print(f"Output video saved at: {output_video_path}")

    # Print analytics summary report
    print("\n=== Analytics Report ===")
    print(f"Total frames processed: {total_frames}")
    print(f"Complete list of object classes detected: {list(class_counts.keys())}")
    print("Count of each unique object type detected:")
    for cls, cnt in class_counts.items():
        print(f"  {cls}: {cnt}")
    print(f"Average processing time per frame: {avg_time_per_frame:.4f} seconds")
    if confidence_scores:
        min_conf = min(confidence_scores)
        max_conf = max(confidence_scores)
        print(f"Minimum confidence score observed: {min_conf:.4f}")
        print(f"Maximum confidence score observed: {max_conf:.4f}")
    else:
        print("No confidence scores recorded.")

    # Save analytics report to file
    save_analytics_report(analytics_report_path, total_frames, class_counts, avg_time_per_frame,
                          min_conf if confidence_scores else None,
                          max_conf if confidence_scores else None)


# Main function for CLI argument parsing and execution
def main():
    parser = argparse.ArgumentParser(description="Real-Time Object Detection and Tracking")
    parser.add_argument("--video_path", type=str, default="D:\\Object detection Task\\Testing_Videos\\Test5.mp4",  ## Path for input video
                        help="Path to input video file")
    parser.add_argument("--weights_path", type=str, default=r"D:\Object detection Task\best.pt",  # Path for model weight(best.pt)
                        help="Path to YOLOv8 model weights")
    parser.add_argument("--conf_threshold", type=float, default=0.4,
                        help="Confidence threshold for detections")

    args = parser.parse_args(args=[])
    run_object_detection_and_tracking(args.video_path, args.weights_path, args.conf_threshold)


if __name__ == "__main__":
    main()
