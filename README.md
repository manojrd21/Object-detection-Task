# Real-Time Object Detection System

## Overview
This project implements a real-time object detection and tracking system using a pre-trained YOLOv8 model and DeepSORT tracker. It processes video inputs, detects objects with bounding boxes, tracks unique objects across frames, and generates analytics reports.

---

## Installation Instructions

1. Clone the repository to your local machine.

2. Create and activate a Python virtual environment (.venv) using Python 3.10:

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
```

3. Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

(Note: The packages are pre-installed in the `.venv` used here, so if you activate it, the installation step can be skipped.)

---

## System Requirements

- Python 3.8 or higher (recommended Python 3.10)
- Windows or Linux or MacOS system
- GPU acceleration is optional but can speed up processing
- At least 4GB of RAM recommended

---

## Usage Instructions

1. Update the following paths in the `task_testing.py` before running:

- **Input video file path**  
  Example: `D:\Object detection Task\Testing_Videos\Test5.mp4`

- **Model weights path (best.pt)**  
  Example: `D:\Object detection Task\best.pt`

- **Output directory path for saving results**  
  Example: `D:\Object detection Task\Outputs`

2. Activate the virtual environment:
```bash
.venv\Scripts\activate
```

3. Run the detection and tracking script:
```bash
python task_testing.py --video_path "D:\Object detection Task\Testing_Videos\Test5.mp4" --weights_path "D:\Object detection Task\best.pt" --conf_threshold 0.4
```

4. After processing, output video and analytics report will be available in the specified output directory.

---

## Expected Input and Output

### Inputs

- Video file path (`.mp4`, `.avi`, etc.) or webcam device ID (default 0), passed via command-line argument

- Model weights path (`best.pt` pre-trained YOLOv8 weights)

- Confidence threshold (float, default 0.4) to filter detections

### Outputs

- The processed video file (output_tracked.mp4) displays bounding boxes with detected classes for each frame, saved in the output folder.

- The analytics report (text file) summarizing:
    - Total number of frames processed
    - Detected object classes and their counts
    - Average processing time per frame
    - Confidence score statistics (min and max)

- Both files are saved in the output folder specified in the code (D:\Object detection Task\Outputs or your relevant path).


---

## Requirements

This project depends on the following Python packages (also in `requirements.txt`):
```bash
ultralytics>=8.3.220
opencv-python==4.11.0.86
opencv-python-headless==4.11.0.86
norfair==2.2.0
deep-sort-realtime==1.3.2
numpy>=1.26.0
torch>=2.3.0
torchvision>=0.18.0
```

---

## Notes

- Do not modify or retrain the model; use the pre-trained weights provided.

- The system is designed to run offline once the environment is setup.

- Test with multiple video formats and webcam where applicable.

- If you encounter issues, check your environment activation and dependency versions.

---
