# Real-Time Livestock Detection System (YOLOv8)

## Project Overview
This repository implements a computer vision pipeline for **automated livestock monitoring**. Leveraging the YOLOv8 architecture, the system performs real-time object detection on video feeds to identify cattle.

This project serves as a Proof of Concept (PoC) for Precision Livestock Farming (PLF) solutions, addressing needs such as automated headcounts and visual health monitoring via edge-deployable inference engines.

## Technical Architecture

### 1. Model Architecture
* **Model:** YOLOv8-Nano (Optimized for Edge Computing).
* **Parameters:** ~3.2 Million.
* **Justification:** Selected for its superior latency/accuracy trade-off, allowing for high-FPS processing on resource-constrained devices (e.g., NVIDIA Jetson, Raspberry Pi) without significant loss in mAP compared to larger models.

### 2. Training Pipeline
* **Framework:** PyTorch & Ultralytics.
* **Strategy:** Transfer Learning on COCO-pretrained weights.
* **Data:** Fine-tuned on a specialized livestock dataset (Roboflow) for 15 epochs to adapt feature extraction layers to specific animal features.

## Results

### Static Inference
The model demonstrates robust detection capabilities on individual frames with high confidence scores.

![Static Result](result_image.jfif)

### Video Inference
The pipeline maintains consistent object tracking across frames, handling movement and partial occlusion effectively.

[🎥 **Download/View Processed Video (output_detected.mp4)**](output_detected.mp4)
*(Note: Click the link above to access the full video artifact hosted in this repository).*

## Installation & Usage

### Prerequisites
* Python 3.8+
* `pip` package manager

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/livestock-detection-yolo.git](https://github.com/YOUR_USERNAME/livestock-detection-yolo.git)
    cd livestock-detection-yolo
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Execution
To run inference on a new video file:

1.  Place your video file (e.g., `input.mp4`) in the project root.
2.  Edit `src/track_livestock.py` to point to your video filename (or run via CLI if configured).
3.  Run the script:
    ```bash
    python src/track_livestock.py
    ```

## Repository Structure
* `src/`: Source code for training (Jupyter Notebook) and inference (Python script).
* `best.pt`: Serialized PyTorch model weights (Fine-tuned).
* `requirements.txt`: Python dependencies.

---
*AgTech Computer Vision Portfolio Project.*