import cv2
from ultralytics import YOLO

def process_video(input_path, output_path, model_path):
    """
    Loads a trained YOLOv8 model, runs inference on a video file,
    and saves the output with bounding box annotations.
    """
    
    # 1. Load the custom trained model weights
    print(f"Loading model from: {model_path}...")
    model = YOLO(model_path)

    # 2. Initialize video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Retrieve video properties for the output writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Configure MP4 codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Starting inference process...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLOv8 inference
        # conf=0.5: Only detect objects with >50% confidence
        results = model.predict(frame, conf=0.5, verbose=False)

        # Overlay detection bounding boxes on the frame
        annotated_frame = results[0].plot()

        # Write the processed frame to the output video
        out.write(annotated_frame)

    # 3. Release resources
    cap.release()
    out.release()
    print(f"Done. Annotated video saved to: {output_path}")

if __name__ == "__main__":
    # Define file paths
    INPUT_VIDEO = "input_cows_video.mp4"
    OUTPUT_VIDEO = "output_detected.mp4"
    MODEL_WEIGHTS = "best.pt"

    process_video(INPUT_VIDEO, OUTPUT_VIDEO, MODEL_WEIGHTS)