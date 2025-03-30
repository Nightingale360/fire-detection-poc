from ultralytics import YOLO

# Load a pre-trained YOLOv8 model (nano version is used here; you can change to 'yolov8s.pt' or others)
model = YOLO('yolov8n.pt')

# Start training
model.train(
    data='data.yaml',      # Path to your dataset config file
    epochs=15,             # Number of training epochs
    imgsz=640,             # Input image size
    batch=16,              # Batch size
    project='runs',        # Output directory
    name='detect',         # Name of this training run
    exist_ok=True          # Overwrite if the folder already exists
)
