from ultralytics import YOLO
import sys

# Load the trained model
model = YOLO('runs/detect/train5/weights/best.pt')

# Run inference on an image (you can change the path below)
results = model('data/sample.jpg')

# Save the detection result (with bounding boxes)
results.save()  # Output will be saved to runs/detect/predict/
