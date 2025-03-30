# 🔥 Fire Detection Using YOLOv8

This project is a Proof-of-Concept (POC) system for detecting and locating fire regions in drone images using the YOLOv8 object detection model.

It was developed as part of the **AT2 Sprint 1 Artefact – Week 6 Deliverable** for our AI Studio course.

---

## 📁 Project Structure


---

## 📦 Dataset

We used a fire image dataset from **Kaggle**, which includes images of fire and corresponding bounding box annotations for object detection.

🔗 Dataset link (Kaggle):  
[FLAME - Fire Detection Dataset](https://www.kaggle.com/datasets/phylake1337/fire-dataset)

Only a small number of sample images and labels are included in this repo (`data/`) for demonstration purposes. The full dataset is used during model training.

---

## 🧠 Model and Training

We used **YOLOv8** from the [Ultralytics](https://github.com/ultralytics/ultralytics) library to train a fire detection model.

**Training Parameters**:
- Model: YOLOv8n
- Epochs: 15
- Batch size: 16
- Image size: 640x640
- Data format: YOLOv8 `.txt` annotation (class, x_center, y_center, width, height)

The training script is provided in `scripts/train.py`.

---

## 🚀 Inference (POC Demo)

We tested the trained model on several drone images.  
The model successfully detected fire regions and returned bounding boxes with confidence scores.

**Trained Model**:  
📍 `runs/detect/train5/weights/best.pt`

To run inference on an image:

```bash
python scripts/detect.py
