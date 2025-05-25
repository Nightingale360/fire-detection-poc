# 🔥 Fire Detection Using YOLOv11

This project is a Proof-of-Concept (POC) system for detecting and localizing fire regions in drone images using the **YOLOv11** object detection model.

It was developed as part of the **AT4 Sprint 3 Artefact – Week 13 Deliverable** for our AI Studio course.

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

We trained the fire detection model using **YOLOv11** (from [Ultralytics](https://github.com/ultralytics/ultralytics)).

### 🔧 Training Parameters:
- Model: `yolol11.pt` (YOLOv11 pretrained weights)
- Epochs: `50`
- Batch size: `128`
- Image size: `640 × 640`
- Dataset: FLAME
- Label format: YOLO format `.txt` — `[class_id x_center y_center width height]` (normalized)


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
