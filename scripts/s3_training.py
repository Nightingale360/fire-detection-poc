# s3_training.py
import os
import argparse
from clearml import Task, StorageManager
from ultralytics import YOLO

# 1) Parse CLI args for ClearMl
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_task_id", required=True,
    help="Task ID from Step 2 (yolo_dataset)"
)
parser.add_argument(
    "--model_arch", default="yolo11n.pt",
    help="YOLO 11n architecture"
)
parser.add_argument(
    "--epochs", type=int, default=50,
    help="Number of epochs"
)
parser.add_argument(
    "--batch", type=int, default=50,
    help="Number in a batch"
)
parser.add_argument(
    "--imgsz", type=int, default=640,
    help="Image size"
)
args = parser.parse_args()

# 2) Init ClearML Task
task = Task.init(
    project_name="AlphaFirewatch",
    task_name="Step 3: Fire and Smoke Detection Training"
)
task.connect(vars(args))

# 3) offload remotely
task.execute_remotely()

# 4) Fetch the prepped YOLO dataset
local_data = StorageManager.get_local_copy(
    task_id=args.dataset_task_id,
    artifact_name="yolo_dataset"
)

data_yaml = os.path.join(local_data, "data.yaml")
if not os.path.isfile(data_yaml):
    raise FileNotFoundError(f"data.yaml not found in {local_data}")
print(f"✅ Loaded data.yaml from {data_yaml}")

# 5) Train with Ultralytics
model = YOLO(args.model_arch)
# results = model.train(
#     data=data_yaml,
#     epochs=args.epochs,
#     imgsz=args.imgsz,
#     project="AlphaFirewatch",
#     name="yolov11_training",
#     exist_ok=True
# )

val_results = model.val( data=data_yaml, imgsz=args.imgsz, batch=16,verbose=True )

map50 = float(val_results.box.map50)

task.get_logger().report_scalar( title="train/metrics", series="val/mAP50", value=map50, iteration=0)

# 6) Upload best weights
run_dir = model.run_dir  # e.g. runs/AlphaFirewatch/yolov8_training
best_pt = os.path.join(run_dir, "weights", "best.pt")
if os.path.isfile(best_pt):
    task.upload_artifact(name="best_weights", artifact_object=best_pt)
    print(f"✅ Uploaded best weights: {best_pt}")
else:
    print("⚠️  best.pt not found")

# 7) Emit this Task’s ID for CI chaining
print(task.id)
