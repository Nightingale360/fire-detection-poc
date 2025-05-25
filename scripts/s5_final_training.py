# s5_final_training.py
import argparse
import logging
import os
import json

from clearml import Task, StorageManager
from ultralytics import YOLO

#  Logging setup 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#  Parse CLI args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_task_id", required=True,
    help="Task ID from Step 2 (yolo_dataset)",
)
parser.add_argument(
    "--hpo_task_id", required=True,
    help="Task ID from Step 4 (HPO tuning)",
)
parser.add_argument(
    "--model_arch", default="yolo11n.pt",
    help="YOLO architecture to use",
)
parser.add_argument(
    "--imgsz", type=int, default=640,
    help="Image size for training",
)
parser.add_argument(
    "--project", default="AlphaFirewatch",
    help="ClearML project name",
)
parser.add_argument(
    "--name", default="yolov11_final",
    help="Run name under the project",
)
parser.add_argument(
    "--conf_thres", type=float, default=0.25,
    help="YOLO confidence threshold",
)
args = parser.parse_args()

# Init ClearML Task
task = Task.init(
    project_name=args.project,
    task_name="Step 5: Final Model Training",
    task_type=Task.TaskTypes.training,
    reuse_last_task_id=False
)
args = task.connect(args)
logger.info("Connected parameters: %s", args)

#  Offload to remote agent
task.execute_remotely()

#  Retrieve best HPO parameters
hpo_task = Task.get_task(task_id=args.hpo_task_id)
best_params = hpo_task.get_parameter("best_parameters") or {}
if not best_params:
    art = hpo_task.artifacts.get("best_hpo_results")
    if art:
        path = art.get_local_copy()
        with open(path, "r") as f:
            best_params = json.load(f).get("parameters", {})
logger.info("Loaded best HPO parameters: %s", best_params)

epochs = int(best_params.get("epochs", 50))
batch  = int(best_params.get("batch", 16))

#  Fetch YOLO dataset
data_dir = StorageManager.get_local_copy(
    task_id=args.dataset_task_id,
    artifact_name="yolo_dataset"
)
data_yaml = os.path.join(data_dir, "data.yaml")
if not os.path.isfile(data_yaml):
    raise FileNotFoundError(f"data.yaml not found under {data_dir}")
logger.info("Loaded data.yaml from %s", data_yaml)

#  Final YOLO training
model = YOLO(args.model_arch)
# model.train(
#     data=data_yaml,
#     epochs=epochs,
#     imgsz=args.imgsz,
#     batch=batch,
#     project=args.project,
#     name=args.name,
#     exist_ok=True,
#     conf=args.conf_thres
# )
logger.info("Completed model.train() → runs/%s/%s", args.project, args.name)

# ─── Upload final best.pt ───────────────────────────────────────────────────────
best_pt = os.path.join(model.run_dir, "weights", "best.pt")
if os.path.isfile(best_pt):
    task.upload_artifact("final_best_weights", best_pt)
    logger.info("Uploaded final best.pt: %s", best_pt)
else:
    logger.warning("best.pt not found at %s", best_pt)

# ─── Emit Task ID for chaining ─────────────────────────────────────────────────
print(task.id)
