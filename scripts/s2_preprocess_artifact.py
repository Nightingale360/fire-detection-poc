# s2_preprocess_artifact.py
import os
import zipfile
import argparse
from clearml import Task, StorageManager

# 1) Parse CLI args
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_task_id", required=True,
    help="Task ID from Step 1 (raw_dataset_zip)"
)
args = parser.parse_args()

# Explicit validation
if not args.dataset_task_id:
    raise ValueError("Missing dataset_task_id — Supply Step 1 Task ID")

# 2) Init ClearML Task
task = Task.init(
    project_name="AlphaFirewatch",
    task_name="Step 2: Unzip YOLO Dataset"
)
task.connect(vars(args))

# 3) offload remotely
task.execute_remotely()

# 4) Download the raw ZIP
local_zip = StorageManager.get_local_copy(
    task_id=args.dataset_task_id,
    artifact_name="raw_dataset_zip"
)
print(f"✅ Downloaded raw zip to: {local_zip}")

# 5) Unzip into ./yolo_data
extract_dir = os.path.join(os.getcwd(), "yolo_data")
os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(local_zip, "r") as zf:
    zf.extractall(extract_dir)
print(f"✅ Extracted contents to: {extract_dir}")

# 6) Upload the extracted folder
task.upload_artifact(
    name="yolo_dataset",
    artifact_object=extract_dir
)
print("📦 Uploaded yolo_dataset")
