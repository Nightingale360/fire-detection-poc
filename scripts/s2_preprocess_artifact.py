# s2_preprocess_artifact.py
import os
import zipfile
import argparse
from clearml import Task, StorageManager

# Parse CLI args ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_task_id", required=True,
    help="ClearML Task ID from Step 1 ( this is needed so we can chain the different pipeline stages)"
)
args = parser.parse_args()

# Init ClearML Task ───────────────────────────────────────────────────────
task = Task.init(
    project_name="AlphaFirewatch",
    task_name="Step 2: Unzip YOLO Dataset"
)

task.execute_remotely()

# get dataset from task's artifact
if not args.dataset_task_id
    raise ValueError("Missing dataset link")

local_zip = StorageManager.get_local_copy(
    task_id=args.dataset_task_id,
    artifact_name="labelled_dataset_zip"
)

print(f"✅ Downloaded raw zip to: {local_zip}")


extract_dir = os.path.join(os.getcwd(), "yolo_data")
os.makedirs(extract_dir, exist_ok=True)


with zipfile.ZipFile(local_zip, "r") as zf:
    zf.extractall(extract_dir)
print(f"✅ Extracted contents to: {extract_dir}")

# Upload the extracted folder
task.upload_artifact(
    name="yolo_dataset",
    artifact_object=extract_dir
)
print("📦 Uploaded artifact in the background…")

# Print out this Task’s ID for CI chaining
print(task.id)
