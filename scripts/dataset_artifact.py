"dataset_artifact.py"
from clearml import Task, StorageManager
import zipfile
import os

# 1) Initialize your ClearML task
task = Task.init(
    project_name="AlphaFirewatch",
    task_name="Step 1: Download Compressed Dataset"
)

task.execute_remotely()

# 2) Download the ZIP from Google Drive
drive_url = (
    "https://drive.google.com/uc?"
    "export=download&"
    "id=13aMEsMKUWP88o2qcR4SRsn8ZeyGchJsD"
)
local_zip = StorageManager.get_local_copy(remote_url=drive_url, artifact_name="labelled_dataset_zip")

print(f"✅ Downloaded ZIP to: {local_zip}")

task.upload_artifact(
    name="labelled_dataset_zip",
    artifact_object=local_zip
)

print("📦 Uploading artifacts in the background…")
print("Done🔥")
