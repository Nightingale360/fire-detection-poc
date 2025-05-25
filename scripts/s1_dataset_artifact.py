# s1_dataset_artifact.py
from clearml import Task, StorageManager

# 1) Init ClearML Task
task = Task.init(
    project_name="AlphaFirewatch",
    task_name="Step 1: Download Compressed Dataset"
)

# 2) Offload to remote agent
task.execute_remotely()

# 3) Download the ZIP from Google Drive
drive_url = (
    "https://drive.google.com/uc?export=download&id=13aMEsMKUWP88o2qcR4SRsn8ZeyGchJsD"
)
local_zip = StorageManager.get_local_copy(remote_url=drive_url)
print(f"✅ Downloaded ZIP to: {local_zip}")

# 4) Upload the ZIP as an artifact
task.upload_artifact(
    name="raw_dataset_zip",
    artifact_object=local_zip
)
print("📦 Uploaded raw_dataset_zip")
