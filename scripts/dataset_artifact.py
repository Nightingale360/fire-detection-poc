"dataset_artifact.py"
from clearml import Task, StorageManager
import zipfile
import os

# 1) Initialize your ClearML task
task = Task.init(
    project_name="AlphaFirewatch",
    task_name="Step 1: Download & Unzip Dataset"
)

task.execute_remotely()

# 2) Download the ZIP from Google Drive
drive_url = (
    "https://drive.google.com/uc?"
    "export=download&"
    "id=13aMEsMKUWP88o2qcR4SRsn8ZeyGchJsD"
)
local_zip = StorageManager.get_local_copy(remote_url=drive_url)

print(f"✅ Downloaded ZIP to: {local_zip}")

# 3) Unzip to a known folder
extract_dir = os.path.join(os.getcwd(), "data")
os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(local_zip, 'r') as zf:
    zf.extractall(extract_dir)
print(f"✅ Extracted contents to: {extract_dir}")

# 4) Upload the unzipped directory as an artifact
#    ClearML will recursively upload the folder’s contents
task.upload_artifact(
    name="labelled_dataset",
    artifact_object=extract_dir
)

print("📦 Uploading artifacts in the background…")
print("Done🔥")
