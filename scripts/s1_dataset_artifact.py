# s1_dataset_artifact.py
import time
from clearml import Task, StorageManager

# ─── 1) Init ClearML Task ──────────────────────────────────────────────────────
print("→ Initializing ClearML Task…")
task = Task.init(
    project_name="AlphaFirewatch",
    task_name="Step 1: Download Compressed Dataset"
)
print(f"→ Task initialized: {task.id}")

# ─── 2) Offload to remote agent ────────────────────────────────────────────────
print("→ Offloading execution to remote agent…")
task.execute_remotely()

# ─── 3) Download the ZIP from Google Drive ─────────────────────────────────────
drive_url = (
    "https://drive.google.com/uc?export=download&id=13aMEsMKUWP88o2qcR4SRsn8ZeyGchJsD"
)
print(f"→ Downloading from: {drive_url}")
start = time.time()
local_zip = StorageManager.get_local_copy(remote_url=drive_url)
elapsed = time.time() - start
print(f"✅ Download complete ({elapsed:.1f}s): {local_zip!r}")

# ─── 4) Upload the ZIP as an artifact ──────────────────────────────────────────
print("→ Uploading ZIP as artifact ‘raw_dataset_zip’…")
task.upload_artifact(
    name="raw_dataset_zip",
    artifact_object=local_zip
)
print("📦 Uploaded raw_dataset_zip")

# ─── 5) Done ───────────────────────────────────────────────────────────────────
print("🎉 s1_dataset_artifact.py finished successfully") 
