# run_pipeline.py
from clearml.automation import PipelineController

EXECUTION_QUEUE = "FireWatchQueue"

def pre_cb(pipeline, node, current_override):
    print(f"→ Launching step `{node.name}` using base task {node.base_task_id}")
    return True

def post_cb(pipeline, node):
    print(f"✓ Completed step `{node.name}` → new Task ID {node.executed}")

# Create the pipeline controller ─────────────────────────────────────────
pipe = PipelineController(
    project="AlphaFirewatch",
    name="Firewatch End-to-End Pipeline",
    version="0.0.5",
    add_pipeline_tags=False,
    pre_execute_callback=pre_cb,
    post_execute_callback=post_cb,# don’t auto-tag every run
)

pipe.set_default_execution_queue("task")

# Step 1: Download compressed dataset
pipe.add_step(
    name="step1_download_zip",
    base_task_project="AlphaFirewatch",
    base_task_name="Step 1: Download Compressed Dataset",
    execution_queue=EXECUTION_QUEUE,
)

# Step 2: Unzip YOLO Dataset
pipe.add_step(
    name="step2_unzip",
    parents=["step1_download_zip"],
    base_task_project="AlphaFirewatch",
    base_task_name="Step 2: Unzip YOLO Dataset",
    execution_queue=EXECUTION_QUEUE,
    parameter_override={
        # this matches the argparse name in s2_preprocess_artifact.py
        "dataset_task_id": "${step1_download_zip.id}"
    }
)

# Step 3: Fire & Smoke Detection Training
pipe.add_step(
    name="step3_train",
    parents=["step2_unzip"],
    base_task_project="AlphaFirewatch",
    base_task_name="Step 3: Fire and Smoke Detection Training",
    execution_queue=EXECUTION_QUEUE,
    parameter_override={
        # match the argparse dest names in s3_training.py
        "dataset_task_id": "${step2_unzip.id}",
        "model_arch":      "yolo11n.pt",
        "epochs":          50,
        "batch":           50,
        "imgsz":           640,
    }
)

# Step 4: HPO
# Step 4: Hyper-Parameter Optimization
pipe.add_step(
    name="step4_hpo",
    parents=["step3_train"],
    base_task_project="AlphaFirewatch",
    base_task_name="HPO: FireWatch YOLO Tuning",
    execution_queue=EXECUTION_QUEUE,
    parameter_override={
        # these must match your argparse dest names in s4_hpo.py
        "dataset_task_id":   "${step2_unzip.id}",
        "train_task_id":     "${step3_train.id}",
        "num_trials":        4,
        "epochs":            100,
        "time_limit_minutes": 60,
        "test_queue":        EXECUTION_QUEUE,
    },
)

# Step 5: Final Model Training
pipe.add_step(
    name="step5_final",
    parents=["step4_hpo", "step2_unzip"],
    base_task_project="AlphaFirewatch",
    base_task_name="Step 5: Final Model Training",
    execution_queue=EXECUTION_QUEUE,
    parameter_override={
        "dataset_task_id": "${step2_unzip.id}",
        "hpo_task_id":     "${step4_hpo.id}",
        "model_arch":      "yolo11n.pt",
        "imgsz":           640,
        "project":         "AlphaFirewatch",
        "name":            "yolov11_final",
        "conf_thres":      0.25,
    }
)




# ─── 7) Kick it off ───────────────────────────────────────────────────────────
# Choose `local=True` if you want each step to run in this process,
# or `local=False` (the default) to dispatch to your agents.
pipe.start(queue="task")
print("Pipeline launched, controller ID:", pipe.id)
