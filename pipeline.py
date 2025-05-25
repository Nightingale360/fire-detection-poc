from clearml.automation import PipelineController

EXECUTION_QUEUE = "FireWatchQueue"

pipe = PipelineController(
    project="AlphaFirewatch",
    name="Firewatch End-to-End Pipeline",
    version="0.0.5",
    add_pipeline_tags=False,
)
pipe.set_default_execution_queue(EXECUTION_QUEUE)

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
        "Args/dataset_task_id": "${step1_download_zip.id}"
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
        "Args/dataset_task_id": "${step2_unzip.id}",
        "Args/model_arch":      "yolo11n.pt",
        "Args/epochs":          50,
        "Args/batch":           50,
        "Args/imgsz":           640,
    }
)

# Step 4: Hyper‐Parameter Optimization
pipe.add_step(
    name="step4_hpo",
    parents=["step3_train"],
    base_task_project="AlphaFirewatch",
    base_task_name="HPO: FireWatch YOLO Tuning",
    execution_queue=EXECUTION_QUEUE,
    parameter_override={
        "Args/dataset_task_id":   "${step2_unzip.id}",
        "Args/train_task_id":     "${step3_train.id}",
        "Args/num_trials":        4,
        "Args/epochs":            100,
        "Args/time_limit_minutes": 60,
        "Args/test_queue":        EXECUTION_QUEUE,
    }
)

# Step 5: Final Model Training
pipe.add_step(
    name="step5_final",
    parents=["step4_hpo", "step2_unzip"],
    base_task_project="AlphaFirewatch",
    base_task_name="Step 5: Final Model Training",
    execution_queue=EXECUTION_QUEUE,
    parameter_override={
        "Args/dataset_task_id": "${step2_unzip.id}",
        "Args/hpo_task_id":     "${step4_hpo.id}",
        "Args/model_arch":      "yolo11n.pt",
        "Args/imgsz":           640,
        "Args/project":         "AlphaFirewatch",
        "Args/name":            "yolov11_final",
        "Args/conf_thres":      0.25,
    }
)

pipe.start(queue="EXECUTION_QUEUE")
print("Pipeline launched, controller ID:", pipe.id)
