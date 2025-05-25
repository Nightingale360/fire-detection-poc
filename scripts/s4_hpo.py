# s4_hpo.py
import argparse
import logging
import time
import json

from clearml import Task, Dataset
from clearml.automation import HyperParameterOptimizer
from clearml.automation import UniformIntegerParameterRange, UniformParameterRange

# ─── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Parse CLI args ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_task_id", required=True,
    help="Task ID from Step 2 (yolo_dataset)"
)
parser.add_argument(
    "--train_task_id", required=True,
    help="Task ID from Step 3 (baseline YOLO training)"
)
parser.add_argument(
    "--num_trials", type=int, default=4,
    help="Total HPO trials to run"
)
parser.add_argument(
    "--epochs", type=int, default=50,
    help="Upper bound for HPO search on number of epochs"
)
parser.add_argument(
    "--time_limit_minutes", type=int, default=20,
    help="Overall HPO time limit (minutes)"
)
parser.add_argument(
    "--test_queue", default="FireWatchQueue",
    help="Queue name for spawned trial tasks"
)
args = parser.parse_args()

# ─── Initialize ClearML optimizer task ──────────────────────────────────────────
task = Task.init(
    project_name="AlphaFirewatch",
    task_name="HPO: FireWatch YOLO Tuning",
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False,
)
args = task.connect(args)
logger.info("Connected parameters: %s", args)

# ─── Offload optimizer to agent ────────────────────────────────────────────────
task.execute_remotely()

# Verify dataset artifact exists
try:
    ds = Dataset.get(dataset_id=args.dataset_task_id)
    logger.info("Using dataset: %s (%s)", ds.name, args.dataset_task_id)
except Exception as e:
    logger.error("Failed to fetch dataset: %s", e)
    raise

# ─── Configure HyperParameterOptimizer ────────────────────────────────────────
hpo = HyperParameterOptimizer(
    base_task_id=args.train_task_id,
    hyper_parameters=[
        # Tune number of epochs between 20 and max
        UniformIntegerParameterRange(
            "epochs",
            min_value=20,
            max_value=args.epochs  
        ),
        # Tune batch size between 16 and 64
        UniformIntegerParameterRange("batch", min_value=16, max_value=128),
    ],
    # This matches  YOLO validation logging:
    objective_metric_title="train/metrics",
    objective_metric_series="val/mAP50",
    objective_metric_sign="max",

    max_number_of_concurrent_tasks=2,
    total_max_jobs=args.num_trials,
    optimization_time_limit=args.time_limit_minutes * 60,
    pool_period_min=1.0,
    execution_queue=args.test_queue,
    save_top_k_tasks_only=1,

    # make sure each trial knows where to fetch data and queue
    parameter_override={
        "args.dataset_task_id": args.dataset_task_id,
        "args.test_queue":       args.test_queue,
    }
)

# Run & wait
logger.info("Starting HPO: %d trials, %d min", args.num_trials, args.time_limit_minutes)
hpo.start()
time.sleep(args.time_limit_minutes * 60)
logger.info("Time limit reached, stopping HPO")
hpo.stop()

# Retrieve best experiment
top = hpo.get_top_experiments(top_k=1)
if not top:
    raise RuntimeError("HPO produced no completed trials")

best = top[0]
best_id = best.id
best_params = best.get_parameters()
metrics     = best.get_last_scalar_metrics()
best_map50  = metrics.get("train/metrics", {}).get("val/mAP50", None)

logger.info(
    "Best trial %s → params: %s  val/mAP50=%.4f",
    best_id, best_params, best_map50
)

#  Persist best results
out = {
    "best_task_id":    best_id,
    "best_parameters": best_params,
    "best_map50":      best_map50,
}
with open("best_hpo_results.json", "w") as fp:
    json.dump(out, fp, indent=2)

task.upload_artifact("best_hpo_results", "best_hpo_results.json")
task.set_parameter("best_task_id",    best_id)
task.set_parameter("best_map50",      best_map50)
task.set_parameter("best_parameters", best_params)
logger.info("Uploaded best_hpo_results.json and set task parameters")

print("🎯 HPO complete.")
