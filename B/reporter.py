import os
import json
import pandas as pd


class Reporter:

    def __init__(self, logs_dir: str):
        self.logs_dir = logs_dir
        self.checkpoint_prefix = "checkpoint-"
        self.json_file_name = "trainer_state.json"

    def create_fine_tuning_log(
        self, checkpoints_dir: str, task_name: str, model_name: str
    ) -> dict:

        steps = []
        for _, dirs, _ in os.walk(checkpoints_dir, topdown=False):
            for name in dirs:
                steps.append(int(name.split("-")[1]))
        last_step = max(steps)

        file_path = os.path.join(
            checkpoints_dir, f"{self.checkpoint_prefix}{last_step}", self.json_file_name
        )

        trainer_state = json.load(open(file_path))

        logs = []
        final_epoch_log = None
        for log in trainer_state["log_history"]:
            if "eval_accuracy" in log:

                entry = {
                    "task": task_name,
                    "model": model_name,
                    "epoch": int(log["epoch"]),
                    "accuracy": log["eval_accuracy"]["accuracy"],
                    "loss": log["eval_loss"],
                    "step": log["step"],
                }

                if final_epoch_log is None:
                    final_epoch_log = entry
                elif final_epoch_log["epoch"] < int(log["epoch"]):
                    final_epoch_log = entry

                logs.append(entry)

        df = pd.DataFrame(logs)

        df.to_csv(os.path.join(self.logs_dir, f"{task_name}_{model_name}"), index=False)

        return final_epoch_log
