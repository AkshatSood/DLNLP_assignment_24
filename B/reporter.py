import os
import json
import pandas as pd


class Reporter:

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.checkpoint_prefix = "checkpoint-"
        self.json_file_name = "trainer_state.json"

    def create_fine_tuning_log(self, checkpoints_dir: str, log_name: str):

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
        for log in trainer_state["log_history"]:
            if "eval_accuracy" in log:
                logs.append(
                    {
                        "epoch": int(log["epoch"]),
                        "accuracy": log["eval_accuracy"]["accuracy"],
                        "loss": log["eval_loss"],
                        "step": log["step"],
                    }
                )

        df = pd.DataFrame(logs)

        df.to_csv(os.path.join(self.log_dir, log_name), index=False)
