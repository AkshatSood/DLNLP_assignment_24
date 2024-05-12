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

        def __get_entry_from_log(log, prefix: str):
            return {
                "Task": task_name,
                "Model": model_name,
                "Type": prefix,
                "Epoch": int(log["epoch"]),
                "Accuracy": log[f"{prefix}_accuracy"]["accuracy"],
                "F1": log[f"{prefix}_f1"]["f1"],
                "Precision": log[f"{prefix}_precision"]["precision"],
                "Recall": log[f"{prefix}_recall"]["recall"],
                "Loss": log[f"{prefix}_loss"],
                "Step": log["step"],
                "Runtime": log[f"{prefix}_runtime"],
                "Samples Per Second": log[f"{prefix}_samples_per_second"],
                "Steps Per Second": log[f"{prefix}_steps_per_second"],
            }

        logs = []
        selected_epoch = None
        selected_loss = None
        for log in trainer_state["log_history"]:
            if "eval_accuracy" in log:
                eval_entry = __get_entry_from_log(log, "eval")

                if selected_loss is None:
                    selected_loss = eval_entry["Loss"]
                    selected_epoch = eval_entry["Epoch"]
                elif selected_loss >= eval_entry["Loss"]:
                    selected_loss = eval_entry["Loss"]
                    selected_epoch = eval_entry["Epoch"]

                logs.append(eval_entry)

            if "train_accuracy" in log:
                train_entry = __get_entry_from_log(log, "train")

                logs.append(train_entry)

        df = pd.DataFrame(logs)

        df.to_csv(
            os.path.join(self.logs_dir, f"{task_name}_{model_name}.csv"), index=False
        )

        return [log for log in logs if log["Epoch"] == selected_epoch]
