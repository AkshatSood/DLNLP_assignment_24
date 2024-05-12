import os
import json
import pandas as pd
import matplotlib.pyplot as plt


class Reporter:

    def __init__(self, logs_dir: str, plots_dir: str):
        self.logs_dir = logs_dir
        self.plots_dir = plots_dir
        self.checkpoint_prefix = "checkpoint-"
        self.json_file_name = "trainer_state.json"

    def __get_logs(self, checkpoints_dir: str, task_name: str, model_name: str) -> list:
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

        return logs, selected_epoch, selected_loss

    def create_fine_tuning_log(
        self, checkpoints_dir: str, task_name: str, model_name: str
    ) -> list:

        logs, selected_epoch, selected_loss = self.__get_logs(
            checkpoints_dir=checkpoints_dir, task_name=task_name, model_name=model_name
        )

        df = pd.DataFrame(logs)

        df.to_csv(
            os.path.join(self.logs_dir, f"{task_name}_{model_name}.csv"), index=False
        )

        return [log for log in logs if log["Epoch"] == selected_epoch]

    def create_fine_tuning_plots(
        self, checkpoints_dir: str, task_name: str, model_name: str
    ) -> None:
        logs, selected_epoch, selected_loss = self.__get_logs(
            checkpoints_dir=checkpoints_dir, task_name=task_name, model_name=model_name
        )

        def __extract_metrics_from_logs(logs):
            epoch = [log["Epoch"] for log in logs]
            accuracy = [log["Accuracy"] for log in logs]
            loss = [log["Loss"] for log in logs]

            return epoch, accuracy, loss

        train_epoch, train_accuracy, train_loss = __extract_metrics_from_logs(
            [log for log in logs if log["Type"] == "train"]
        )
        eval_epoch, eval_accuracy, eval_loss = __extract_metrics_from_logs(
            [log for log in logs if log["Type"] == "eval"]
        )

        fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(12, 4))

        plt.suptitle(
            f"Task {task_name} ({model_name}) - Fine Tuning Results (Selected Epoch: {selected_epoch})",
            size="x-large",
        )

        axs[0].plot(eval_epoch, eval_loss, label="Validation", c="red")
        axs[0].plot(train_epoch, train_loss, label="Training", c="blue")
        axs[0].axvline(
            x=selected_epoch,
            color="gray",
            linestyle="--",
            label="Selected Model",
            lw=3,
            alpha=0.5,
        )
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("Loss")
        axs[0].set_xlim([1 - 0.05, len(eval_epoch) + 0.05])
        axs[0].set_title("Loss")
        axs[0].set_xticks(eval_epoch)
        axs[0].legend()
        axs[0].grid(linestyle="--")

        axs[1].plot(eval_epoch, eval_accuracy, label="Validation", c="red")
        axs[1].plot(train_epoch, train_accuracy, label="Training", c="blue")
        axs[1].axvline(
            x=selected_epoch,
            color="gray",
            linestyle="--",
            label="Selected Model",
            lw=3,
            alpha=0.5,
        )
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("Accuracy")
        axs[1].set_xlim([1 - 0.05, len(eval_epoch) + 0.05])
        axs[1].set_title("Accuracy")
        axs[1].set_xticks(eval_epoch)
        axs[1].grid(linestyle="--")

        fig.savefig(
            os.path.join(self.plots_dir, f"FTLA_{task_name}_{model_name}.png"),
            dpi=800,
            format="png",
            bbox_inches="tight",
        )
