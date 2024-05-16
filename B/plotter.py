import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from B.reporter import Reporter


class Plotter:

    def __init__(self, fine_tuning_logs_dir: str, plots_dir: str):
        self.plots_dir = plots_dir
        self.reporter = Reporter(logs_dir=fine_tuning_logs_dir)

    def create_fine_tuning_plots(
        self, checkpoints_dir: str, task_name: str, model_name: str
    ) -> None:
        logs, selected_epoch, selected_loss = self.reporter.get_fine_tuning_logs(
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

    def __create_testing_accuracy_summary_plot(
        self, colors: list, patches: list, tasks: list
    ) -> None:
        testing_accuracy = [
            0.9018,
            0,
            0.9300,
            0.9395,
            0.9361,
            0.9396,
            0.7888,
            0.9384,
            0,
            0.9320,
            0.9166,
            0.9351,
            0.9246,
            0.9404,
            0.9276,
            0,
            0.9333,
            0.9146,
            0.9359,
            0.9192,
            0.9409,
            0.9268,
            0,
            0.9387,
            0.9246,
            0.9341,
            0.9293,
            0.9414,
            0.9297,
            0,
            0.9363,
            0.9234,
            0.9381,
            0.9238,
            0.9423,
            0.9286,
        ]

        fig = plt.figure(figsize=(8, 4))
        plt.grid(axis="y")
        plt.bar(tasks, testing_accuracy, align="center", width=0.8, color=colors)
        plt.ylim([0.9, 0.945])
        plt.axhline(
            y=testing_accuracy[0],
            color=colors[0],
            linestyle="--",
            lw=3,
        )
        plt.xticks(tasks, rotation=70)
        plt.legend(handles=patches, loc="lower right", prop={"size": 8})
        plt.gca().set_yticklabels(
            ["{:.1f}%".format(x * 100) for x in plt.gca().get_yticks()]
        )
        plt.xlabel("Tasks")
        plt.ylabel("Testing Accuracy")
        plt.title("Testing Accuracy by Task (and Model)")

        fig.savefig(
            os.path.join(self.plots_dir, "Summary - Testing Accuracy.png"),
            dpi=800,
            format="png",
            bbox_inches="tight",
        )

    def __create_validation_accuracy_summary_plot(
        self, colors: list, patches: list, tasks: list
    ) -> None:
        validation_accuracy = [
            0.90810,
            0,
            0.93170,
            0.94070,
            0.93550,
            0.94310,
            0.78940,
            0.94220,
            0,
            0.93425,
            0.91745,
            0.93795,
            0.92545,
            0.94115,
            0.92830,
            0,
            0.93500,
            0.91930,
            0.93725,
            0.92150,
            0.94105,
            0.92950,
            0,
            0.93775,
            0.92675,
            0.93635,
            0.92825,
            0.94160,
            0.93240,
            0,
            0.9383,
            0.92155,
            0.93980,
            0.92395,
            0.94185,
            0.93155,
        ]

        fig = plt.figure(figsize=(8, 4))
        plt.grid(axis="y")
        plt.bar(tasks, validation_accuracy, align="center", width=0.8, color=colors)
        plt.ylim([0.9, 0.945])
        plt.axhline(
            y=validation_accuracy[0],
            color=colors[0],
            linestyle="--",
            lw=3,
        )
        plt.xticks(tasks, rotation=70)
        plt.legend(handles=patches, loc="lower right", prop={"size": 8})
        plt.gca().set_yticklabels(
            ["{:.1f}%".format(x * 100) for x in plt.gca().get_yticks()]
        )
        plt.xlabel("Tasks")
        plt.ylabel("Validation Accuracy")
        plt.title("Validation Accuracy by Task (and Model)")

        fig.savefig(
            os.path.join(self.plots_dir, "Summary - Validation Accuracy.png"),
            dpi=800,
            format="png",
            bbox_inches="tight",
        )

    def __create_testing_f1_summary_plot(
        self, colors: list, patches: list, tasks: list
    ) -> None:
        testing_f1 = [
            0.9015,
            0,
            0.9299,
            0.9392,
            0.9358,
            0.9398,
            0.7877,
            0.9381,
            0,
            0.9319,
            0.9165,
            0.9351,
            0.9245,
            0.9403,
            0.9275,
            0,
            0.9333,
            0.9144,
            0.9358,
            0.9191,
            0.9409,
            0.9268,
            0,
            0.9386,
            0.9245,
            0.9341,
            0.9292,
            0.9414,
            0.9296,
            0,
            0.9362,
            0.9233,
            0.9381,
            0.9238,
            0.9423,
            0.9286,
        ]

        fig = plt.figure(figsize=(8, 4))
        plt.grid(axis="y")
        plt.bar(tasks, testing_f1, align="center", width=0.8, color=colors)
        plt.ylim([0.9, 0.945])
        plt.axhline(
            y=testing_f1[0],
            color=colors[0],
            linestyle="--",
            lw=3,
        )
        plt.xticks(tasks, rotation=70)
        plt.legend(handles=patches, loc="lower right", prop={"size": 8})
        plt.xlabel("Tasks")
        plt.ylabel("F1 Score")
        plt.title("Testing F1 Scores by Task (and Model)")

        fig.savefig(
            os.path.join(self.plots_dir, "Summary - Testing F1.png"),
            dpi=800,
            format="png",
            bbox_inches="tight",
        )

    def __create_ft_time_per_epoch_summary_plot(
        self, colors: list, patches: list, tasks: list
    ) -> None:
        time_per_ft_epoch = [
            1314.63,
            1241.95,
            702.27,
            702.72,
            1378.03,
            1294.05,
            0,
            611.05,
            619.81,
            315.65,
            335.55,
            594.98,
            597.71,
            0,
            614.20,
            613.01,
            315.08,
            327.89,
            599.78,
            606.70,
            0,
            614.06,
            604.15,
            321.32,
            318.58,
            606.75,
            592.01,
            0,
            616.29,
            616.75,
            313.15,
            314.61,
            608.32,
            616.78,
        ]

        fig = plt.figure(figsize=(8, 4))
        plt.grid(axis="y")
        plt.bar(
            tasks[2:], time_per_ft_epoch, align="center", width=0.8, color=colors[2:]
        )
        plt.xticks(tasks[2:], rotation=70)
        plt.legend(handles=patches[1:], loc="upper right", prop={"size": 8})
        plt.xlabel("Tasks")
        plt.ylabel("Time (s)")
        plt.title("Fine Tuning Time Per Epoch by Task (and Model)")

        fig.savefig(
            os.path.join(self.plots_dir, "Summary - FT Time per Epoch.png"),
            dpi=800,
            format="png",
            bbox_inches="tight",
        )

    def __create_testing_time_summary_plot(
        self, colors: list, patches: list, tasks: list
    ) -> None:
        testing_time = [
            388.60,
            377.96,
            199.68,
            198.61,
            380.74,
            388.96,
            0,
            398.24,
            395.82,
            206.51,
            206.46,
            402.51,
            401.49,
            0,
            396.23,
            396.65,
            206.45,
            207.14,
            402.27,
            402.98,
            0,
            396.30,
            397.41,
            206.61,
            206.80,
            402.66,
            403.22,
            0,
            471.60,
            458.01,
            212.18,
            213.53,
            412.04,
            486.09,
        ]

        fig = plt.figure(figsize=(8, 4))
        plt.grid(axis="y")
        plt.bar(tasks[2:], testing_time, align="center", width=0.8, color=colors[2:])
        plt.xticks(tasks[2:], rotation=70)
        plt.legend(handles=patches[1:], loc="lower right", prop={"size": 8})
        plt.xlabel("Tasks")
        plt.ylabel("Time (s)")
        plt.title("Testing Time by Task (and Model)")

        fig.savefig(
            os.path.join(self.plots_dir, "Summary - Testing Time.png"),
            dpi=800,
            format="png",
            bbox_inches="tight",
        )

    def __create_selected_epoch_summary_plot(
        self, colors: list, patches: list, tasks: list
    ) -> None:
        selected_epoch = [
            1,
            1,
            1,
            2,
            1,
            1,
            0,
            9,
            10,
            7,
            9,
            7,
            9,
            0,
            10,
            10,
            7,
            10,
            7,
            9,
            0,
            8,
            10,
            5,
            9,
            7,
            9,
            0,
            7,
            10,
            7,
            9,
            7,
            9,
        ]

        fig = plt.figure(figsize=(8, 4))
        plt.grid(axis="y")
        plt.bar(tasks[2:], selected_epoch, align="center", width=0.8, color=colors[2:])
        plt.xticks(tasks[2:], rotation=70)
        plt.legend(handles=patches[1:], loc="lower right", prop={"size": 8})
        plt.xlabel("Tasks")
        plt.ylabel("Epoch")
        plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        plt.title("Selected Epoch During Fine Tuning by Task (and Model)")

        fig.savefig(
            os.path.join(self.plots_dir, "Summary - Selected Epoch.png"),
            dpi=800,
            format="png",
            bbox_inches="tight",
        )

    def create_bar_charts(self) -> None:
        colors = [
            "#8e918f",
            "w",
            "#1f77b4",
            "#1f77b4",
            "#ff7f0e",
            "#ff7f0e",
            "#2ca02c",
            "#2ca02c",
            "w",
            "#1f77b4",
            "#1f77b4",
            "#ff7f0e",
            "#ff7f0e",
            "#2ca02c",
            "#2ca02c",
            "w",
            "#1f77b4",
            "#1f77b4",
            "#ff7f0e",
            "#ff7f0e",
            "#2ca02c",
            "#2ca02c",
            "w",
            "#1f77b4",
            "#1f77b4",
            "#ff7f0e",
            "#ff7f0e",
            "#2ca02c",
            "#2ca02c",
            "w",
            "#1f77b4",
            "#1f77b4",
            "#ff7f0e",
            "#ff7f0e",
            "#2ca02c",
            "#2ca02c",
        ]

        nb_patch = mpatches.Patch(color="#8e918f", label="Naïve Bayes")
        bert_patch = mpatches.Patch(color="#1f77b4", label="BERT")
        distilbert_patch = mpatches.Patch(color="#ff7f0e", label="DistilBERT")
        roberta_patch = mpatches.Patch(color="#2ca02c", label="RoBERTa")
        patches = [nb_patch, bert_patch, distilbert_patch, roberta_patch]

        tasks = [
            "A0",
            "",
            "B1_v1",
            "B1_v2",
            "B2_v1",
            "B2_v2",
            "B3_v1",
            "B3_v2",
            " ",
            "C1_v1",
            "C1_v2",
            "C2_v1",
            "C2_v2",
            "C3_v1",
            "C3_v2",
            "  ",
            "D1_v1",
            "D1_v2",
            "D2_v1",
            "D2_v2",
            "D3_v1",
            "D3_v2",
            "   ",
            "E1_v1",
            "E1_v2",
            "E2_v1",
            "E2_v2",
            "E3_v1",
            "E3_v2",
            "    ",
            "F1_v1",
            "F1_v2",
            "F2_v1",
            "F2_v2",
            "F3_v1",
            "F3_v2",
        ]

        self.__create_testing_accuracy_summary_plot(
            colors=colors, patches=patches, tasks=tasks
        )

        self.__create_validation_accuracy_summary_plot(
            colors=colors, patches=patches, tasks=tasks
        )

        self.__create_testing_f1_summary_plot(
            colors=colors, patches=patches, tasks=tasks
        )

        self.__create_ft_time_per_epoch_summary_plot(
            colors=colors, patches=patches, tasks=tasks
        )

        self.__create_testing_time_summary_plot(
            colors=colors, patches=patches, tasks=tasks
        )

        self.__create_selected_epoch_summary_plot(
            colors=colors, patches=patches, tasks=tasks
        )
