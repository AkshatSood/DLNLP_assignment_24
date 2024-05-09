import os
from datetime import datetime
import json, codecs


class Logger:

    def __init__(self, logs_dir: str):
        self.logs_dir = logs_dir
        self.logs = []
        self.execution_start_time = None

    def print_execution_start(self):
        timestamp = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        self.execution_start_time = timestamp
        print(
            f"\n\n\n\n\n\n************* NEW EXECUTION STARTED AT {timestamp} *************\n\n\n\n\n\n\n"
        )

    def print_heading(self, text: str):
        print(
            "\n\n\n#####################################################################"
        )
        print("---------------------------------------------------------------------")
        print(f"=> {text}")
        print("---------------------------------------------------------------------")
        print(
            "#####################################################################\n\n\n"
        )

    def format_dict(self, dictionary: dict) -> str:
        return json.dumps(dictionary, sort_keys=True, indent=4)

    def log_fine_tuning_duration(
        self, task: str, model: str, time_taken: float
    ) -> None:
        self.logs.append(
            {
                "task": task,
                "model": model,
                "type": "Fine Tuning Duration",
                "time": datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
                "time_taken": time_taken,
            }
        )

    def log_evaluation_results(self, task: str, model: str, results: dict) -> None:
        self.logs.append(
            {
                "task": task,
                "model": model,
                "type": "Evaluation Results",
                "time": datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
                "evaluation_results": results,
            }
        )

    def log_fine_tuning_results(self, task: str, model: str, results: dict) -> None:
        self.logs.append(
            {
                "task": task,
                "model": model,
                "type": "Fine Tuning Results",
                "time": datetime.now().strftime("%d/%m/%Y, %H:%M:%S"),
                "tuning_results": results,
            }
        )

    def create_log(self):
        execution_end_time = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        json.dump(
            {
                "Execution Start Time": self.execution_start_time,
                "Execution End Time": execution_end_time,
                "Logs": self.logs,
            },
            codecs.open(
                filename=os.path.join(self.logs_dir, f"log_{timestamp}.json"),
                mode="w",
                encoding="utf-8",
            ),
        )
