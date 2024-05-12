import os
from time import time
import json, codecs
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


class Evaluator:

    def __init__(
        self,
        x_test,
        y_test,
        label_map: dict,
        evaluations_dir: str,
        test_outputs_dir: str,
    ):
        self.x_test = x_test
        self.y_test = y_test
        self.label_map = label_map
        self.evaluations_dir = evaluations_dir
        self.test_outputs_dir = test_outputs_dir

    def create_evaluations(self, model, tokenizer, model_name: str, task_name: str):

        classes = [self.label_map[i] for i in self.y_test]
        outputs_df = pd.DataFrame(
            data={"text": self.x_test, "test_class": classes, "test_label": self.y_test}
        )

        pred_labels = []
        pred_classes = []
        start_time = time()
        for text in tqdm(self.x_test):
            inputs = tokenizer.encode(text, return_tensors="pt")
            logits = model(inputs).logits
            predictions = torch.argmax(logits)

            label = predictions.tolist()

            pred_labels.append(label)
            pred_classes.append(self.label_map[label])

        outputs_df["pred_classes"] = pred_classes
        outputs_df["pred_labels"] = pred_labels

        end_time = time()
        evaluation = {
            "model": model_name,
            "task": task_name,
            "evaluation_time": end_time - start_time,
            "accuracy": accuracy_score(y_true=self.y_test, y_pred=pred_labels),
            "f1_score": f1_score(
                y_true=self.y_test, y_pred=pred_labels, average="macro"
            ),
            "precision": precision_score(
                y_true=self.y_test, y_pred=pred_labels, average="macro"
            ),
            "recall": recall_score(
                y_true=self.y_test, y_pred=pred_labels, average="macro"
            ),
            "confusion_matrix": confusion_matrix(
                y_true=self.y_test,
                y_pred=pred_labels,
                labels=list(self.label_map.keys()),
            ).tolist(),
        }

        # Write the outputs to a CSV file
        outputs_df.to_csv(
            os.path.join(self.test_outputs_dir, f"{task_name}_{model_name}.csv")
        )

        # Write the evaluation to a JSON file
        json.dump(
            evaluation,
            codecs.open(
                filename=os.path.join(
                    self.evaluations_dir, f"{task_name}_{model_name}.json"
                ),
                mode="w",
                encoding="utf-8",
            ),
        )

        return evaluation
