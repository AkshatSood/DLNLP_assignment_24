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

    def create_evaluations(
        self,
        model,
        tokenizer,
        model_name: str,
        task_name: str,
        create_outputs: bool = True,
    ):

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

        if create_outputs:
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

    def create_nb_evaluations(
        self,
        model,
        x_test: list,
        y_test: list,
        model_name: str,
        task_name: str,
        x_train: list = None,
        y_train: list = None,
        x_val: list = None,
        y_val: list = None,
        only_test: bool = False,
    ) -> list:

        y_test_pred = model.predict(x_test)

        test_evaluation = {
            "model": model_name,
            "task": task_name,
            "evaluation_time": 0,
            "accuracy": accuracy_score(y_true=y_test, y_pred=y_test_pred),
            "f1_score": f1_score(y_true=y_test, y_pred=y_test_pred, average="macro"),
            "precision": precision_score(
                y_true=y_test, y_pred=y_test_pred, average="macro"
            ),
            "recall": recall_score(y_true=y_test, y_pred=y_test_pred, average="macro"),
            "confusion_matrix": confusion_matrix(
                y_true=y_test,
                y_pred=y_test_pred,
                labels=list(self.label_map.keys()),
            ).tolist(),
        }

        if not only_test:
            y_train_pred = model.predict(x_train)
            y_val_pred = model.predict(x_val)

            all_evaluations = [
                {
                    "model": model_name,
                    "task": task_name,
                    "split": "test",
                    "evaluation_time": 0,
                    "accuracy": accuracy_score(y_true=y_test, y_pred=y_test_pred),
                    "f1_score": f1_score(
                        y_true=y_test, y_pred=y_test_pred, average="macro"
                    ),
                    "precision": precision_score(
                        y_true=y_test, y_pred=y_test_pred, average="macro"
                    ),
                    "recall": recall_score(
                        y_true=y_test, y_pred=y_test_pred, average="macro"
                    ),
                    "confusion_matrix": confusion_matrix(
                        y_true=y_test,
                        y_pred=y_test_pred,
                        labels=list(self.label_map.keys()),
                    ).tolist(),
                },
                {
                    "model": model_name,
                    "task": task_name,
                    "split": "validation",
                    "evaluation_time": 0,
                    "accuracy": accuracy_score(y_true=y_val, y_pred=y_val_pred),
                    "f1_score": f1_score(
                        y_true=y_val, y_pred=y_val_pred, average="macro"
                    ),
                    "precision": precision_score(
                        y_true=y_val, y_pred=y_val_pred, average="macro"
                    ),
                    "recall": recall_score(
                        y_true=y_val, y_pred=y_val_pred, average="macro"
                    ),
                    "confusion_matrix": confusion_matrix(
                        y_true=y_val,
                        y_pred=y_val_pred,
                        labels=list(self.label_map.keys()),
                    ).tolist(),
                },
                {
                    "model": model_name,
                    "task": task_name,
                    "split": "training",
                    "evaluation_time": 0,
                    "accuracy": accuracy_score(y_true=y_train, y_pred=y_train_pred),
                    "f1_score": f1_score(
                        y_true=y_train, y_pred=y_train_pred, average="macro"
                    ),
                    "precision": precision_score(
                        y_true=y_train, y_pred=y_train_pred, average="macro"
                    ),
                    "recall": recall_score(
                        y_true=y_train, y_pred=y_train_pred, average="macro"
                    ),
                    "confusion_matrix": confusion_matrix(
                        y_true=y_train,
                        y_pred=y_train_pred,
                        labels=list(self.label_map.keys()),
                    ).tolist(),
                },
            ]

        # Write the evaluation to a JSON file
        json.dump(
            test_evaluation,
            codecs.open(
                filename=os.path.join(
                    self.evaluations_dir, f"{task_name}_{model_name}.json"
                ),
                mode="w",
                encoding="utf-8",
            ),
        )

        if only_test:
            return test_evaluation

        return all_evaluations
