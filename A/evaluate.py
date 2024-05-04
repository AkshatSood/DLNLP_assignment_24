import evaluate
import torch
import pandas as pd
from tqdm import tqdm


class Evaluate:

    def __init__(self, x_test, y_test, label_map):
        self.x_test = x_test
        self.y_test = y_test

        self.accuracy = evaluate.load("accuracy")
        self.f1_score = evaluate.load("f1")
        self.precision = evaluate.load("precision")
        self.recall = evaluate.load("recall")

        classes = [label_map[i] for i in y_test]

        self.outputs_df = pd.DataFrame(
            data={"text": x_test, "class": classes, "label": y_test}
        )
        self.evaluations = []

    def evaluate(self, model, tokenizer, name):

        print(f"Evaluating {name}...")

        y_pred = []
        for text in tqdm(self.x_test):
            inputs = tokenizer.encode(text, return_tensors="pt")
            logits = model(inputs).logits
            predictions = torch.argmax(logits)
            y_pred.append(predictions.tolist())

        self.outputs_df[f"{name}"] = y_pred

        self.evaluations.append(
            {
                "model": name,
                "accuracy": self.accuracy.compute(
                    predictions=y_pred, references=self.y_test
                ),
            }
        )

    def get_evaluation(self):
        return pd.DataFrame(data=self.evaluations)

    def get_outputs(self):
        return self.outputs_df
