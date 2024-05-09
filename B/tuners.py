from time import time
from copy import deepcopy
import numpy as np
from transformers import (
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    IntervalStrategy,
)
import evaluate


class EvaluateTrainingCallback(TrainerCallback):

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, metric_key_prefix="train"
            )
            return control_copy


class Tuner:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        checkpoints_dir: str,
        learning_rate: float = 5e-5,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        weight_decay: float = 0.01,
        epochs: int = 5,
        evaluation_strategy: str = IntervalStrategy.EPOCH,
        save_strategy: str = IntervalStrategy.EPOCH,
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "loss",
        greater_is_better: bool = False,
        seed: int = 42,
    ):
        self.__print_config(
            checkpoints_dir=checkpoints_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            weight_decay=weight_decay,
            epochs=epochs,
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            seed=seed,
        )

        self.model = model
        self.tokenizer = tokenizer

        training_args = TrainingArguments(
            output_dir=checkpoints_dir,
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            seed=seed,
        )

        accuracy = evaluate.load("accuracy")
        f1 = evaluate.load("f1")
        precision = evaluate.load("precision")
        recall = evaluate.load("recall")

        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=1)
            return {
                "accuracy": accuracy.compute(
                    predictions=predictions, references=labels
                ),
                "f1": f1.compute(
                    predictions=predictions, references=labels, average="macro"
                ),
                "precision": precision.compute(
                    predictions=predictions, references=labels, average="macro"
                ),
                "recall": recall.compute(
                    predictions=predictions, references=labels, average="macro"
                ),
            }

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=compute_metrics,
        )

        self.trainer.add_callback(EvaluateTrainingCallback(self.trainer))

        print(f"\n=> Tuner - Training Device: {training_args.device}\n")

    def __print_config(
        self,
        checkpoints_dir: str,
        learning_rate: float,
        per_device_train_batch_size: int,
        per_device_eval_batch_size: int,
        weight_decay: float,
        epochs: int,
        evaluation_strategy: str,
        save_strategy: str,
        load_best_model_at_end: bool,
        metric_for_best_model: str,
        greater_is_better: bool,
        seed: int,
    ):
        print("\n=> Tuner - Configration:")
        print(f"\tCheckpoints Dir: {checkpoints_dir}")
        print(f"\tTraining Arguments - Learning Rate: {learning_rate}")
        print(
            f"\tTraining Arguments - Per Device Train Batch Size: {per_device_train_batch_size}"
        )
        print(
            f"\tTraining Arguments - Per Device Eval Batch Size: {per_device_eval_batch_size}"
        )
        print(f"\tTraining Arguments - Weight Decay: {weight_decay}")
        print(f"\tTraining Arguments - Epochs: {epochs}")
        print(f"\tTraining Arguments - Evaluation Strategy: {evaluation_strategy}")
        print(f"\tTraining Arguments - Save Strategy: {save_strategy}")
        print(f"\tTraining Arguments - Metric for Best Model: {metric_for_best_model}")
        print(f"\tTraining Arguments - Greater is Better: {greater_is_better}")
        print(
            f"\tTraining Arguments - Load Best Model at End: {load_best_model_at_end}"
        )
        print(f"\tTraining Arguments - Seed: {seed}\n")

    def get_decay_parameter_names(self):
        return self.trainer.get_decay_parameter_names(self.model)

    def get_trainable_parameters(self):
        return (
            self.model.num_parameters(only_trainable=True),
            self.model.num_parameters(),
        )

    def fine_tune(self, output_dir):
        start_time = time()
        self.trainer.train()
        end_time = time()
        self.trainer.save_model(output_dir)

        print(
            f'\n\n=> Model training complete. Saved at "{output_dir}"\n=> Time Taken: {end_time - start_time} seconds\n\n'
        )

        return end_time - start_time
