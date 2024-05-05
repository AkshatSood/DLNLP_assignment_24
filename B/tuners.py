from time import time
import numpy as np
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig
import evaluate


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
        evaluation_strategy: str = "epoch",
        save_strategy: str = "epoch",
        load_best_model_at_end: bool = True,
    ):
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
        )

        accuracy = evaluate.load("accuracy")

        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=1)
            return {
                "accuracy": accuracy.compute(predictions=predictions, references=labels)
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

    def get_decay_parameter_names(self):
        return self.trainer.get_decay_parameter_names(self.model)

    def get_trainable_parameters(self):
        return self.trainer.get_num_trainable_parameters()

    def fine_tune(self, output_dir):
        start_time = time()
        self.trainer.train()
        end_time = time()
        self.trainer.save_model(output_dir)

        print(f"\n\nTime Taken: {end_time - start_time} seconds\n\n")


class LoraTuner:

    def __init__(
        self, model, tokenizer, train_dataset, eval_dataset, checkpoints_dir, logs_dir
    ):
        self.model = model
        self.tokenizer = tokenizer

        peft_config = LoraConfig(
            task_type="SEQ_CLS",
            r=4,
            lora_alpha=32,
            lora_dropout=0.01,
            target_modules=["q_lin"],
        )

        self.model = get_peft_model(self.model, peft_config=peft_config)

        # Hyperparameters
        lr = 1e-3
        batch_size = 4
        num_epochs = 10

        training_args = TrainingArguments(
            output_dir=checkpoints_dir,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        accuracy = evaluate.load("accuracy")

        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=1)
            return {
                "accuracy": accuracy.compute(predictions=predictions, references=labels)
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

    def get_trainable_parameters(self):
        return self.model.print_trainable_parameters()

    def fine_tune(self, output_dir):
        self.trainer.train()
        self.trainer.save_model(output_dir)
