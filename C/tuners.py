from time import time
import numpy as np
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig
import evaluate


class LoraTuner:

    def __init__(
        self,
        model=None,
        tokenizer=None,
        train_dataset=None,
        eval_dataset=None,
        checkpoints_dir: str = None,
        learning_rate: float = 5e-5,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        weight_decay: float = 0.01,
        epochs: int = 5,
        evaluation_strategy: str = "epoch",
        save_strategy: str = "epoch",
        load_best_model_at_end: bool = True,
        seed: int = 42,
        lora_task_type: str = "SEQ_CLS",
        lora_r: int = 4,
        lora_alpha: int = 32,
        lora_dropout: float = 0.01,
        lora_target_modules: list = None,
    ):
        self.model = model
        self.tokenizer = tokenizer

        peft_config = LoraConfig(
            task_type=lora_task_type,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
        )

        self.model = get_peft_model(self.model, peft_config=peft_config)

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
            seed=seed,
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
        return self.model.get_nb_trainable_parameters()

    def fine_tune(self, output_dir):
        start_time = time()
        self.trainer.train()
        end_time = time()
        self.trainer.save_model(output_dir)

        print(
            f'\n\nModel training complete. Saved at "{output_dir}"\nTime Taken: {end_time - start_time} seconds\n\n'
        )
