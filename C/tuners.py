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
        lora_use_rslora: bool = False,
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
            seed=seed,
            lora_task_type=lora_task_type,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            lora_use_rslora=lora_use_rslora,
        )

        peft_config = LoraConfig(
            task_type=lora_task_type,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            use_rslora=lora_use_rslora,
        )

        self.model = get_peft_model(model, peft_config=peft_config)
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
            seed=seed,
        )

        accuracy = evaluate.load("accuracy")

        def compute_metrics(p):
            predictions, references = p
            predictions = np.argmax(predictions, axis=1)
            return {
                "accuracy": accuracy.compute(
                    predictions=predictions, references=references
                )
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

        print(f"\n=> LoRATuner - Training Device: {training_args.device}\n")

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
        seed: int,
        lora_task_type: str,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: list,
        lora_use_rslora: bool,
    ):
        print("\n=> LoRATuner - Configration:")
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
        print(
            f"\tTraining Arguments - Load Best Model at End: {load_best_model_at_end}"
        )
        print(f"\tTraining Arguments - Seed: {seed}")
        print(f"\tLoRA Config - Task Type: {lora_task_type}")
        print(f"\tLoRA Config - R: {lora_r}")
        print(f"\tLoRA Config - Alpha: {lora_alpha}")
        print(f"\tLoRA Config - Dropout: {lora_dropout}")
        print(f"\tLoRA Config - Target Modules: {lora_target_modules}")
        print(f"\tLoRA Config - Use RS LoRA: {lora_use_rslora}\n")

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
            f'\n\n=> Model training complete. Saved at "{output_dir}"\n=> Time Taken: {end_time - start_time} seconds\n\n'
        )
