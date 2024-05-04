from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig
import evaluate
import numpy as np
from A.models import DistilBertUncased


class LoraTuner:

    def __init__(self, model, tokenizer, train_dataset, eval_dataset):
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
            output_dir="./test",
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

    def fine_tune(self):
        self.trainer.train()
        self.trainer.save_model("./example")
