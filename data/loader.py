from datasets import load_dataset
from torch.utils.data import DataLoader


def process(batch, indices, tokenizer, text_fields, padding, truncation, max_length):
    # Either encode single sentence or sentence pairs

    texts_or_text_pairs = batch[text_fields[0]]

    # Tokenize the text/text pairs
    features = tokenizer.batch_encode_plus(
        texts_or_text_pairs,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
    )

    # idx is unique ID we can use to link predictions to original data
    features["idx"] = indices

    return features


class Dataset:

    def __init__(
        self, dataset_name: str = "ag_news", seed: int = 42, val_samples: int = 20000
    ):
        self.dataset_name = dataset_name
        self.val_samples = val_samples
        self.seed = seed
        self.data = load_dataset(self.dataset_name, None)

        # Setup train, validation, test splits
        if self.val_samples > 0:
            split = self.data["train"].train_test_split(
                self.val_samples, seed=self.seed
            )
            self.data["train"] = split["train"]
            self.data["validation"] = split["test"]

    def tokenize(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenized_data = self.data.map(
            process,
            batched=True,
            with_indices=True,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "text_fields": ["text"],
                "padding": True,
                "truncation": True,
                "max_length": 128,
            },
        )

        self.tokenized_data.rename_column("label", "labels")

        print(self.tokenized_data)

        print(self.tokenized_data["train"][0])

    def get_train_loader(self):
        return DataLoader(self.tokenized_data["train"], batch_size=5)

    def get_test_loader(self):
        return DataLoader(self.tokenized_data["test"], batch_size=5)

    def get_validation_loader(self):
        return DataLoader(self.tokenized_data["validation"], batch_size=7600)

    def get(self):
        return self.data
