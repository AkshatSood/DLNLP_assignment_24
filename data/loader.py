from datasets import load_dataset


class DataLoader:

    def __init__(
        self, dataset_name: str = "ag_news", seed: int = 42, val_samples: int = 20000
    ):
        self.dataset_name = dataset_name
        self.val_samples = val_samples
        self.seed = seed

    def load(self):
        self.data = load_dataset(self.dataset_name, None)

        # Setup train, validation, test splits
        if self.val_samples > 0:
            split = self.data["train"].train_test_split(self.val_samples)
            self.data["train"] = split["train"]
            self.data["validation"] = split["test"]
