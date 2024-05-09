from omegaconf.dictconfig import DictConfig
from datasets.dataset_dict import DatasetDict
from datasets import load_dataset


class DatasetLoader:
    def __init__(
        self,
        name: str,
        num_labels: int,
        text_header: str,
        label_header: str,
        id2label: dict,
        label2id: dict,
    ):
        self.name = name
        self.num_labels = num_labels
        self.text_header = text_header
        self.label_header = label_header
        self.id2label = id2label
        self.label2id = label2id

        self.dataset = load_dataset(name)

    def tokenize(self, tokenizer, max_length) -> DatasetDict:

        def __tokenize_mapping(samples):
            text = samples[self.text_header]
            tokenized_data = tokenizer(
                text, return_tensors="np", truncation=True, max_length=max_length
            )
            return tokenized_data

        self.dataset = self.dataset.map(__tokenize_mapping, batched=True)

        return self.dataset

    def information(self) -> dict:

        distributions = {}
        for split in self.dataset.keys():
            label_counts = {}

            for sample in self.dataset[split]:
                label = sample[self.label_header]
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1

            distributions[split] = label_counts

        return {
            "name": self.name,
            "id2label": self.id2label,
            "label2id": self.label2id,
            "num_columns": self.dataset.num_columns,
            "num_rows": self.dataset.num_rows,
            "column_names": self.dataset.column_names,
            "shapes": self.dataset.shape,
            "label_distribution": distributions,
        }


class AGNewsDatasetLoader(DatasetLoader):

    def __init__(self, config: DictConfig):
        DatasetLoader.__init__(
            self,
            name="ag_news",
            num_labels=4,
            text_header="text",
            label_header="label",
            id2label={0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
            label2id={"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3},
        )

        # Create a validation dataset from the test dataset
        split = self.dataset["train"].train_test_split(
            test_size=config.validation_size, stratify_by_column="label", seed=42
        )
        self.dataset["train"] = split["train"]
        self.dataset["validation"] = split["test"]

    def load(self) -> DatasetDict:
        return self.dataset

    def tokenize(self, tokenizer, max_length: int = 512) -> DatasetDict:
        return DatasetLoader.tokenize(self, tokenizer=tokenizer, max_length=max_length)
