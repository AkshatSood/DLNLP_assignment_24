from datasets import load_dataset


class Dataset:
    def __init__(
        self,
        name: str,
        text_header: str,
        label_header: str,
        id2label: dict,
        label2id: dict,
    ):
        self.name = name
        self.text_header = text_header
        self.label_header = label_header
        self.id2label = id2label
        self.label2id = label2id

        self.dataset = load_dataset(name)

    def tokenize(self, tokenizer, max_length):

        def __tokenize_mapping(samples):
            text = samples[self.text_header]
            tokenized_data = tokenizer(
                text, return_tensors="np", truncation=True, max_length=max_length
            )
            return tokenized_data

        self.dataset = self.dataset.map(__tokenize_mapping, batched=True)

        return self.dataset


class AGNewsDataset(Dataset):

    def __init__(self):
        Dataset.__init__(
            name="ag_news",
            text_header="text",
            label_header="label",
            id2label={0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
            label2id={"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3},
        )

    def load(self):
        return self.dataset

    def get_id2label(self):
        return self.id2label

    def get_label2id(self):
        return self.label2id

    def tokenize(self, tokenizer, max_length: int = 512):
        return Dataset.tokenize(tokenizer=tokenizer, max_length=max_length)
