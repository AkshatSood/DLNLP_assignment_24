from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class BertBaseUncased:
    def __init__(self):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=4
        )


class DistilBertUncased:

    def __init__(
        self, num_labels: int = 2, id2label: dict = None, label2id: dict = None
    ):
        self.model_name = "distilbert-base-uncased"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, add_prefix_space=True
        )

        self.tokenizer.truncation_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def load(self):
        return self.model, self.tokenizer
