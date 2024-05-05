from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class ModelForSequenceClassification:

    def __init__(self, model_name_or_path: str, tokenizer_name: str):
        self.model = AutoModelForSequenceClassification(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, add_prefix_space=True
        )

        self.tokenizer.truncation_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def load(self):
        return self.model, self.tokenizer


class BertBaseUncased:
    def __init__(
        self,
        num_labels: int = 2,
        id2label: dict = None,
        label2id: dict = None,
        path=None,
    ):
        self.name = "bert-base-uncased"

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.name if path is None else path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.name, add_prefix_space=True)

        self.tokenizer.truncation_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def load(self):
        return self.model, self.tokenizer


class DistilBertUncased:

    def __init__(
        self,
        num_labels: int = 2,
        id2label: dict = None,
        label2id: dict = None,
        path=None,
    ):
        self.name = "distilbert-base-uncased"

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.name if path is None else path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.name, add_prefix_space=True)

        self.tokenizer.truncation_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def load(self):
        return self.model, self.tokenizer


class BertLargeUncased:

    def __init__(
        self,
        num_labels: int = 2,
        id2label: dict = None,
        label2id: dict = None,
        path=None,
    ):
        self.name = "bert-large-uncased"

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.name if path is None else path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.name, add_prefix_space=True)

        self.tokenizer.truncation_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def load(self):
        return self.model, self.tokenizer


class CamembertBase:

    def __init__(
        self,
        num_labels: int = 2,
        id2label: dict = None,
        label2id: dict = None,
        path=None,
    ):
        self.name = "almanach/camembert-base"

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.name if path is None else path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.name, add_prefix_space=True)

        self.tokenizer.truncation_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def load(self):
        return self.model, self.tokenizer
