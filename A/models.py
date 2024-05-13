from transformers import (
    DistilBertForSequenceClassification,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    BertTokenizerFast,
    DistilBertTokenizerFast,
    RobertaTokenizerFast,
)


class BertBaseUncased:
    def __init__(
        self,
        num_labels: int = 2,
        id2label: dict = None,
        label2id: dict = None,
        path: str = None,
    ):
        self.name = "bert-base-uncased"

        if path is None:
            print(
                f"\n=> No fine tuned model provided. Fetching pre trained BertForSequenceClassification from HG\n"
            )
        else:
            print(
                f"\n=> Fine tuned BertForSequenceClassification model provided at {path}\n"
            )

        self.model = BertForSequenceClassification.from_pretrained(
            self.name if path is None else path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        self.tokenizer = BertTokenizerFast.from_pretrained(
            self.name if path is None else path, add_prefix_space=True
        )

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
        path: str = None,
    ):
        self.name = "distilbert-base-uncased"

        if path is None:
            print(
                f"\n=> No fine tuned model provided. Fetching pre trained DistilBertForSequenceClassification from HG\n"
            )
        else:
            print(
                f"\n=> Fine tuned DistilBertForSequenceClassification model provided at {path}\n"
            )

        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.name if path is None else path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            self.name if path is None else path, add_prefix_space=True
        )

        self.tokenizer.truncation_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def load(self):
        return self.model, self.tokenizer


class RobertaBase:
    def __init__(
        self,
        num_labels: int = 2,
        id2label: dict = None,
        label2id: dict = None,
        path: str = None,
    ):
        self.name = "roberta-base"

        if path is None:
            print(
                f"\n=> No fine tuned model provided. Fetching pre trained RobertaForSequenceClassification from HG\n"
            )
        else:
            print(
                f"\n=> Fine tuned RobertaForSequenceClassification model provided at {path}\n"
            )

        self.model = RobertaForSequenceClassification.from_pretrained(
            self.name if path is None else path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            self.name if path is None else path, add_prefix_space=True
        )

        self.tokenizer.truncation_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def load(self):
        return self.model, self.tokenizer
