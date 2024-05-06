from transformers import (
    DistilBertForSequenceClassification,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    DistilBertTokenizer,
    BertTokenizer,
)


# class ModelForSequenceClassification:

#     def __init__(self, model_name_or_path: str, tokenizer_name: str):
#         self.model = AutoModelForSequenceClassification(model_name_or_path)
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             tokenizer_name, add_prefix_space=True
#         )

#         self.tokenizer.truncation_side = "left"

#         if self.tokenizer.pad_token is None:
#             self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
#             self.model.resize_token_embeddings(len(self.tokenizer))

#     def load(self):
#         return self.model, self.tokenizer


class BertBaseUncased:
    def __init__(
        self,
        num_labels: int = 2,
        id2label: dict = None,
        label2id: dict = None,
        path=None,
    ):
        self.name = "bert-base-uncased"

        if not path is None:
            print(f"\n=> Fine tuned model provided at {path}\n")

        self.model = BertForSequenceClassification.from_pretrained(
            self.name if path is None else path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        self.tokenizer = BertTokenizer.from_pretrained(self.name, add_prefix_space=True)

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

        if not path is None:
            print(f"\n=> Fine tuned model provided at {path}\n")

        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.name if path is None else path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        self.tokenizer = DistilBertTokenizer.from_pretrained(
            self.name, add_prefix_space=True
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
        path=None,
    ):
        self.name = "roberta-base"

        if not path is None:
            print(f"\n=> Fine tuned model provided at {path}\n")

        self.model = RobertaForSequenceClassification.from_pretrained(
            self.name if path is None else path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        self.tokenizer = RobertaTokenizer.from_pretrained(
            self.name, add_prefix_space=True
        )

        self.tokenizer.truncation_side = "left"

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))

    def load(self):
        return self.model, self.tokenizer


# class BertLargeUncased:

#     def __init__(
#         self,
#         num_labels: int = 2,
#         id2label: dict = None,
#         label2id: dict = None,
#         path=None,
#     ):
#         self.name = "bert-large-uncased"

#         self.model = AutoModelForSequenceClassification.from_pretrained(
#             self.name if path is None else path,
#             num_labels=num_labels,
#             id2label=id2label,
#             label2id=label2id,
#         )

#         self.tokenizer = AutoTokenizer.from_pretrained(self.name, add_prefix_space=True)

#         self.tokenizer.truncation_side = "left"

#         if self.tokenizer.pad_token is None:
#             self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
#             self.model.resize_token_embeddings(len(self.tokenizer))

#     def load(self):
#         return self.model, self.tokenizer
