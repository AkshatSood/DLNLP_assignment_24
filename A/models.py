from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (
    DistilBertForSequenceClassification,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    BertTokenizerFast,
    DistilBertTokenizerFast,
    RobertaTokenizerFast,
)


class NaiveBayes:

    def __init__(
        self,
        vect_max_features: int = 20000,
        vect_ngram_range: tuple = (1, 1),
        vect_use_idf: bool = True,
        clf_alpha: float = 0.1,
        gridsearch: bool = False,
        gs_vect_max_features: list = [5000, 10000, 15000, 20000],
        gs_vect_ngram_range: list = [(1, 1), (1, 2), (2, 2)],
        gs_vect_use_idf: list = [True, False],
        gs_clf_alpha: list = [0.1, 0.5, 1.0],
        gs_cv: int = 5,
        gs_verbose: int = 5,
    ):

        self.gridsearch = gridsearch
        if gridsearch:
            pipeline = Pipeline([("vect", TfidfVectorizer()), ("clf", MultinomialNB())])

            parameters = {
                "vect__max_features": gs_vect_max_features,
                "vect__ngram_range": gs_vect_ngram_range,
                "vect__use_idf": gs_vect_use_idf,
                "clf__alpha": gs_clf_alpha,
            }

            self.model = GridSearchCV(
                pipeline, parameters, cv=gs_cv, verbose=gs_verbose, n_jobs=-1
            )

        else:
            self.model = make_pipeline(
                TfidfVectorizer(
                    max_features=vect_max_features,
                    ngram_range=vect_ngram_range,
                    use_idf=vect_use_idf,
                ),
                MultinomialNB(alpha=clf_alpha),
            )

    def fit(self, x_train: list, y_train: list) -> None:
        if self.gridsearch:
            print("\n=>Performing Grid Search Cross Validation\n")

        self.model.fit(x_train, y_train)

        if self.gridsearch:
            print(f"\n=> Best Parameters from Grid Search: {self.model.best_params_}")

    def predict(self, x: list) -> list:
        return self.model.predict(x)


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
