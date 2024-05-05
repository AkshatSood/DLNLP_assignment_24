from omegaconf import OmegaConf
from dataset import AGNewsDatasetLoader
from A.models import (
    DistilBertUncased,
    BertBaseUncased,
    BertLargeUncased,
    CamembertBase,
    CamembertLarge,
)
from A.evaluator import Evaluator
from B.tuners import LoraTuner


def __print(text):
    print("\n\n#######################################################################")
    print(f"=> {text}")
    print("#######################################################################\n\n")


config = OmegaConf.load(open("./config.yaml"))

# ======================================================================================================================
# Data preprocessing
# data_train, data_val, data_test = data_preprocessing(args...)

ag_news = AGNewsDatasetLoader(config.dataset.ag_news)
# print(ag_news.information())

dataset = ag_news.load()

x_test = dataset["test"][ag_news.text_header]
y_test = dataset["test"][ag_news.label_header]

# ======================================================================================================================
# Task A
# model_A = A(args...)                 # Build model object.
# acc_A_train = model_A.train(args...) # Train model based on the training set (you should fine-tune your model based on validation set.)
# acc_A_test = model_A.test(args...)   # Test model based on the test set.
# Clean up memory/GPU etc...             # Some code to free memory if necessary.

evaluator = Evaluator(
    x_test=x_test,
    y_test=y_test,
    label_map=ag_news.id2label,
    output_dir=config.A.output_dir,
)


def evaluate_distilbert_base_uncased():
    __print(f"Evaluating {config.A.distilbert_base_uncased.name}...")

    model, tokenizer = DistilBertUncased(
        num_labels=ag_news.num_labels,
        id2label=ag_news.id2label,
        label2id=ag_news.label2id,
    ).load()

    evaluator.create_evaluations(
        model=model, tokenizer=tokenizer, name=config.A.distilbert_base_uncased.name
    )


def evaluate_bert_base_uncased():
    __print(f"Evaluating {config.A.bert_base_uncased.name}...")

    model, tokenizer = BertBaseUncased(
        num_labels=ag_news.num_labels,
        id2label=ag_news.id2label,
        label2id=ag_news.label2id,
    ).load()

    evaluator.create_evaluations(
        model=model, tokenizer=tokenizer, name=config.A.bert_base_uncased.name
    )


def evaluate_bert_large_uncased():
    __print(f"Evaluating {config.A.bert_large_uncased.name}...")

    model, tokenizer = BertLargeUncased(
        num_labels=ag_news.num_labels,
        id2label=ag_news.id2label,
        label2id=ag_news.label2id,
    ).load()

    evaluator.create_evaluations(
        model=model, tokenizer=tokenizer, name=config.A.bert_large_uncased.name
    )


def evaluate_camembert_base():
    __print(f"Evaluating {config.A.camembert_base.name}...")

    model, tokenizer = CamembertBase(
        num_labels=ag_news.num_labels,
        id2label=ag_news.id2label,
        label2id=ag_news.label2id,
    ).load()

    evaluator.create_evaluations(
        model=model, tokenizer=tokenizer, name=config.A.camembert_base.name
    )


if config.A.distilbert_base_uncased.evaluate:
    evaluate_distilbert_base_uncased()

if config.A.bert_base_uncased.evaluate:
    evaluate_bert_base_uncased()

if config.A.bert_large_uncased.evaluate:
    evaluate_bert_large_uncased()

if config.A.camembert_base.evaluate:
    evaluate_camembert_base()


# ======================================================================================================================
# Task B
# model_B = B(args...)
# acc_B_train = model_B.train(args...)
# acc_B_test = model_B.test(args...)
# Clean up memory/GPU etc...


# tokenized_dataset = ag_news.tokenize(tokenizer=tokenizer)

# print(tokenized_dataset)


# tuner = LoraTuner(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=tokenized_dataset["train"],
#     eval_dataset=tokenized_dataset["validation"],
#     checkpoints_dir="./B/checkpoints/BertBaseUncased",
#     logs_dir="./B/logs",
# )

# print(tuner.get_trainable_parameters())

# tuner.fine_tune(output_dir="./B/models/BertBaseUncased")

# ======================================================================================================================
## Print out your results with following format:
# print('TA:{},{};TB:{},{};'.format(acc_A_train, acc_A_test,
#                                                         acc_B_train, acc_B_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A_train = 'TBD'
# acc_B_test = 'TBD'
