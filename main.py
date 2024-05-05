from omegaconf import OmegaConf
from dataset import AGNewsDatasetLoader
from A.models import (
    DistilBertUncased,
    BertBaseUncased,
    RobertaBase,
)
from A.evaluator import Evaluator
from B.tuners import Tuner


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


def evaluate_A_distilbert_base_uncased():
    __print(f"Evaluating {config.A.distilbert_base_uncased.name}...")

    model, tokenizer = DistilBertUncased(
        num_labels=ag_news.num_labels,
        id2label=ag_news.id2label,
        label2id=ag_news.label2id,
    ).load()

    evaluator.create_evaluations(
        model=model, tokenizer=tokenizer, name=config.A.distilbert_base_uncased.name
    )


def evaluate_A_bert_base_uncased():
    __print(f"Evaluating {config.A.bert_base_uncased.name}...")

    model, tokenizer = BertBaseUncased(
        num_labels=ag_news.num_labels,
        id2label=ag_news.id2label,
        label2id=ag_news.label2id,
    ).load()

    evaluator.create_evaluations(
        model=model, tokenizer=tokenizer, name=config.A.bert_base_uncased.name
    )


def evaluate_A_roberta_base():
    __print(f"Evaluating {config.A.roberta_base.name}...")

    model, tokenizer = RobertaBase(
        num_labels=ag_news.num_labels,
        id2label=ag_news.id2label,
        label2id=ag_news.label2id,
    ).load()

    evaluator.create_evaluations(
        model=model, tokenizer=tokenizer, name=config.A.roberta_base.name
    )


if config.A.distilbert_base_uncased.evaluate:
    evaluate_A_distilbert_base_uncased()

if config.A.bert_base_uncased.evaluate:
    evaluate_A_bert_base_uncased()

if config.A.roberta_base.evaluate:
    evaluate_A_roberta_base()


# ======================================================================================================================
# Task B
# model_B = B(args...)
# acc_B_train = model_B.train(args...)
# acc_B_test = model_B.test(args...)
# Clean up memory/GPU etc...


def fine_tune_B_bert_base_uncased(args):
    __print(f"Fine tuning {args.name}...")

    model, tokenizer = BertBaseUncased(
        num_labels=ag_news.num_labels,
        id2label=ag_news.id2label,
        label2id=ag_news.label2id,
    ).load()

    tokenized_dataset = ag_news.tokenize(tokenizer=tokenizer)

    tuner = Tuner(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        checkpoints_dir=args.checkpoints_dir,
        learning_rate=args.training_args.learning_rate,
        per_device_eval_batch_size=args.training_args.per_device_eval_batch_size,
        per_device_train_batch_size=args.training_args.per_device_train_batch_size,
        weight_decay=args.training_args.weight_decay,
        epochs=args.training_args.epochs,
    )

    print(f"\nWeight Decay Parameter Names:\n{tuner.get_decay_parameter_names()}")
    # print(f"\nNumber of Tunable Parameters:\n{tuner.get_trainable_parameters()}")

    tuner.fine_tune(output_dir=args.model_dir)


def evaluate_B_bert_base_uncased():
    pass


def fine_tune_B_distilbert_base_uncased(args):
    __print(f"Fine tuning {args.name}...")

    model, tokenizer = DistilBertUncased(
        num_labels=ag_news.num_labels,
        id2label=ag_news.id2label,
        label2id=ag_news.label2id,
    ).load()

    tokenized_dataset = ag_news.tokenize(tokenizer=tokenizer)

    tuner = Tuner(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        checkpoints_dir=args.checkpoints_dir,
        learning_rate=args.training_args.learning_rate,
        per_device_eval_batch_size=args.training_args.per_device_eval_batch_size,
        per_device_train_batch_size=args.training_args.per_device_train_batch_size,
        weight_decay=args.training_args.weight_decay,
        epochs=args.training_args.epochs,
    )

    print(f"\nWeight Decay Parameter Names:\n{tuner.get_decay_parameter_names()}")
    # print(f"\nNumber of Tunable Parameters:\n{tuner.get_trainable_parameters()}")

    tuner.fine_tune(output_dir=args.model_dir)


def evaluate_B_distilbert_base_uncased():
    pass


def fine_tune_B_roberta_base(args):
    __print(f"Fine tuning {args.name}...")

    model, tokenizer = RobertaBase(
        num_labels=ag_news.num_labels,
        id2label=ag_news.id2label,
        label2id=ag_news.label2id,
    ).load()

    tokenized_dataset = ag_news.tokenize(tokenizer=tokenizer)

    tuner = Tuner(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        checkpoints_dir=args.checkpoints_dir,
        learning_rate=args.training_args.learning_rate,
        per_device_eval_batch_size=args.training_args.per_device_eval_batch_size,
        per_device_train_batch_size=args.training_args.per_device_train_batch_size,
        weight_decay=args.training_args.weight_decay,
        epochs=args.training_args.epochs,
    )

    # print(f"\nWeight Decay Parameter Names:\n{tuner.get_decay_parameter_names()}")
    # print(f"\nNumber of Tunable Parameters:\n{tuner.get_trainable_parameters()}")

    tuner.fine_tune(output_dir=args.model_dir)


def evaluate_B_roberta_base():
    pass


if config.B.bert_base_uncased.fine_tune:
    fine_tune_B_bert_base_uncased(config.B.bert_base_uncased)

if config.B.bert_base_uncased.evaluate:
    evaluate_B_bert_base_uncased()

if config.B.distilbert_base_uncased.fine_tune:
    fine_tune_B_distilbert_base_uncased(config.B.distilbert_base_uncased)

if config.B.distilbert_base_uncased.evaluate:
    evaluate_B_distilbert_base_uncased()

if config.B.roberta_base.fine_tune:
    fine_tune_B_roberta_base(config.B.roberta_base)

if config.B.roberta_base.evaluate:
    evaluate_B_roberta_base()


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
