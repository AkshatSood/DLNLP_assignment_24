from omegaconf import OmegaConf
from dataset import AGNewsDatasetLoader
from A.models import (
    DistilBertUncased,
    BertBaseUncased,
    RobertaBase,
)
from A.evaluator import Evaluator
from B.tuners import Tuner
from C.tuners import LoraTuner

# TODO: Remove this later
from datetime import datetime

timestamp = datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
print(
    f"\n\n\n\n\n\n************* NEW EXECUTION STARTED AT {timestamp} *************\n\n\n\n\n\n\n"
)


def __print(text):
    print("\n\n\n#####################################################################")
    print(f"=> {text}")
    print("#####################################################################\n\n\n")


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
    output_dir=config.results_dir,
)


def get_model_and_tokenizer(model_name: str, path: str = None):
    if model_name == "BERT":
        return BertBaseUncased(
            num_labels=ag_news.num_labels,
            id2label=ag_news.id2label,
            label2id=ag_news.label2id,
            path=path,
        ).load()
    elif model_name == "DistilBERT":
        return DistilBertUncased(
            num_labels=ag_news.num_labels,
            id2label=ag_news.id2label,
            label2id=ag_news.label2id,
            path=path,
        ).load()
    elif model_name == "RoBERTa":
        return RobertaBase(
            num_labels=ag_news.num_labels,
            id2label=ag_news.id2label,
            label2id=ag_news.label2id,
            path=path,
        ).load()


def evaluate_A(args):
    __print(f"TASK A: Evaluating {args.name}...")

    model, tokenizer = get_model_and_tokenizer(model_name=args.model_name)
    evaluator.create_evaluations(model=model, tokenizer=tokenizer, name=args.name)


if config.A.bert_base_uncased.evaluate:
    evaluate_A(args=config.A.bert_base_uncased)

if config.A.distilbert_base_uncased.evaluate:
    evaluate_A(args=config.A.distilbert_base_uncased)

if config.A.roberta_base.evaluate:
    evaluate_A(args=config.A.roberta_base)


# ======================================================================================================================
# Task B
# model_B = B(args...)
# acc_B_train = model_B.train(args...)
# acc_B_test = model_B.test(args...)
# Clean up memory/GPU etc...


def fine_tune_B(args):
    __print(f"TASK B: Fine tuning {args.name}...")

    model, tokenizer = get_model_and_tokenizer(model_name=args.model_name)

    tokenized_dataset = ag_news.tokenize(tokenizer=tokenizer)
    # print(tokenized_dataset)
    # tokenized_dataset = tokenized_dataset.remove_columns("text")
    # print(tokenized_dataset)

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

    # print(f"\n=> Weight Decay Parameter Names: {tuner.get_decay_parameter_names()}")
    # print(f"\n=> Number of Tunable Parameters: {tuner.get_trainable_parameters()}")

    tuner.fine_tune(output_dir=args.model_dir)


def evaluate_B(args):
    __print(f"TASK B: Evaluating {args.name}...")

    model, tokenizer = get_model_and_tokenizer(
        model_name=args.model_name, path=args.model_dir
    )
    evaluator.create_evaluations(model=model, tokenizer=tokenizer, name=args.name)


if config.B.bert_base_uncased.fine_tune:
    fine_tune_B(config.B.bert_base_uncased)

if config.B.bert_base_uncased.evaluate:
    evaluate_B(config.B.bert_base_uncased)

if config.B.distilbert_base_uncased.fine_tune:
    fine_tune_B(config.B.distilbert_base_uncased)

if config.B.distilbert_base_uncased.evaluate:
    evaluate_B(config.B.distilbert_base_uncased)

if config.B.roberta_base.fine_tune:
    fine_tune_B(config.B.roberta_base)

if config.B.roberta_base.evaluate:
    evaluate_B(config.B.roberta_base)


# ======================================================================================================================
# Task C


# TODO: Fox the error with reading modules
def fine_tune_C(args, modules):
    __print(f"TASK C: Fine tuning {args.name}...")

    model, tokenizer = get_model_and_tokenizer(model_name=args.model_name)

    tokenized_dataset = ag_news.tokenize(tokenizer=tokenizer)
    tokenized_dataset = tokenized_dataset.remove_columns("text")

    tuner = LoraTuner(
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
        lora_r=args.training_args.lora_config.r,
        lora_alpha=args.training_args.lora_config.alpha,
        lora_dropout=args.training_args.lora_config.dropout,
        # lora_target_modules=args.training_args.lora_config.target_modules,
        lora_target_modules=modules,
    )

    # print(f"\nWeight Decay Parameter Names:\n{tuner.get_decay_parameter_names()}")
    print(f"\n=>Number of Tunable Parameters: {tuner.get_trainable_parameters()}\n")

    tuner.fine_tune(output_dir=args.model_dir)


def evaluate_C(args):
    __print(f"TASK C: Evaluating {args.name}...")

    model, tokenizer = get_model_and_tokenizer(
        model_name=args.model_name, path=args.model_dir
    )
    evaluator.create_evaluations(model=model, tokenizer=tokenizer, name=args.name)


if config.C.distilbert_base_uncased.fine_tune:
    fine_tune_C(args=config.C.distilbert_base_uncased, modules=["q_lin"])

if config.C.distilbert_base_uncased.evaluate:
    evaluate_C(args=config.C.distilbert_base_uncased)

if config.C.bert_base_uncased.fine_tune:
    fine_tune_C(args=config.C.bert_base_uncased, modules=["query"])

if config.C.bert_base_uncased.evaluate:
    evaluate_C(args=config.C.bert_base_uncased)

if config.C.roberta_base.fine_tune:
    fine_tune_C(args=config.C.roberta_base, modules=["query"])

if config.C.roberta_base.evaluate:
    evaluate_C(args=config.C.roberta_base)

# ======================================================================================================================
## Print out your results with following format:
# print('TA:{},{};TB:{},{};'.format(acc_A_train, acc_A_test,
#                                                         acc_B_train, acc_B_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A_train = 'TBD'
# acc_B_test = 'TBD'
