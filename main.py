from omegaconf import OmegaConf
from A.logger import Logger
from data.dataset import AGNewsDatasetLoader
from A.models import (
    DistilBertUncased,
    BertBaseUncased,
    RobertaBase,
)
from A.evaluator import Evaluator
from B.tuners import Tuner
from B.reporter import Reporter
from C.tuners import LoraTuner

config = OmegaConf.load(open("./config v2.yaml"))

logger = Logger(logs_dir=config.logging.logs_dir)
logger.print_execution_start()


# ======================================================================================================================
# Data preprocessing
# data_train, data_val, data_test = data_preprocessing(args...)

logger.print_heading("Loading AG News Dataset")
ag_news = AGNewsDatasetLoader(config.dataset.ag_news)

print(
    f"\n=> AG News Dataset Information:\n{logger.format_dict(ag_news.information())}\n"
)

dataset = ag_news.load()

x_test = dataset["test"][ag_news.text_header]
y_test = dataset["test"][ag_news.label_header]

# ======================================================================================================================
# Task A

evaluator = Evaluator(
    x_test=x_test,
    y_test=y_test,
    label_map=ag_news.id2label,
    evaluations_dir=config.logging.evaluations_dir,
    test_outputs_dir=config.logging.test_outputs_dir,
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
    logger.print_heading(f"TASK A: Evaluating {args.name}...")

    model, tokenizer = get_model_and_tokenizer(model_name=args.model_name)
    evaluation_results = evaluator.create_evaluations(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model_name,
        task_name=args.name,
    )
    print(f"\n=> Evaluation Results:\n{logger.format_dict(evaluation_results)}\n")
    logger.log_evaluation_results(
        task=args.name, model=args.model_name, results=evaluation_results
    )


for model in config.A.models:
    if model.evaluate:
        evaluate_A(args=model)


# ======================================================================================================================
# Task B

reporter = Reporter(
    logs_dir=config.logging.fine_tuning_logs_dir, plots_dir=config.logging.plots_dir
)


def fine_tune_B(args):
    logger.print_heading(f"TASK B: Fine tuning {args.name} ({args.model_name})...")

    model, tokenizer = get_model_and_tokenizer(model_name=args.model_name)

    tokenized_dataset = ag_news.tokenize(tokenizer=tokenizer)
    tokenized_dataset = tokenized_dataset.remove_columns("text")

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

    # print(f"\n=> Weight Decay Parameter Names: {tuner.get_decay_parameter_names()}\n")
    # print(f"\n=> Number of Tunable Parameters: {tuner.get_trainable_parameters()}\n")

    time_taken = tuner.fine_tune(output_dir=args.model_dir)
    logger.log_fine_tuning_duration(
        task=args.name, model=args.model_name, time_taken=time_taken
    )


def evaluate_B(args):
    logger.print_heading(f"TASK B: Evaluating {args.name} ({args.model_name})...")

    model, tokenizer = get_model_and_tokenizer(
        model_name=args.model_name, path=args.model_dir
    )
    evaluation_results = evaluator.create_evaluations(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model_name,
        task_name=args.name,
    )

    print(f"\n=> Evaluation Results:\n{logger.format_dict(evaluation_results)}\n")
    logger.log_evaluation_results(
        task=args.name, model=args.model_name, results=evaluation_results
    )

    tuning_results = reporter.create_fine_tuning_log(
        checkpoints_dir=args.checkpoints_dir,
        model_name=args.model_name,
        task_name=args.name,
    )
    print(f"\n=> Tuning Results:\n{logger.format_dict(tuning_results)}\n")
    logger.log_fine_tuning_results(
        task=args.name, model=args.model_name, results=tuning_results
    )

    print(f"\n=> Creating Fine Tuning Plot at {config.logging.plots_dir}")
    reporter.create_fine_tuning_plots(
        checkpoints_dir=args.checkpoints_dir,
        model_name=args.model_name,
        task_name=args.name,
    )


for model in config.B.models:
    if model.fine_tune:
        fine_tune_B(args=model)
    if model.evaluate:
        evaluate_B(args=model)


# ======================================================================================================================
# Task C


def fine_tune_C(args):
    logger.print_heading(f"TASK C: Fine tuning {args.name} ({args.model_name})...")

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
        lora_target_modules=list(args.training_args.lora_config.target_modules),
    )

    # print(f"\nWeight Decay Parameter Names:\n{tuner.get_decay_parameter_names()}\n")
    # print(f"\n=>Number of Tunable Parameters: {tuner.get_trainable_parameters()}\n")

    time_taken = tuner.fine_tune(output_dir=args.model_dir)
    logger.log_fine_tuning_duration(
        task=args.name, model=args.model_name, time_taken=time_taken
    )


def evaluate_C(args):
    logger.print_heading(f"TASK C: Evaluating {args.name} ({args.model_name})...")

    model, tokenizer = get_model_and_tokenizer(
        model_name=args.model_name, path=args.model_dir
    )
    evaluation_results = evaluator.create_evaluations(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model_name,
        task_name=args.name,
    )

    print(f"\n=> Evaluation Results:\n{logger.format_dict(evaluation_results)}\n")
    logger.log_evaluation_results(
        task=args.name, model=args.model_name, results=evaluation_results
    )

    tuning_results = reporter.create_fine_tuning_log(
        checkpoints_dir=args.checkpoints_dir,
        model_name=args.model_name,
        task_name=args.name,
    )
    print(f"\n=> Tuning Results:\n{logger.format_dict(tuning_results)}\n")
    logger.log_fine_tuning_results(
        task=args.name, model=args.model_name, results=tuning_results
    )

    print(f"\n=> Creating Fine Tuning Plot at {config.logging.plots_dir}")
    reporter.create_fine_tuning_plots(
        checkpoints_dir=args.checkpoints_dir,
        model_name=args.model_name,
        task_name=args.name,
    )


for model in config.C.models:
    if model.fine_tune:
        fine_tune_C(args=model)
    if model.evaluate:
        evaluate_C(args=model)

# ======================================================================================================================
# Task D


def fine_tune_D(args):
    logger.print_heading(f"TASK D: Fine tuning {args.name} ({args.model_name})...")

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
        lora_target_modules=list(args.training_args.lora_config.target_modules),
    )

    # print(f"\nWeight Decay Parameter Names:\n{tuner.get_decay_parameter_names()}")
    # print(f"\n=>Number of Tunable Parameters: {tuner.get_trainable_parameters()}\n")

    time_taken = tuner.fine_tune(output_dir=args.model_dir)
    logger.log_fine_tuning_duration(
        task=args.name, model=args.model_name, time_taken=time_taken
    )


def evaluate_D(args):
    logger.print_heading(f"TASK D: Evaluating {args.name} ({args.model_name})...")

    model, tokenizer = get_model_and_tokenizer(
        model_name=args.model_name, path=args.model_dir
    )
    evaluation_results = evaluator.create_evaluations(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model_name,
        task_name=args.name,
    )

    print(f"\n=> Evaluation Results:\n{logger.format_dict(evaluation_results)}\n")
    logger.log_evaluation_results(
        task=args.name, model=args.model_name, results=evaluation_results
    )

    tuning_results = reporter.create_fine_tuning_log(
        checkpoints_dir=args.checkpoints_dir,
        model_name=args.model_name,
        task_name=args.name,
    )
    print(f"\n=> Tuning Results:\n{logger.format_dict(tuning_results)}\n")
    logger.log_fine_tuning_results(
        task=args.name, model=args.model_name, results=tuning_results
    )

    print(f"\n=> Creating Fine Tuning Plot at {config.logging.plots_dir}")
    reporter.create_fine_tuning_plots(
        checkpoints_dir=args.checkpoints_dir,
        model_name=args.model_name,
        task_name=args.name,
    )


for model in config.D.models:
    if model.fine_tune:
        fine_tune_D(args=model)
    if model.evaluate:
        evaluate_D(args=model)

# ======================================================================================================================
# Task E


def fine_tune_E(args):
    logger.print_heading(f"TASK E: Fine tuning {args.name} ({args.model_name})...")

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
        lora_target_modules=list(args.training_args.lora_config.target_modules),
        lora_use_rslora=True,
    )

    # print(f"\nWeight Decay Parameter Names:\n{tuner.get_decay_parameter_names()}")
    # print(f"\n=>Number of Tunable Parameters: {tuner.get_trainable_parameters()}\n")

    time_taken = tuner.fine_tune(output_dir=args.model_dir)
    logger.log_fine_tuning_duration(
        task=args.name, model=args.model_name, time_taken=time_taken
    )


def evaluate_E(args):
    logger.print_heading(f"TASK E: Evaluating {args.name} ({args.model_name})...")

    model, tokenizer = get_model_and_tokenizer(
        model_name=args.model_name, path=args.model_dir
    )
    evaluation_results = evaluator.create_evaluations(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model_name,
        task_name=args.name,
    )

    print(f"\n=> Evaluation Results:\n{logger.format_dict(evaluation_results)}\n")
    logger.log_evaluation_results(
        task=args.name, model=args.model_name, results=evaluation_results
    )

    tuning_results = reporter.create_fine_tuning_log(
        checkpoints_dir=args.checkpoints_dir,
        model_name=args.model_name,
        task_name=args.name,
    )
    print(f"\n=> Tuning Results:\n{logger.format_dict(tuning_results)}\n")
    logger.log_fine_tuning_results(
        task=args.name, model=args.model_name, results=tuning_results
    )

    print(f"\n=> Creating Fine Tuning Plot at {config.logging.plots_dir}")
    reporter.create_fine_tuning_plots(
        checkpoints_dir=args.checkpoints_dir,
        model_name=args.model_name,
        task_name=args.name,
    )


for model in config.E.models:
    if model.fine_tune:
        fine_tune_E(args=model)
    if model.evaluate:
        evaluate_E(args=model)

# ======================================================================================================================
## Print out your results with following format:
# print('TA:{},{};TB:{},{};'.format(acc_A_train, acc_A_test,
#                                                         acc_B_train, acc_B_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A_train = 'TBD'
# acc_B_test = 'TBD'

if config.logging.create_log:
    logger.create_log()

if config.logging.create_evaluations_summary:
    logger.print_heading("Evaluations Summary")
    logger.create_evaluation_summary(
        evaluations_dir=config.logging.evaluations_dir,
        output_dir=config.logging.results_dir,
    )

if config.logging.create_fine_tuning_logs_summary:
    logger.print_heading("Fine Tuning Summary")
    logger.create_fine_tuning_logs_summary(
        fine_tuning_logs_dir=config.logging.fine_tuning_logs_dir,
        output_dir=config.logging.results_dir,
    )
