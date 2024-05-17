from omegaconf import OmegaConf
from data.dataset import AGNewsDatasetLoader
from A.logger import Logger
from A.evaluator import Evaluator
from A.models import NaiveBayes, BertBaseUncased, DistilBertUncased, RobertaBase

# Read the config file
config = OmegaConf.load(open("./demo config.yaml"))

# Set up logger
logger = Logger(logs_dir=config.logging.logs_dir)
logger.print_execution_start()

# ======================================================================================================================
# Data preprocessing
#   Load the AG News dataset

logger.print_heading("Loading AG News Dataset")
ag_news = AGNewsDatasetLoader(config.dataset.ag_news)

print(
    f"\n=> AG News Dataset Information:\n{logger.format_dict(ag_news.information())}\n"
)

dataset = ag_news.load()

x_test = dataset["test"][ag_news.text_header]
y_test = dataset["test"][ag_news.label_header]


# ======================================================================================================================
# Evaluate the models

evaluator = Evaluator(
    x_test=x_test,
    y_test=y_test,
    label_map=ag_news.id2label,
    evaluations_dir=config.logging.evaluations_dir,
    test_outputs_dir=config.logging.test_outputs_dir,
)

if config.naive_bayes.evaluate:
    logger.print_heading(f"TASK A0: Evaluating Naive Bayes Classifier...")

    x_train = dataset["train"][ag_news.text_header]
    y_train = dataset["train"][ag_news.label_header]

    # Train and evaluate the Naive Bayes Classifier
    model = NaiveBayes(gridsearch=False)

    model.fit(x_train=x_train, y_train=y_train)

    evaluation_results = evaluator.create_nb_evaluations(
        model,
        x_test=x_test,
        y_test=y_test,
        model_name=config.naive_bayes.model_name,
        task_name=config.naive_bayes.name,
        only_test=True,
    )

    print(f"\n=> Evaluation Results:\n{logger.format_dict(evaluation_results)}\n")
    logger.log_evaluation_results(
        task=config.naive_bayes.name,
        model=config.naive_bayes.model_name,
        results=evaluation_results,
    )


def get_model_and_tokenizer(name: str, path: str):
    if name == "BERT":
        return BertBaseUncased(
            num_labels=ag_news.num_labels,
            id2label=ag_news.id2label,
            label2id=ag_news.label2id,
            path=path,
        ).load()
    elif name == "DistilBERT":
        return DistilBertUncased(
            num_labels=ag_news.num_labels,
            id2label=ag_news.id2label,
            label2id=ag_news.label2id,
            path=path,
        ).load()
    elif name == "RoBERTa":
        return RobertaBase(
            num_labels=ag_news.num_labels,
            id2label=ag_news.id2label,
            label2id=ag_news.label2id,
            path=path,
        ).load()


def evaluate_model(args):
    logger.print_heading(
        f"TASK {args.name}: Evaluating {args.name} {args.model_name}..."
    )

    model, tokenizer = get_model_and_tokenizer(
        name=args.model_name, path=args.model_dir
    )
    evaluation_results = evaluator.create_evaluations(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model_name,
        task_name=args.name,
        create_outputs=False,
    )
    print(f"\n=> Evaluation Results:\n{logger.format_dict(evaluation_results)}\n")
    logger.log_evaluation_results(
        task=args.name, model=args.model_name, results=evaluation_results
    )


for model in config.models:
    if model.evaluate:
        evaluate_model(args=model)


logger.print_heading("Evaluations Summary")
logger.create_evaluation_summary(
    evaluations_dir=config.logging.evaluations_dir,
    output_dir=config.logging.results_dir,
)

logger.create_log()
