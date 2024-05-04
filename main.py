from dataset import AGNewsDatasetLoader
from A.models import DistilBertUncased
from B.tuners import LoraTuner

# ======================================================================================================================
# Data preprocessing
# data_train, data_val, data_test = data_preprocessing(args...)

ag_news_loader = AGNewsDatasetLoader()
dataset = ag_news_loader.load()

x_test = dataset["test"][ag_news_loader.text_header]
y_test = dataset["test"][ag_news_loader.label_header]

# ======================================================================================================================
# Task A
# model_A = A(args...)                 # Build model object.
# acc_A_train = model_A.train(args...) # Train model based on the training set (you should fine-tune your model based on validation set.)
# acc_A_test = model_A.test(args...)   # Test model based on the test set.
# Clean up memory/GPU etc...             # Some code to free memory if necessary.

model, tokenizer = DistilBertUncased(
    num_labels=4, id2label=ag_news_loader.id2label, label2id=ag_news_loader.label2id
).load()

# ======================================================================================================================
# Task B
# model_B = B(args...)
# acc_B_train = model_B.train(args...)
# acc_B_test = model_B.test(args...)
# Clean up memory/GPU etc...


tokenized_dataset = ag_news_loader.tokenize(tokenizer=tokenizer)

print(tokenized_dataset)


tuner = LoraTuner(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    checkpoints_dir="./B/checkpoints/DistilBertUncased",
)

tuner.fine_tune(output_dir="./B/models/DistilBertUncased")

# ======================================================================================================================
## Print out your results with following format:
# print('TA:{},{};TB:{},{};'.format(acc_A_train, acc_A_test,
#                                                         acc_B_train, acc_B_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A_train = 'TBD'
# acc_B_test = 'TBD'
