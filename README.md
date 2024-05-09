# DLNLP_assignment_24
ELEC0141 Deep Learning for Natural Language Processing Assignment (2024)

## Dataset

The [ag_news](https://huggingface.co/datasets/ag_news) dataset was used. This dataset consists of samples of news articles and corresponding classification into popular news categories, namely *World*, *Sports*, *Business*, and *Sci/Tech*.

This dataset can be loaded into the project using the [dataset.py](./dataset.py) script. The dataset provided on [HuggingFace](https://huggingface.co/) contains a training and a test split. This script creates a validation split from the provided training split to create the following splits:

- **Training Set**: 100,000 samples (25,000 per category)
- **Validation Set**: 20,000 samples (4,000 per category)
- **Test Set** 7,600 samples (1,900 per category)

Some samples from the dataset have been provided below:

| label | category | text                                                                                                                                                                                                                                                                     |
| :---: | :------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|   0   |  World   | Democratic Senator Urges Energy Reform (AP) AP - Congress must pass legislation to protect the nation's electricity grid if it wants to avoid repeats of the devastating outages that rolled across eight states last year, Sen. Maria Cantwell, D-Wash., said Saturday. |
|   1   |  Sports  | Tiger Runs Out of Steam After Storming Start KOHLER, Wisconsin (Reuters) - Tiger Woods failed to make the most of a red-hot start in the U.S. PGA Championship third round on Saturday, having to settle for a three-under-par 69.                                       |
|   2   | Business | Oil and Economy Cloud Stocks' Outlook (Reuters) Reuters - Soaring crude prices plus worries\about the economy and the outlook for earnings are expected to\hang over the stock market next week during the depth of the\summer doldrums.                                 |
|   3   | Sci/Tech | Gene Blocker Turns Monkeys Into Workaholics - Study (Reuters) Reuters - Procrastinating monkeys were turned\into workaholics using a gene treatment to block a key brain\compound, U.S. researchers reported on Wednesday.                                               |

## Tasks

In order to evaluate the performance of various models and fine tuning techniques, the project code has been devided into various tasks.

### Task A - Load and Evaluate Pretrained SequenceClassificationModels

This task aims to load the following pretrained models (from [HuggingFace](https://huggingface.co/)) which need to be fine tuned on the text classification task.

- **BERT** ([BertForSequenceClassification](https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertForSequenceClassification))
- **DistilBERT** ([DistilBertForSequenceClassification](https://huggingface.co/docs/transformers/en/model_doc/distilbert#transformers.DistilBertForSequenceClassification))
- **RoBERTa** ([RobertaForSequenceClassification](https://huggingface.co/docs/transformers/en/model_doc/roberta#transformers.RobertaForSequenceClassification))

It also sets up an *evaluation* class which can be used to evaluate this non fine tuned models and the fine tuned models created in subsequent tasks. As these pre trained models have not been fine tuned they are not expected to perform poorly.

### Task B - Fine Tune and Evaluate Models

This task aims to fine tune the 3 models using [HG Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer). Due to time and hardware limitations, limited hyperparameter testing was done.

The various *training arguments* used during this task have been listed below. More information on these can be found [here](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments).

|       | Learning Rate | Per device train batch size | Per device eval batch size | Weight decay | Epochs |
| :---: | :-----------: | :-------------------------: | :------------------------: | :----------: | :----: |
|  v1   |     5e-5      |              8              |             8              |     0.01     |   5    |
|  v2   |     1e-5      |              8              |             8              |     0.01     |   5    |



### Task C - Fine Tune (with LoRA) and Evaluate Models

Aims to fine tune the models by employing Low-Rank Adaptation ([LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)). LoRA the fine-tuning process more efficient by reducing the number of trainable parameters in the base model. LoRA can selectively update different components of the attention mechanism (such as the query, key, and value matrices) and make an impact on the model's performance and adaptability.

Task C focuses on updating the *query* matrices of the attention mechanism. Queries are used to generate the attention score by interacting with *keys* to determine the relevance or importance of other parts in the input sequence. This can refine the model’s ability to focus on different parts of the input based on context and can be particularly useful for tasks requiring nuanced understanding or context-sensitive decisions.

### Task D - Fine Tune (with LoRA) and Evaluate Models
Similar to Task C, Task D aims to fine tune the models by employing LoRA. Instead of focusing on the *query* matrices, Task D focuses on training the *key* matrices. This can make the model more or less sensitive to specific features in the input data. This is useful in tasks where certain input features need to be emphasized or suppressed.

### Task E - Fine Tune (with Rank Stabalised LoRA) and Evaluate Models
Similar to Task C, Task E aims to fine tune the models by employing LoRA but with rank stablisation which has been proven to improve the effectiveness of LoRA.


## Results

The fine tuning time and relevant metrics have been listed below. For each task (i.e., A, B, ..., E) all three models have been used (1 = BERT, 2 = DistilBERT, 3 = RoBERTa, such that A1 means task A using BERT) along with both sets of the training arguments (*v1* and *v2*, such that B1.1 means task B using BERT and training arguments *v1*)

## Setup and Execution

The conda environment can be created using the [environment.yml](./environment.yml) file provided. The following commands can be run in order to create and access the environment.

```bash
conda env create -f environment.yml
conda activate dlnlp24
```

Once in the conda environment, the entire pipeline can be executed from [main.py](./main.py), using the following command. Based on which tasks (and corresponding subtasks) you wish to run, the [config.yaml](./config.yaml) file will need to be configured as well. This file is read at the start of the program execution and tasks are performed based on it. Fine tuning hyperparameters can also be updated for each task via this file.

```bash
python main.py
```



_It is important to note that the following would be required to run the entire pipeline_

- *A large amount of disk space (~40GB) to store the checkpoints and final models during fine tuning. Additional CSV and JSON files are created for evaluation and logs, but they have been included in this repository ([here](./results/)).*
- *A large RAM and a good quality GPU.*
- *The estimated runtime for the entire pipeline is (TODO), but this can vary depending on the platform that the pipeline is running on.*



## Project Structure

The entire code structure along with the functions of each file have been provided below.

- [main.py](./main.py) - Runs the entire pipeline for the project based on the configuration values provided.
- [config.yaml](./config.yaml) - Lists the configuration arguments for each task and subtask executed in the project. Some key configuration arguments have been detailed below:
  - *logs_dir*: The directory where the logs for each task will be stored.
  - *model_name*: Used to select the base model which will be fine tuned in the task
  - *fine_tune* (bool): If set to True, the model specified model will be fine tuned using the *training_args*. During the fine tuning process, checkpoints will saved in the *checkpoints_dir* and the selected model will be saved in the *model_dir*. If set to False, this step will be skipped.
  - *evaluate* (bool): If set to True, the fine tuned model (from the *model_dir*) will be evaluated against the test dataset and the results will be stored in the *results_dir*. If set to False, this step will be skipped.
  - *parse_logs* (bool): If set to True, the checkpoints created during fine tuning (in the *checkpoints_dir*) will be used to create a log for each epochs loss and accuracy in the *logs_dir*. If set to False, this step will be skipped.
  - *training_args*: Used to set the training arguments for fine tuning.
- [environment.yml](./environment.yml) - File used to create the conda environment. Contains information about dependencies used in the project.
- [README.md](./README.md) - This file :smile:
- [.gitignore](./.gitignore) - Git ignore