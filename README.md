# DLNLP_assignment_24

**ELEC0141 Deep Learning for Natural Language Processing Assignment (2024)**

The evolution of Transformer-based models has significantly advanced natural language processing, leading to widespread adoption in various applications. This project evaluates the performance and computational efficiency of BERT, DistilBERT, and RoBERTa for text classification tasks using the AG News dataset. Implementing these models with libraries from HuggingFace, it explores methods to optimize the fine-tuning process, focusing on Parameter-Efficient Fine-Tuning (PEFT) techniques. Specifically, it employs Low-Rank Adaptation (LoRA) and its rank-stabilized variant. Experimental analysis indicates that while all models benefit from PEFT, the rank-stabilized LoRA with RoBERTa achieves the best results, offering a balanced approach to maintaining high accuracy while reducing computational costs. This study highlights the potential of advanced fine-tuning strategies to enhance the accessibility and efficiency of powerful language models in practical applications.


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
Similar to Task C, Task E aims to fine tune the models by employing LoRA (updating the *query* matrices) but with rank stablisation which has been proven to improve the effectiveness of LoRA.

### Task E - Fine Tune (with Rank Stabalised LoRA) and Evaluate Models
Similar to Task C, Task E aims to fine tune the models by employing LoRA (updating the *key* matrices) but with rank stablisation which has been proven to improve the effectiveness of LoRA.


## Results

The fine tuning time and relevant metrics have been listed below. For each task (i.e., A, B, ..., F) all three models have been used (1 = BERT, 2 = DistilBERT, 3 = RoBERTa, such that A1 means task A using BERT) along with both sets of the training arguments (*v1* and *v2*, such that B1.1 means task B using BERT and training arguments *v1*)

|   **Task**   |   **Model**    | **Time Per Epoch (s)** | **Training Accuracy** | **Validation Accuracy** | **Test Accuracy** | **Test F1 Score** |
| :----------: | :------------: | :--------------------: | :-------------------: | :---------------------: | :---------------: | :---------------: |
|   **$A0$**   | **NaiveBayes** |         **-**          |     **91.806\%**      |      **90.810\%**       |   **90.184\%**    |    **0.9015**     |
|     $A1$     |      BERT      |           -            |           -           |            -            |     21.921\%      |      0.1474       |
|     $A2$     |   DistilBERT   |           -            |           -           |            -            |     15.513\%      |      0.1413       |
|     $A3$     |    RoBERTa     |           -            |           -           |            -            |     25.000\%      |      0.1000       |
|   $B1\_v1$   |      BERT      |        1314.63         |       94.987\%        |        93.170\%         |     93.000\%      |      0.9299       |
|   $B1\_v2$   |      BERT      |        1241.95         |       95.641\%        |        94.070\%         |     93.947\%      |      0.9392       |
|   $B2\_v1$   |   DistilBERT   |         702.27         |       95.409\%        |        93.550\%         |     93.506\%      |      0.9358       |
| **$B2\_v2$** | **DistilBERT** |       **702.72**       |     **96.860\%**      |      **94.310\%**       |   **93.961\%**    |    **0.9398**     |
|   $B3\_v1$   |    RoBERTa     |        1378.03         |       79.252\%        |        78.940\%         |     78.882\%      |      0.7877       |
|   $B3\_v2$   |    RoBERTa     |        1294.05         |       95.120\%        |        94.220\%         |     93.842\%      |      0.9381       |
|   $C1\_v1$   |      BERT      |         611.05         |       94.521\%        |        93.425\%         |     93.197\%      |      0.9319       |
|   $C1\_v2$   |      BERT      |         619.81         |       92.105\%        |        91.745\%         |     91.658\%      |      0.9165       |
|   $C2\_v1$   |   DistilBERT   |         315.65         |       94.843\%        |        93.795\%         |     93.513\%      |      0.9351       |
|   $C2\_v2$   |   DistilBERT   |         335.55         |       92.838\%        |        92.545\%         |     92.461\%      |      0.9245       |
| **$C3\_v1$** |  **RoBERTa**   |       **594.98**       |     **94.760\%**      |      **94.115\%**       |   **94.039\%**    |    **0.9403**     |
|   $C3\_v2$   |    RoBERTa     |         597.71         |       93.185\%        |        92.830\%         |     92.763\%      |      0.9275       |
|   $D1\_v1$   |      BERT      |         614.20         |       94.615\%        |        93.500\%         |     93.329\%      |      0.9333       |
|   $D1\_v2$   |      BERT      |         613.01         |       91.871\%        |        91.930\%         |     91.461\%      |      0.9144       |
|   $D2\_v1$   |   DistilBERT   |         315.08         |       94.749\%        |        93.725\%         |     93.592\%      |      0.9358       |
|   $D2\_v2$   |   DistilBERT   |         327.89         |       92.397\%        |        92.150\%         |     91.921\%      |      0.9191       |
| **$D3\_v1$** |  **RoBERTa**   |       **599.78**       |     **94.890\%**      |      **94.105\%**       |   **94.092\%**    |    **0.9409**     |
|   $D3\_v2$   |    RoBERTa     |         606.70         |       93.228\%        |        92.950\%         |     92.684\%      |      0.9268       |
|   $E1\_v1$   |      BERT      |         614.06         |       95.350\%        |        93.775\%         |     93.868\%      |      0.9386       |
|   $E1\_v2$   |      BERT      |         604.15         |       93.135\%        |        92.675\%         |     92.461\%      |      0.9245       |
|   $E2\_v1$   |   DistilBERT   |         321.32         |       94.814\%        |        93.635\%         |     93.408\%      |      0.9341       |
|   $E2\_v2$   |   DistilBERT   |         318.58         |       93.203\%        |        92.825\%         |     92.934\%      |      0.9292       |
| **$E3\_v1$** |  **RoBERTa**   |       **606.75**       |     **95.103\%**      |      **94.160\%**       |   **94.145\%**    |    **0.9414**     |
|   $E3\_v2$   |    RoBERTa     |         592.01         |       93.591\%        |        93.240\%         |     92.974\%      |      0.9296       |
|   $F1\_v1$   |      BERT      |         616.29         |       95.228\%        |        93.830\%         |     93.632\%      |      0.9362       |
|   $F1\_v2$   |      BERT      |         616.75         |       92.672\%        |        92.155\%         |     92.342\%      |      0.9233       |
|   $F2\_v1$   |   DistilBERT   |         313.15         |       95.221\%        |        93.980\%         |     93.816\%      |      0.9381       |
|   $F2\_v2$   |   DistilBERT   |         314.61         |       92.855\%        |        92.395\%         |     92.382\%      |      0.9238       |
| **$F3\_v1$** |  **RoBERTa**   |       **608.32**       |     **95.346\%**      |      **94.185\%**       |   **94.237\%**    |    **0.9423**     |
|   $F3\_v2$   |    RoBERTa     |         616.78         |       93.424\%        |        93.155\%         |     92.868\%      |      0.9286       |

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

- *A large amount of disk space (~38GB) to store the checkpoints and final models during fine tuning. Additional CSV and JSON files are created for evaluation and logs, but they have been included in this repository ([results](./results/) and [logs](./logs/)).*
- *A large RAM and a good quality GPU.*
- *The estimated runtime for the entire pipeline is 67 hours, but this can vary depending on the platform/environment that the pipeline is running on.*


### Running the Demo

For the purposes of testing a subsection of the entire pipeline a demo has been setup. This demo does not perform any fine-tuning, but evaluates the performance of all the models. The following steps can be followed to run this demo.

```bash
conda env create -f environment.yml
conda activate dlnlp24

python demo.py
```

The demo will use the [demo config.yaml](./demo%20config.yaml) file to check which models need to be evaluated. Currently, only 3 tasks (*A0*, *B2_v2*, and *F3_v1*) have been enabled. To enable additional tasks, the `evaluate` parameter needs to be changed to `True`.


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
- [data/dataset.py](./data/dataset.py) - This script is used to load the AG News dataset.
- [A/](./A/)
  - [models.py](./A/models.py) - This file implements the Naive Bayes classifier, BertBaseUncased, DistilbertBaseUncased, and RobertaBase models. Classes are created for each classifier. For the language models, the `load()` function is used to return the model and the tokenizer, which can then be used in [main.py](./main.py) to perform tasks based on the configuration.
  - [evaluator.py](./A/evaluator.py) - Used to evaluate the performance of the models. It expects the test dataset from AG News, a model and the tokenizer. It generates the outputs for the test dataset, evaluates the model, and stores the evaluation results in the [evaluations directory](./results/evaluations/). For each model, it creates a CSV with the outputs for each sample, and a JSON file which contains the evaluation metrics. This file is used in the future to produce the evaluations summary.
  - [logger.py](./A/logger.py) - This script is used to keep track of logs during the execution of the main pipeline. It has functions to keep track of certain types of dicts (such as the evaluations or the fine tuning report), and then store the output at the end of the execution in the [logs directory](./logs/).
- [B/](./B/)
  - [models/](./B/models/) - Has been omitted from the repository as the file size was too big.
  - [checkpoints/](./B/checkpoints/) - Has been omitted from the repository as the file size was too big.
  - [B/plotter.py](./B/plotter.py) - This script is used to generate various plots during the project. These generated logs can be found in the [plots directory](./results/plots/). This file is used by subsequent tasks as well, since the plotting functionality would remain largly similar.
  - [B/reporter.py](./B/reporter.py) - This script is used to parse the fine-tune logs created during model training. The logs create a file called `trainer_state.json`, which contains information about each step and epoch. This file also contains the train and validation evaluations for each epoch, which the reporter extracts from the latest epoch and stores the results in the [tuning logs directory](./results/tuning_logs/). his file is used by subsequent tasks as well, since the functionality would remain largly similar.
  - [B/tuners.py](./B/tuners.py) - This file sets up the fine tuning functions for the model based on the provided arguments. This file only implements full-featured tuning in the `Trainer` class.
- [C/](./C/)
  - [models/](./C/models/) - Contains the fine-tuned models from task *C*.
  - [checkpoints/](./D/checkpoints/) - Omitted from this reposiroty due to redundant file data. The best checkpoint for each model was selected as the final model and stored in the models directory.
  - [tuners.py](./C/tuners.py) - This script is used to set up the fine-tuning using LoRA. Subsequent tasks use this file as well as the functionality is similar, but different configuration arguments are parsed to configure the `LoraTrainer` class for each task.
- [D/](./D/)
  - [models/](./D/models/) - Contains the fine-tuned models from task *D*.
  - [checkpoints/](./D/checkpoints/) - Omitted from this reposiroty due to redundant file data. The best checkpoint for each model was selected as the final model and stored in the models directory.
- [E/](./E/)
  - [models/](./E/models/) - Contains the fine-tuned models from task *E*.
  - [checkpoints/](./E/checkpoints/) - Omitted from this reposiroty due to redundant file data. The best checkpoint for each model was selected as the final model and stored in the models directory.
- [F/](./F/)
  - [models/](./F/models/) - Contains the fine-tuned models from task *F*.
  - [checkpoints/](./F/checkpoints/) - Omitted from this reposiroty due to redundant file data. The best checkpoint for each model was selected as the final model and stored in the models directory.
- [logs/](./logs/) - This directory stores the logs produced during pipeline execution.
- [results](./results/) - This directory contains the results from the pipeline execution. The base directory contains summary CSVs created from the data in the sub directories.
  - [evaluations/](./results/evaluations/) - Contains JSON files which store the evaluations for each model that has been evaluated.
  - [plots/](./results/plots/) - Contains plots that were generated during model execution (fine tuning plots) and additional summary plots used in the report.
  - [test_outputs/](./results/test_outputs/) - Contains CSV files which have contain the prediction results for each sample of the test data.
  - [tuning_logs](./results/tuning_logs/) - Contains JSON files which contain the per epoch evaluation (on both training and validation datasets) produceed during the fine-tuning process.
- [demo/](./demo/) - Contains the outputs produced by running the [demo.py](./demo.py) script.
- [environment.yml](./environment.yml) - File used to create the conda environment. Contains information about dependencies used in the project.
- [README.md](./README.md) - This file :smile:
- [.gitignore](./.gitignore) - Git ignore