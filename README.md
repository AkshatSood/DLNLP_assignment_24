# DLNLP_assignment_24
ELEC0141 Deep Learning for Natural Language Processing Assignment (2024)

## Dataset

## Tasks

In order to evaluate the performance of various models and fine tuning techniques, the project code has been devided into various tasks.

### Task A - Load and Evaluate Pretrained SequenceClassificationModels

This task aims to load the following pretrained models (from [HuggingFace](https://huggingface.co/)) which need to be fine tuned on the text classification task.

- BERT ([BertForSequenceClassification](https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertForSequenceClassification))
- DistilBERT ([DistilBertForSequenceClassification](https://huggingface.co/docs/transformers/en/model_doc/distilbert#transformers.DistilBertForSequenceClassification))
- RoBERTa ([RobertaForSequenceClassification](https://huggingface.co/docs/transformers/en/model_doc/roberta#transformers.RobertaForSequenceClassification))

It also sets up an *evaluation* class which can be used to evaluate this non fine tuned models and the fine tuned models created in subsequent tasks. As these pre trained models have not been fine tuned they are not expected to perform poorly.

### Task B - Fine Tune and Evaluate Models

This task aims to fine tune the 3 models using [HG Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer). Due to time and hardware limitations, limited hyperparameter testing was done.

The various *training arguments* used during this task have been listed below. More information on these can be found [here](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments).

|       | Learning Rate | Per device train batch size | Per device eval batch size | Weight decay | Epochs |
| :---: | :-----------: | :-------------------------: | :------------------------: | :----------: | :----: |
|  v1   |     5e-5      |              8              |             8              |     0.01     |   5    |
|  v2   |               |                             |                            |              |        |



### Task C - Fine Tune (with LoRA) and Evaluate Models

Aims to fine tune the models by employing Low-Rank Adaptation ([LoRA](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)). LoRA the fine-tuning process more efficient by reducing the number of trainable parameters in the base model.



## Results

## Setup and Execution

The conda environment can be created using the [environment.yml](./environment.yml) file provided. The following commands can be run in order to create and access the environment.

```
conda env create -f environment.yml
conda activate dlnlp24
```

Once in the conda environment, the entire pipeline can be executed from [main.py](./main.py), using the following command. Based on which tasks (and corresponding subtasks) you wish to run, the [config.yaml](./config.yaml) file will need to be configured as well. This file is read at the start of the program execution and tasks are performed based on it. Fine tuning hyperparameters can also be updated for each task via this file.

```
python main.py
```

_It is important to note that a large amount of disk space is required in order to run the entire pipeline. A large RAM, and a good quality GPU will also be required. It is hard to estimate the run time of the entire pipeline, but it is expected to take multiple hours_



## Project Structure

The entire code structure along with the functions of each file have been provided below.

- [main.py](./main.py) - Runs the entire pipeline for the project based on the configuration values provided.
- [config.yaml](./config.yaml) - Lists the configuration arguments for each task and subtask executed in the project.
- [environment.yml](./environment.yml) - File used to create the conda environment. Contains information about dependencies used in the project.
- [README.md](./README.md) - This file :)
- [.gitignore](./.gitignore) - Git ignore