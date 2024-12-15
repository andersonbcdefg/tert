GLUE_TASKS_CONFIG = {
  "CoLA": {
    "dataset_name": "glue",
    "dataset_config": "cola",
    "input1": "sentence",
    "input2": None,  # Single sentence input
    "label": "label",
    "num_classes": 2,
    "metrics": ["matthews"],
  },
  "SST-2": {
    "dataset_name": "glue",
    "dataset_config": "sst2",
    "input1": "sentence",
    "input2": None,  # Single sentence input
    "label": "label",
    "num_classes": 2,
    "metrics": ["accuracy"],
  },
  "QQP": {
    "dataset_name": "glue",
    "dataset_config": "qqp",
    "input1": "question1",
    "input2": "question2",
    "label": "is_duplicate",
    "num_classes": 2,
    "metrics": ["accuracy", "f1"],
  },
  "STS-B": {
    "dataset_name": "glue",
    "dataset_config": "stsb",
    "input1": "sentence1",
    "input2": "sentence2",
    "label": "score",
    "num_classes": 1,  # Regression task
    "metrics": ["pearson"],
  },
  "MNLI": {
    "dataset_name": "glue",
    "dataset_config": "mnli",
    "input1": "premise",
    "input2": "hypothesis",
    "label": "label",
    "num_classes": 3,
    "metrics": ["accuracy"],
  },
  "QNLI": {
    "dataset_name": "glue",
    "dataset_config": "qnli",
    "input1": "question",
    "input2": "sentence",
    "label": "label",
    "num_classes": 2,
    "metrics": ["accuracy"],
  },
  "RTE": {
    "dataset_name": "glue",
    "dataset_config": "rte",
    "input1": "sentence1",
    "input2": "sentence2",
    "label": "label",
    "num_classes": 2,
    "metrics": ["accuracy"],
  },
}


# from typing import Literal
# from .tokenizer import Tokenizer
# from datasets import load_dataset

# # Configuration mapping for GLUE tasks

# def load_and_tokenize_glue_split(
#     task_name,
#     eval_split: Literal["dev", "test"],
#     max_length: int = 128
# ):
#     task_config = GLUE_TASKS_CONFIG[task_name]
#     dataset_name = task_config["dataset_name"]
#     dataset_config = task_config["dataset_config"]
#     input1_col = task_config["input1"]
#     input2_col = task_config["input2"]
#     label_col = task_config["label"]
#     num_classes = task_config["num_classes"]
#     metrics_list = task_config["metrics"]
#     dataset = load_dataset(dataset_name, dataset_config)
#     tokenizer = Tokenizer()

#     # 3. Define the tokenization function
#     def tokenize_function(batch):
#       sentence1s = batch[input1_col]
#       if not input2_col:
#         return tokenizer.encode_batch(
#           sentence1s, max_length=max_length, collate_strategy="max_length"
#         )
#       sentence2s = batch[input2_col]
#       return tokenizer.encode_batch_pairs(
#         sentence1s, sentence2s, max_length=max_length, collate_strategy="max_length"
#       )

#     # 4. Apply the tokenization to the entire dataset
#     print("Tokenizing the dataset...")
#     tokenized_datasets = dataset.map(tokenize_function, batched=True)

#     return tokenized_datasets, metrics_list, num_classes
