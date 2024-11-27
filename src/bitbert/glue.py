"""
Routine to download GLUE data adapted from https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
Eval adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
"""
import os
import sys
import shutil
import argparse
import tempfile
import urllib.request
import zipfile
from .layers import Model
from dataclasses import dataclass
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import yaml
from .tokenizer import Tokenizer
from .finetune import BERTClassifier
from torch.utils.data import DataLoader

# Configuration mapping for GLUE tasks
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


def download_glue(glue_metadata: dict, data_dir="glue"):
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)
  # metadata = yaml.safe_load(open(metadata_file, 'r'))
  for task in glue_metadata["tasks"]:
    print(f"Downloading and extracting {task}...")
    data_file = f"{task}.zip"
    urllib.request.urlretrieve(glue_metadata["task_urls"][task], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
      zip_ref.extractall(data_dir)
    os.remove(data_file)
    if task == "CoLA":
      # add header to CoLA train, dev
      cola_train_df = pd.read_csv(
        os.path.join(data_dir, "CoLA", "train.tsv"),
        sep="\t",
        header=None,
        names=["sentence_source", "label", "label_notes", "sentence"],
      )
      cola_eval_df = pd.read_csv(
        os.path.join(data_dir, "CoLA", "dev.tsv"),
        sep="\t",
        header=None,
        names=["sentence_source", "label", "label_notes", "sentence"],
      )
      cola_train_df.to_csv(
        os.path.join(data_dir, "CoLA", "train.tsv"), sep="\t", index=False
      )
      cola_eval_df.to_csv(
        os.path.join(data_dir, "CoLA", "dev.tsv"), sep="\t", index=False
      )
      print("Added header to CoLA train & dev.")
  print("Done!")


def parse_mnli_line(idx, line):
  items = line.strip().split("\t")
  if len(items) != 12 and len(items) != 16:
    print(f"Invalid line: {idx}", end=", ")
    return None
  premise = items[8].strip()
  hypothesis = items[9].strip()
  gold_label = items[-1].strip()
  if gold_label not in ["entailment", "contradiction", "neutral"]:
    print(f"Invalid gold label: {gold_label}")
    return None
  return {
    "premise": premise,
    "hypothesis": hypothesis,
    "label": 0 if gold_label == "entailment" else 1 if gold_label == "neutral" else 2,
  }


def parse_stsb_line(idx, line):
  items = line.strip().split("\t")
  if len(items) != 10:
    print(f"Invalid line: {idx}")
    return None
  sentence1 = items[7].strip()
  sentence2 = items[8].strip()
  score = items[-1].strip()
  try:
    score = float(score)
  except:
    print(f"Invalid label: {score}")
    return None
  return {"sentence1": sentence1, "sentence2": sentence2, "score": score}


def parse_qnli_line(idx, line):
  items = line.strip().split("\t")
  if len(items) != 4:
    print(f"Invalid line: {idx}")
    return None
  question = items[1].strip()
  sentence = items[2].strip()
  label = items[-1].strip()
  if label not in ["entailment", "not_entailment"]:
    print(f"Invalid label: {label}")
    return None
  return {
    "question": question,
    "sentence": sentence,
    "label": 0 if label == "entailment" else 1,
  }


def load_mnli(data_dir="glue", split="train"):
  records = []
  with open(os.path.join(data_dir, "MNLI", f"{split}.tsv"), "r") as f:
    lines = f.readlines()
  for idx, line in enumerate(lines):
    if idx == 0:
      continue
    record = parse_mnli_line(idx, line)
    if record is not None:
      records.append(record)
  df = pd.DataFrame.from_records(records)
  return df


def load_stsb(data_dir="glue", split="train"):
  records = []
  with open(os.path.join(data_dir, "STS-B", f"{split}.tsv"), "r") as f:
    lines = f.readlines()
  for idx, line in enumerate(lines):
    if idx == 0:
      continue
    record = parse_stsb_line(idx, line)
    if record is not None:
      records.append(record)
  df = pd.DataFrame.from_records(records)
  return df


def load_qnli(data_dir="glue", split="train"):
  records = []
  with open(os.path.join(data_dir, "QNLI", f"{split}.tsv"), "r") as f:
    lines = f.readlines()
  for idx, line in enumerate(lines):
    if idx == 0:
      continue
    record = parse_qnli_line(idx, line)
    if record is not None:
      records.append(record)
  df = pd.DataFrame.from_records(records)
  return df


def load_rte(data_dir="glue", split="train"):
  df = pd.read_csv(os.path.join(data_dir, "RTE", f"{split}.tsv"), sep="\t", header=0)
  df.label = df.label.apply(lambda x: 0 if x == "entailment" else 1)
  return df


# load data, output (sentence1, sentence2, label) lists
def load_task(task, glue_metadata, data_dir="glue", split="train"):
  if task == "MNLI":
    df = load_mnli(data_dir, split)
  elif task == "STS-B":
    df = load_stsb(data_dir, split)
  elif task == "QNLI":
    df = load_qnli(data_dir, split)
  elif task == "RTE":
    df = load_rte(data_dir, split)
  else:
    df = pd.read_csv(os.path.join(data_dir, task, f"{split}.tsv"), sep="\t", header=0)
  sentence1_key = glue_metadata["task_cols"][task]["sentence1"]
  sentence2_key = glue_metadata["task_cols"][task]["sentence2"]
  label_key = glue_metadata["task_cols"][task]["label"]
  sentence1s = df[sentence1_key].values
  sentence2s = df[sentence2_key].values if sentence2_key is not None else None
  labels = df[label_key].values
  return sentence1s, sentence2s, labels


# def test_load_data():
#     metadata = yaml.safe_load(open("glue_metadata.yaml", 'r'))
#     for task in metadata['tasks']:
#         for split in ["train", "dev", "dev_matched", "dev_mismatched"]:
#             if os.path.exists(os.path.join("glue", task, f"{split}.tsv")):
#                 print(f"Loading {task} {split}...")
#                 sentence1s, sentence2s, labels = load_task(task, metadata, split=split)
#                 dataset = FineTuneDataset(
#                     sentence1s,
#                     sentence2s,
#                     labels,
#                     metadata['num_classes'][task],
#                     max_len=512
#                 )
#                 print(f"Loaded {len(dataset)} examples")


# def finetune_and_eval(model_config, task, finetune_config, glue_metadata, tokenizer):

#     # If configs are paths, load them from yaml
#     if isinstance(finetune_config, str):
#         finetune_config = FineTuneConfig.from_yaml(finetune_config)
#     if isinstance(model_config, str):
#         model_config = BERTConfig.from_yaml(model_config)

#     # Initialize wandb
#     wandb.init(
#         project="cramming-finetune-" + task,
#         config={"ft-config":finetune_config, "model-config":model_config}
#     )

#     # Create base model & fine-tuning model
#     if finetune_config.dropout != model_config.dropout:
#         print("Warning: finetune_config.dropout != model_config.dropout, using finetune_config.dropout")
#         model_config.dropout = finetune_config.dropout
#     base_model = BERT(model_config)
#     base_model.load_weights_from_checkpoint(finetune_config.checkpoint_path)
#     model = BERTForFineTuning(base_model, glue_metadata['num_classes'][task], dropout=finetune_config.dropout)
#     model.to(device)

#     # Create optimizer and scheduler
#     optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_config.lr, weight_decay=finetune_config.weight_decay)
#     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=finetune_config.lr,
#         steps_per_epoch=len(train_dataloader), epochs=finetune_config.num_epochs, pct_start=0.0)

#     # Train model
#     print("Training!")
#     model.train()
#     step = 0
#     for epoch in range(finetune_config.num_epochs):
#         print(f"Epoch {epoch+1}/{finetune_config.num_epochs}")
#         for x, y, mask in train_dataloader:
#             step += 1
#             x, y, mask = x.to(device), y.to(device), mask.to(device)
#             optimizer.zero_grad(set_to_none=True)
#             loss = model(x, y, mask)
#             wandb.log({"train-loss": loss.item()})
#             if step % 100 == 0:
#                 print(f"Step {step}, loss: {loss.item()}")
#             loss.backward()
#             optimizer.step()
#             scheduler.step()
#         if task != "MNLI":
#             result = eval_model(model, dev_dataloader, glue_metadata['num_classes'][task], metrics=glue_metadata['metrics'][task])
#             print(f"Dev {task} results after {epoch + 1} epochs:\n{result}")
#         else:
#             result_matched = eval_model(model, dev_matched_dataloader, glue_metadata['num_classes'][task], metrics=glue_metadata['metrics'][task])
#             result_mismatched = eval_model(model, dev_mismatched_dataloader, glue_metadata['num_classes'][task], metrics=glue_metadata['metrics'][task])
#             print(f"Dev {task} results after {epoch + 1} epochs:\n{result_matched}\n{result_mismatched}")
#     wandb.finish()


# def run_glue(model_config, finetune_config):
#     # download glue if it doesn't exist
#     if isinstance(finetune_config, str):
#         finetune_config = FineTuneConfig.from_yaml(finetune_config)
#     if not os.path.exists("glue") or not os.path.exists("glue/CoLA"):
#         download_glue(metadata_file=finetune_config.metadata_file)
#     glue_metadata = yaml.safe_load(open(finetune_config.metadata_file, 'r'))

#     # load tokenizer
#     tokenizer = load_tokenizer(finetune_config.tokenizer_path)

#     for task in finetune_config.tasks:
#         finetune_and_eval(model_config, task, finetune_config, glue_metadata, tokenizer)
