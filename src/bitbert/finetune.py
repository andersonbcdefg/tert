# The code here supports finetuning BERT on downstream tasks that can
# be framed as sentence or sentence pair classification (or regression).
from datasets.dataset_dict import DatasetDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from pydantic import BaseModel
from src.bitbert.bert_padding import pad_input
from tqdm.auto import tqdm
from src.bitbert.util import send_to_device
from .model import Model
from .tokenizer import Tokenizer, STARTOFTEXT, ENDOFTEXT, SEP, special_tokens
from datasets import load_dataset
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.bitbert.layers import MAPHead, ModelArgs, BitBertBlock, BertBlock
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
from modal import Image, App, gpu, Volume
from .scheduler import (
    get_one_cycle_scheduler,
    get_constant_scheduler
)
from transformers import (
    Trainer,
    TrainingArguments,
)
from .glue import GLUE_TASKS_CONFIG
# torch.set_float32_matmul_precision("high")
START_ID = special_tokens[STARTOFTEXT]
END_ID = special_tokens[ENDOFTEXT]
SEP_ID = special_tokens[SEP]

class FineTuneArgs(BaseModel):
    checkpoint: str  # path to the model checkpoint
    task_name: str = "RTE"  # name of the task to finetune on
    max_length: int = 128
    batch_size: int = 8
    dropout: float = 0.1
    epochs: int = 5
    learning_rate: float = 1.0e-3
    schedule: Literal["one_cycle", "constant"] = "one_cycle"
    weight_decay: float = 0.01

class SequenceClassifierOutput(dict):
    def __init__(self, loss=None, logits=None, hidden_states=None, attentions=None, acc=None):
        super().__init__()
        self['acc'] = acc
        self["loss"] = loss
        self["logits"] = logits
        self["hidden_states"] = hidden_states
        self["attentions"] = attentions

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"'SequenceClassifierOutput' object has no attribute '{key}'"
            )

    def __setattr__(self, key, value):
        self[key] = value


# Supports binary, multiclass, and regression tasks
# since we trained with no pooler head, we just use the raw STARTOFTEXT token
class BERTClassifier(nn.Module):
    def __init__(
        self,
        bert,
        num_classes,
        batch_size: int,
        max_seq_len: int,
        dropout=0.1,
        pooling_strategy: Literal["mean", "first", "max", "map"] = "first",
    ):
        super().__init__()
        self.bert: Model = bert
        if pooling_strategy == "map":
            self.pooler = MAPHead(bert.args.d_model, 4 * bert.args.d_model)
        self.bert.output = None # remove the linear head
        print(
            "Initialized backbone with",
            sum(p.numel() for p in self.bert.parameters()) / 1e6,
            "M parameters",
        )
        self.dropout_p = dropout
        self.output_head = nn.Linear(bert.args.d_model, num_classes)
        self.pooling_strategy = pooling_strategy
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        nn.init.trunc_normal_(self.output_head.weight, std=0.002)
        print(
            "Initialized classifier with",
            sum([p.numel() for n, p in self.named_parameters() if p.requires_grad]) / 1e6,
            "M parameters",
        )

    def _initialize_sep_token(
        self,
        sep_token_id: int = SEP_ID,
        token_ids_to_average: list[int] = [START_ID, END_ID],
    ):
        """
        Because the SEP token isn't used during pretraining for this model,
        we have to (or should?) initialize it to a reasonable value for finetuning.
        Default strategy is average the STARTOFTEXT and ENDOFTEXT tokens. Could also try
        setting it to the value of a period, comma, other sort of separator.
        """
        tokens_to_average_idxs = torch.tensor(token_ids_to_average, dtype=torch.long)
        self.bert.tok_embeddings.weight.data[sep_token_id] = (
            self.bert.tok_embeddings.weight[tokens_to_average_idxs].mean(dim=0)
        )

    def _pool(self, outputs, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(outputs)
        if self.pooling_strategy == "mean":
            print(outputs.shape, attention_mask.shape)
            summed = torch.sum(outputs * attention_mask.unsqueeze(-1), dim=1)
            return summed / torch.sum(attention_mask, dim=1, keepdim=True)

        elif self.pooling_strategy == "first":
            return outputs[:, 0, :]
        elif self.pooling_strategy == "max":
            raise NotImplementedError("max pooling not implemented yet")
        elif self.pooling_strategy == "map":
            return self.pooler(outputs)
        else:
            raise ValueError(f"Invalid pooling strategy {self.pooling_strategy}")

    def forward(
        self, input_ids, labels, attention_mask, return_dict: bool = False
    ):
        B, L = input_ids.shape
        dropout_p = self.dropout_p if self.training else 0.0
        input_ids, labels, position_ids, block_mask = self.bert.prepare_batch(
            input_ids, attention_mask, labels,
            ignore_labels=True # the labels are not token-wise so dont handle
        )
        outputs = self.bert.forward(
            input_ids, position_ids, block_mask, dropout_p=dropout_p
        )
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        outputs = pad_input(
            outputs.squeeze(0),
            indices, B,
            self.max_seq_len
        )
        pooled = self._pool(outputs, attention_mask)  # (bsz, hidden_size)
        pooled = F.dropout(pooled, p=dropout_p, training=self.training)
        logits = self.output_head(pooled)  # (bsz, num_classes)
        if labels is None:
            if return_dict:
                return SequenceClassifierOutput(
                    loss=None,
                    logits=logits,
                    hidden_states=None,
                    attentions=None,
                )
            else:
                return None, logits
        if self.output_head.out_features == 1:
            loss = F.mse_loss(logits.squeeze(), labels)
            acc = None
        else:
            loss = F.cross_entropy(logits, labels)
            acc = accuracy_score(
                labels.detach().cpu().numpy(),
                logits.argmax(dim=-1).detach().cpu().numpy()
            )

        if return_dict:
            return SequenceClassifierOutput(
                acc=acc,
                loss=loss,
                logits=logits,
                hidden_states=outputs,
                attentions=None,
            )
        else:
            return loss, logits


def finetune_step():
    pass


def evaluate(model, eval_dataloader, num_classes, metrics, device="cuda"):
    model.eval()
    preds = []
    labels = []
    for batch in tqdm(eval_dataloader):
        batch = send_to_device(batch, device)
        with torch.no_grad():
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch['label'],
                return_dict=True
            )
            logits = out.logits
        # if regression task, logits are used directly as predictions
        if num_classes == 1:
            preds.extend(logits.squeeze().cpu().numpy().tolist())
        else:
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
        labels.extend(batch["label"].cpu().numpy().tolist())
    result = {}
    if "matthews" in metrics:
        result["matthews"] = matthews_corrcoef(labels, preds)
    if "accuracy" in metrics:
        result["accuracy"] = accuracy_score(labels, preds)
    if "f1" in metrics:
        result["f1"] = f1_score(labels, preds)
    if "pearson" in metrics:
        result["pearson"] = pearsonr(labels, preds)[0]
    if "spearman" in metrics:
        result["spearman"] = spearmanr(labels, preds)[0]
    model.train()
    return result

def finetune_hf(args: FineTuneArgs):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # 1. get the task config
    task_config = GLUE_TASKS_CONFIG[args.task_name]
    dataset_name = task_config["dataset_name"]
    dataset_config = task_config["dataset_config"]
    input1_col = task_config["input1"]
    input2_col = task_config["input2"]
    label_col = task_config["label"]
    num_classes = task_config["num_classes"]
    metrics_list = task_config["metrics"]

    # 2. load and tokenize the dataset
    # 2. load and tokenize the dataset
    print(f"Loading the {args.task_name} dataset...")
    dataset: DatasetDict = load_dataset(dataset_name, dataset_config) # pyright: ignore
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    def tokenize_function(batch):
        if input2_col:
            return tokenizer(
                batch[input1_col],
                batch[input2_col],
                truncation=True,
                padding=True,
                max_length=args.max_length,
                return_tensors="pt"
            )
        else:
            return tokenizer(
                batch[input1_col],
                truncation=True,
                padding=True,
                max_length=args.max_length,
                return_tensors="pt"
            )
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(  # pyright: ignore
        "torch", columns=["input_ids", "attention_mask", "label"]
    )
    train_dataset = tokenized_datasets["train"]
    eval_dataset = (
        tokenized_datasets["validation"]
        if "validation" in tokenized_datasets
        else tokenized_datasets["validation_matched"]
    )

    # 3. initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint,
        num_labels=num_classes
    )
    classifier.to(device)
    classifier.train()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, # pyright: ignore
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, # pyright: ignore
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1
    )
    total_steps = (len(train_dataset) // args.batch_size + 1) * args.epochs
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    if args.schedule == "one_cycle":
        scheduler = get_one_cycle_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_frac=0.5
        )
    else:
        scheduler = get_constant_scheduler(
            optimizer,
            total_steps=total_steps
        )

    eval_results = []
    train_losses = []
    train_accs = []
    with tqdm(total=total_steps) as pbar:
        for epoch in range(args.epochs):
            for batch in train_dataloader:
                batch = send_to_device(batch, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = classifier(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["label"],
                        return_dict=True
                    )
                loss = out.loss
                acc = None # out.acc
                train_losses.append(loss.item())
                train_accs.append(None)
                loss.backward()
                # Clip the gradient norm to a maximum value of 1.0
                nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                pbar.update(1)
                pbar.set_postfix({"loss": loss.item(), "acc": acc})
            # evaluate at end of epoch
            eval_result = evaluate(classifier, eval_dataloader, num_classes, metrics_list, device)
            eval_results.append(eval_result['accuracy'])

    return {
        "eval_results": eval_results,
        "train_losses": train_losses,
        "train_accs": train_accs
    }

def finetune(args: FineTuneArgs):
    # 1. get the task config
    task_config = GLUE_TASKS_CONFIG[args.task_name]
    dataset_name = task_config["dataset_name"]
    dataset_config = task_config["dataset_config"]
    input1_col = task_config["input1"]
    input2_col = task_config["input2"]
    label_col = task_config["label"]
    num_classes = task_config["num_classes"]
    metrics_list = task_config["metrics"]

    # 2. load and tokenize the dataset
    print(f"Loading the {args.task_name} dataset...")
    dataset: DatasetDict = load_dataset(dataset_name, dataset_config) # pyright: ignore
    tokenizer = Tokenizer()
    def tokenize_function(batch):
        if input2_col:
            return tokenizer.encode_batch_pairs(
                batch[input1_col],
                batch[input2_col],
                collate_strategy="max_length",
                max_length=args.max_length
            )
        else:
            return tokenizer.encode_batch(
                batch[input1_col],
                collate_strategy="max_length",
                max_length=args.max_length
            )
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(  # pyright: ignore
        "torch", columns=["input_ids", "attention_mask", "label"]
    )
    train_dataset = tokenized_datasets["train"]
    eval_dataset = (
        tokenized_datasets["validation"]
        if "validation" in tokenized_datasets
        else tokenized_datasets["validation_matched"]
    )

    # 3. initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing model, using device: {device}")
    model = Model(max_seq_len=128, rotary_impl="cos_sin")
    if "final_model.pt" in os.listdir(args.checkpoint):
        state_dict = torch.load(os.path.join(args.checkpoint, "final_model.pt"))
    elif "model.pt" in os.listdir(args.checkpoint):
        state_dict = torch.load(os.path.join(args.checkpoint, "model.pt"))
    else:
        raise ValueError(f"Checkpoint {args.checkpoint} not found")
    model.load_state_dict(state_dict)
    classifier = BERTClassifier(
        model,
        num_classes=num_classes,
        dropout=args.dropout,
        batch_size=args.batch_size,
        max_seq_len=args.max_length,
        pooling_strategy="map" # i suspect it's hard for the model to learn to stuff information in cls token
    )
    print("dropout set to", classifier.dropout_p)
    # 91 = "|", 11 = ",", 279 = "\n\n"
    classifier._initialize_sep_token(token_ids_to_average=[279])
    classifier.to(device)
    classifier.train()

    # 4. train
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, # pyright: ignore
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, # pyright: ignore
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1
    )
    total_steps = (len(train_dataset) // args.batch_size + 1) * args.epochs
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    if args.schedule == "one_cycle":
        scheduler = get_one_cycle_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_frac=0.5
        )
    else:
        scheduler = get_constant_scheduler(
            optimizer,
            total_steps=total_steps
        )

    eval_results = []
    train_losses = []
    train_accs = []
    with tqdm(total=total_steps) as pbar:
        for epoch in range(args.epochs):
            for batch in train_dataloader:
                batch = send_to_device(batch, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = classifier(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["label"],
                        return_dict=True
                    )
                loss = out.loss
                acc = out.acc
                train_losses.append(loss.item())
                train_accs.append(acc)
                loss.backward()
                # Clip the gradient norm to a maximum value of 1.0
                nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                pbar.update(1)
                pbar.set_postfix({"loss": loss.item(), "acc": acc})
            # evaluate at end of epoch
            eval_result = evaluate(classifier, eval_dataloader, num_classes, metrics_list, device)
            eval_results.append(eval_result['accuracy'])

    return {
        "eval_results": eval_results,
        "train_losses": train_losses,
        "train_accs": train_accs
    }

    # # 8. Define the metric computation function
    # def compute_metrics(eval_pred):
    #     logits, labels = eval_pred
    #     predictions = np.argmax(logits, axis=-1)
    #     acc = accuracy_score(labels, predictions)
    #     return {"accuracy": acc}

    # # 9. Set up training arguments
    # training_args = TrainingArguments(
    #     output_dir="./results",  # Output directory
    #     num_train_epochs=5,  # Number of training epochs
    #     per_device_train_batch_size=batch_size,  # Batch size for training
    #     per_device_eval_batch_size=64,  # Batch size for evaluation
    #     evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    #     logging_dir="./logs",  # Directory for storing logs
    #     logging_steps=10,  # Log every 10 steps
    #     save_strategy="no",  # Disable saving checkpoints
    #     load_best_model_at_end=False,  # Do not load the best model at the end
    #     metric_for_best_model="accuracy",
    #     learning_rate=learning_rate,
    # )

    # # 10. Initialize the Trainer
    # trainer = Trainer(
    #     model=classifier,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     tokenizer=tokenizer,
    #     compute_metrics=compute_metrics,
    # )

    # # 11. Start training
    # print("Starting training...")
    # trainer.train()

    # # 12. Evaluate the final model on the validation set
    # print("Evaluating the final model on the validation set...")
    # eval_result = trainer.evaluate()
    # print(f"Final Validation Accuracy: {eval_result['eval_accuracy']:.4f}")
