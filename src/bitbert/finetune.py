# The code here supports finetuning BERT on downstream tasks that can
# be framed as sentence or sentence pair classification (or regression).
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from dataclasses import dataclass
from typing import Literal
from .layers import Model
from .tokenizer import (
    STARTOFTEXT,
    ENDOFTEXT,
    special_tokens
)

START_ID = special_tokens[STARTOFTEXT]
END_ID = special_tokens[ENDOFTEXT]

@dataclass
class FineTuneConfig:
    tasks: list
    num_epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    dropout: float
    metadata_file: str
    checkpoint_path: str
    tokenizer_path: str

    @classmethod
    def from_yaml(cls, path):
        import yaml
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)

# Supports binary, multiclass, and regression tasks
# since we trained with no pooler head, we just use the raw STARTOFTEXT token
class BERTClassifier(nn.Module):
    def __init__(
        self,
        bert,
        num_classes,
        dropout=0.1,
        pooling_strategy: Literal["mean", "first", "max"] = "first"
    ):
        super().__init__()
        self.bert: Model = bert
        self.dropout = nn.Dropout(dropout)
        self.output_head = nn.Linear(bert.args.d_model, num_classes)
        self.pooling_strategy = pooling_strategy
        nn.init.trunc_normal_(self.output_head.weight, std=0.002)

    def _initialize_sep_token(self, sep_token_id: int, token_ids_to_average: list[int] = [START_ID, END_ID]):
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

    def _pool(self, outputs):
        if self.pooling_strategy == "mean":
            return torch.mean(outputs, dim=1)
        elif self.pooling_strategy == "first":
            return outputs[:, 0, :]
        elif self.pooling_strategy == "max":
            return torch.max(outputs, dim=1)[0]
        else:
            raise ValueError(f"Invalid pooling strategy {self.pooling_strategy}")

    def forward(self, input_ids, targets=None, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask) # (bsz, seq_len, hidden_size)
        pooled = self._pool(outputs) # (bsz, hidden_size)
        logits = self.output_head(self.dropout(pooled)) # (bsz, num_classes)
        if targets is None:
            return logits
        if self.output_head.out_features == 1:
            loss = F.mse_loss(logits.squeeze(), targets)
        else:
            loss = F.cross_entropy(logits, targets)
        return loss
