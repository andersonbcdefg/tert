# The code here supports finetuning BERT on downstream tasks that can
# be framed as sentence or sentence pair classification (or regression).
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from dataclasses import dataclass
from transformers import BertModel
from typing import Literal
from .layers import Model
from .tokenizer import STARTOFTEXT, ENDOFTEXT, SEP, special_tokens
from collections import namedtuple

START_ID = special_tokens[STARTOFTEXT]
END_ID = special_tokens[ENDOFTEXT]
SEP_ID = special_tokens[SEP]


# SequenceClassifierOutput = namedtuple(
#     "SequenceClassifierOutput",
#     [
#         "loss",
#         "logits",
#         "hidden_states",
#         "attentions",
#     ],
# )
#
class SequenceClassifierOutput(dict):
  def __init__(self, loss=None, logits=None, hidden_states=None, attentions=None):
    super().__init__()
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
    dropout=0.1,
    pooling_strategy: Literal["mean", "first", "max"] = "first",
  ):
    super().__init__()
    self.bert: Model = bert
    print(
      "Initialized backbone with",
      sum(p.numel() for p in self.bert.parameters()) / 1e6,
      "M parameters",
    )
    self.dropout_p = dropout
    self.output_head = nn.Linear(bert.args.d_model, num_classes)
    self.pooling_strategy = pooling_strategy
    nn.init.trunc_normal_(self.output_head.weight, std=0.002)

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
    self.bert.tok_embeddings.weight.data[
      sep_token_id
    ] = self.bert.tok_embeddings.weight[tokens_to_average_idxs].mean(dim=0)

  def _pool(self, outputs, attention_mask):
    if attention_mask is None:
      attention_mask = torch.ones_like(outputs)
    if self.pooling_strategy == "mean":
      return torch.sum(outputs * attention_mask.unsqueeze(-1), dim=1) / torch.sum(
        attention_mask, dim=1
      )
    elif self.pooling_strategy == "first":
      return outputs[:, 0, :]
    elif self.pooling_strategy == "max":
      raise NotImplementedError("max pooling not implemented yet")
    else:
      raise ValueError(f"Invalid pooling strategy {self.pooling_strategy}")

  def forward(
    self, input_ids, labels=None, attention_mask=None, return_dict: bool = False
  ):
    outputs = self.bert.forward(
      input_ids, attention_mask, dropout_p=self.dropout_p
    )  # (bsz, seq_len, hidden_size)
    pooled = self._pool(outputs, attention_mask)  # (bsz, hidden_size)
    pooled = F.dropout(pooled, p=self.dropout_p, training=self.training)
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
    else:
      loss = F.cross_entropy(logits, labels)

    if return_dict:
      return SequenceClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs,
        attentions=None,
      )
    else:
      return loss, logits
