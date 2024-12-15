import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import BertBlock, BitBertBlock, ModelArgs, LlamaRotary, CosSinRotary
from typing import Literal
from .bert_padding import get_block_mask, pad_input, unpad_input
from .loss import linear_cross_entropy
from .util import pad_or_truncate


def compute_position_ids(
    attention_mask: torch.Tensor # (batch, max_seq, 0s and 1s
):
    seq_lengths = attention_mask.sum(dim=-1)
    # Compute position IDs for each document
    position_ids_list = [
        # using .item() causes a graph break
        torch.arange(length, device=attention_mask.device)  # pyright: ignore
        for length in seq_lengths
    ]
    position_ids = torch.cat(position_ids_list, dim=0)  # (total_valid_tokens,)

    return position_ids


block_types = {"bert": BertBlock, "bitbert": BitBertBlock}


class Model(nn.Module):
    def __init__(
        self,
        args: ModelArgs = ModelArgs(),
        block_class: BertBlock | BitBertBlock = BertBlock,  # pyright: ignore
        max_seq_len: int = 512,
        max_batch_size: int = 1024,
        use_flex_attention: bool = True,
        rotary_impl: Literal["llama", "cos_sin"] = "llama",
    ):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.n_vocab, args.d_model)
        self.layers = nn.ModuleList(
            [block_class(args, use_flex_attention) for _ in range(args.n_layers)]
        )
        self.norm = nn.RMSNorm(normalized_shape=[args.d_model], elementwise_affine=True)
        self.output = nn.Linear(args.d_model, args.n_vocab, bias=False)
        # whether to flatten everything into one long sequence with masking
        self.use_flex_attention = use_flex_attention
        if rotary_impl == "llama":
            self.rotary_emb = LlamaRotary(
                args.d_model // args.n_heads, max_seq_len=max_seq_len
            )
        else:
            self.rotary_emb = CosSinRotary(
                args.d_model // args.n_heads, max_seq_len=max_seq_len
            )

    @staticmethod
    def prepare_batch(
        input_ids: torch.Tensor, # (batch, seqlen)
        attention_mask: torch.Tensor, # (batch, seqlen)
        labels: torch.Tensor, # (batch, seqlen)
        max_flat_seq_len: int | None = None,
        ignore_labels: bool = False
    ):
        """
        separate from forward so it doesn't mess with compile
        """
        # first flatten batch into 1 long sequence
        input_ids_3d = input_ids.unsqueeze(-1)  # (batch, seqlen, 1)
        input_ids, indices, cu_seqlens = unpad_input( # pyright: ignore
            input_ids_3d, attention_mask
        )
        # print("after unpad", input_ids.shape)
        # Remove the fake dimension we added
        input_ids = input_ids.squeeze(-1).unsqueeze(0)  # (1, total_tokens) # pyright: ignore
        # print("after squeeze/unsqueeze", input_ids.shape)
        # pad or truncate input_ids to max_flat_seq_len
        if max_flat_seq_len is not None:
            input_ids = pad_or_truncate(
                input_ids, value=0, dim=1, max_length=max_flat_seq_len
            )
        # print("after pad_or_truncate", input_ids.shape)

        # pad or truncate the labels
        if ignore_labels:
            labels_flat = labels
        else:
            labels_flat = labels[attention_mask.bool()].view(-1)
            if max_flat_seq_len is not None:
                labels_flat = pad_or_truncate(
                    labels_flat, value=-100, dim=0, max_length=max_flat_seq_len
                )

        # truncate/pad cu_seqlens to max_flat_seq_len
        if max_flat_seq_len is not None:
            mask = cu_seqlens < max_flat_seq_len
            cu_seqlens = torch.cat([
                cu_seqlens[mask],
                torch.tensor([max_flat_seq_len], device=cu_seqlens.device, dtype=cu_seqlens.dtype)
            ])

        # get position ids and pad/truncate if necessary
        position_ids = compute_position_ids(attention_mask)
        if max_flat_seq_len is not None:
            position_ids = pad_or_truncate(
                position_ids, value=-1, dim=0, max_length=max_flat_seq_len
            )

        # get block mask
        block_mask = get_block_mask(cu_seqlens)

        return input_ids, labels_flat, position_ids, block_mask


    def forward(
        self,
        input_ids,
        position_ids,
        block_mask=None,
        dropout_p: float = 0.0,
    ):
        assert self.use_flex_attention, "flex attention must be enabled"
        self.rotary_emb.compute_rotation_tensor(position_ids)
        hidden_states = self.tok_embeddings(input_ids)  # 1, total_tokens, d_model
        hidden_states = F.dropout(hidden_states, dropout_p, training=self.training)
        for i, layer in enumerate(self.layers):
            # print(f"layer {i}")
            hidden_states = layer(
                hidden_states, block_mask, self.rotary_emb, dropout_p=dropout_p
            )
        hidden_states = self.norm(
            hidden_states
        )
        return hidden_states
