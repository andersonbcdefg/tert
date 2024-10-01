from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    BlockMask,
)
flex_attention = torch.compile(flex_attention)
from .bert_padding import (
    pad_input,
    unpad_input,
    document_ids_from_cu_seqlens,
    get_block_mask
)
from typing import Any
from transformers.models.bert.modeling_bert import BertAttention as HFBertAttention
from transformers import BertConfig
try:
    from liger_kernel.transformers.rms_norm import LigerRMSNorm as RMSNorm
    from liger_kernel.transformers.functional import liger_cross_entropy as cross_entropy
except ImportError:
    print("liger kernel not available, falling back on pytorch RMSNorm")
    from torch.nn import RMSNorm
    cross_entropy = None

config = BertConfig.from_pretrained("bert-base-uncased")


@dataclass
class ModelArgs:
    n_vocab: int = 32768
    d_model: int = 768
    mlp_hidden_dim: int = 3072
    n_heads: int = 12
    n_layers: int = 12


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    From meta llama: https://github.com/meta-llama/llama/blob/main/llama/model.py
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    From meta llama: https://github.com/meta-llama/llama/blob/main/llama/model.py

    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim, "x must have at least 2 dimensions"
    # make sure freqs_cis is L x D
    assert freqs_cis.shape == (
        x.shape[1],
        x.shape[-1],
    ), f"freqs_cis shape {freqs_cis.shape}, x shape {x.shape}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    From meta llama: https://github.com/meta-llama/llama/blob/main/llama/model.py

    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # should probably reset RoPE for each document in the concatenated sequence
    # by using position_ids
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


@torch.compile
def activation_quant(x):
    """Per−token quantization to 8 bits. No grouping is needed for quantization.
    Args:
    x: an activation tensor with shape [n, d]
    Returns:
    y: a quantized activation tensor with shape [n, d]
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


@torch.compile
def weight_quant(w):
    """Per−tensor quantization to 1.58 bits. No grouping is needed for quantization.
    Args:
    w: a weight tensor with shape [d, k]
    Returns:
    u: a quantized weight with shape [d, k]
    """
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u


class BitLinear(nn.Linear):
    """
    This is only for training, and kernel optimization is needed for efficiency.
    """

    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)

        # initialize self with small weights
        nn.init.trunc_normal_(self.weight, std=0.002)
        self.norm = RMSNorm(in_features, eps=1e-6)

    def forward(self, input):
        """
        Args:
        x: an input tensor with shape [n, d]
        Returns:
        y: an output tensor with shape [n, d]
        """
        d = input.shape[-1]
        w = self.weight  # a weight tensor with shape [d, k]
        x_norm = self.norm(input)
        # A trick for implementing Straight−Through−Estimator (STE) using detach()
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        # print("x", x_quant)
        w_quant = w + (weight_quant(w) - w).detach()
        # print("w", w_quant)
        y = F.linear(x_quant, w_quant)
        return y


class BertMLP(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.linear_in = nn.Linear(d_model, 2 * hidden_dim, bias=False)
        self.linear_out = nn.Linear(hidden_dim, d_model, bias=False)
        nn.init.trunc_normal_(self.linear_in.weight, std=0.002)
        nn.init.trunc_normal_(self.linear_out.weight, std=0.002)

    def forward(self, x):
        up, gate = self.linear_in(x).chunk(2, dim=-1)
        x = F.silu(gate) * up
        return self.linear_out(x)


class BitBertMLP(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.linear_in = BitLinear(d_model, 2 * hidden_dim)
        self.linear_out = BitLinear(hidden_dim, d_model)

    def forward(self, x):
        up, gate = self.linear_in(x).chunk(2, dim=-1)
        x = F.silu(gate) * up
        return self.linear_out(x)


class BertSelfAttention(nn.Module):
    """
    Self-attention will operate on UNPADDED inputs, so it's 1 long sequence.
    This means we have to use a block mask so sequences don't attend to each other.
    """

    def __init__(self, d_model, n_heads, use_flex_attention: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.use_flex_attention = use_flex_attention
        nn.init.trunc_normal_(self.qkv.weight, std=0.002)
        nn.init.trunc_normal_(self.o.weight, std=0.002)

    def forward(self, hidden_states, mask: BlockMask | torch.Tensor, freqs_cis):
        xq, xk, xv = self.qkv(hidden_states).chunk(3, dim=-1)  # [total_tokens, d_model]
        B, L, D = xq.shape  # unpadded, but we unsqueezed batch dim to be 1
        # split heads, but don't transpose until after rotary emb
        xq, xk, xv = map(
            lambda x: x.view(B, L, self.n_heads, D // self.n_heads), (xq, xk, xv)
        )
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        if self.use_flex_attention:
            attn_out = flex_attention(  # expects B, H, L, D
                xq.transpose(1, 2).contiguous(),
                xk.transpose(1, 2).contiguous(),
                xv.transpose(1, 2).contiguous(),
                block_mask=mask,  # pyright:ignore
            )
        else:
            input_dtype = xq.dtype
            attn_out = F.scaled_dot_product_attention(
                xq.transpose(1, 2),
                xk.transpose(1, 2),
                xv.transpose(1, 2),
                # attn_mask=mask.bool().unsqueeze(1).unsqueeze(3).clone() # pyright:ignore
            ).to(input_dtype)
        attn_out = attn_out.transpose(1, 2).flatten(2)  # pyright:ignore
        return self.o(attn_out)


class BitBertSelfAttention(nn.Module):
    """
    Self-attention will operate on UNPADDED inputs, so it's 1 long sequence.
    This means we have to use a block mask so sequences don't attend to each other.
    """

    def __init__(self, d_model, n_heads, use_flex_attention: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_model = d_model
        self.qkv = BitLinear(d_model, 3 * d_model)
        self.o = BitLinear(d_model, d_model)
        self.use_flex_attention = use_flex_attention

    def forward(self, hidden_states, mask: BlockMask | torch.Tensor, freqs_cis):
        xq, xk, xv = self.qkv(hidden_states).chunk(3, dim=-1)  # [total_tokens, d_model]
        B, L, D = xq.shape  # unpadded, but we unsqueezed batch dim to be 1
        # split heads, but don't transpose until after rotary emb
        xq, xk, xv = map(
            lambda x: x.view(B, L, self.n_heads, D // self.n_heads), (xq, xk, xv)
        )
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        if self.use_flex_attention:
            attn_out = flex_attention(  # expects B, H, L, D
                xq.transpose(1, 2),
                xk.transpose(1, 2),
                xv.transpose(1, 2),
                block_mask=mask,  # pyright:ignore
            )
        else:
            input_dtype = xq.dtype
            attn_out = F.scaled_dot_product_attention(
                xq.transpose(1, 2),
                xk.transpose(1, 2),
                xv.transpose(1, 2),
                attn_mask=mask.bool().unsqueeze(1).unsqueeze(3),  # pyright:ignore
            )
        attn_out = attn_out.transpose(1, 2).flatten(2)  # pyright:ignore
        return self.o(attn_out)


class BertBlock(nn.Module):
    """
    Vanilla BERT block with pre-RMSNorm.
    """

    def __init__(self, args: ModelArgs, use_flex_attention: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(args.d_model)
        self.attention = BertSelfAttention(
            args.d_model, args.n_heads, use_flex_attention
        )
        self.norm2 = RMSNorm(args.d_model)
        self.mlp = BertMLP(args.d_model, 3 * args.d_model)

    def forward(self, x, mask, freqs_cis):
        x = x + self.attention(self.norm1(x), mask, freqs_cis)
        x = x + self.mlp(self.norm2(x))
        return x

class BitBertBlock(nn.Module):
    """
    Basically a transformer block, but eschew the layernorm/rmsnorms because they're
    included in the BitLinear layer.
    """

    def __init__(self, args: ModelArgs, use_flex_attention: bool = False):
        super().__init__()
        self.attention = BitBertSelfAttention(
            args.d_model, args.n_heads, use_flex_attention
        )
        self.mlp = BitBertMLP(args.d_model, 3 * args.d_model)

    def forward(self, x, mask: BlockMask | torch.Tensor, freqs_cis):
        x = x + self.attention(x, mask, freqs_cis)
        x = x + self.mlp(x)
        return x


class Model(nn.Module):
    def __init__(
        self,
        block_class: Any,
        args: ModelArgs,
        max_seq_len: int = 512,
        max_batch_size: int = 1024,
        use_flex_attention: bool = True,
    ):
        super().__init__()
        self.tok_embeddings = nn.Embedding(args.n_vocab, args.d_model)
        self.layers = nn.ModuleList(
            [block_class(args, use_flex_attention) for _ in range(args.n_layers)]
        )
        self.norm = nn.RMSNorm(normalized_shape=[args.d_model], elementwise_affine=True)
        self.output = nn.Linear(args.d_model, args.n_vocab, bias=False)
        self.use_flex_attention = use_flex_attention
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                args.d_model // args.n_heads, end=max_seq_len * max_batch_size
            ),
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        apply_lm_head: bool = False
    ):
        orig_batch_size, orig_max_seqlen = input_ids.shape

        hidden_states = self.tok_embeddings(input_ids)  # 1, total_tokens, d_model
        # print("hidden states dtype", hidden_states.dtype)
        # NOW unpad (it expects it to already have a hidden dim)
        if self.use_flex_attention:
            hidden_states, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(
                hidden_states,
                attention_mask,  # unpad requires a third dimension
            )
            # add batch size dim of 1 to make flex_attention happy
            hidden_states = hidden_states.unsqueeze(0)  # pyright:ignore # (1, total_tokens)
            mask = get_block_mask(attention_mask)
        else:
            mask = attention_mask
        total_seq_len = hidden_states.shape[1]
        freqs_cis = self.freqs_cis[:total_seq_len, :]
        for i, layer in enumerate(self.layers):
            # print(f"layer {i}")
            hidden_states = layer(hidden_states, mask, freqs_cis)
        hidden_states = self.norm(hidden_states) # (total_tokens, d_model) or (batch, seq_len, d_model)

        # if labels present, compute loss
        if labels is not None:
            if self.use_flex_attention:
                hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1]) # (total_tokens, d_model)
                labels_flat = labels[attention_mask.bool()].view(-1) # (total_tokens)

                # only do lm_head on tokens that will be predicted
                hidden_states_to_predict = hidden_states_flat[labels_flat != -100]
                labels_to_predict = labels_flat[labels_flat != -100]
            else:
                hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1]) # (bsz x seqlen, d_model)
                labels_flat = labels.view(-1) # (bsz x seqlen)

                # only do lm_head on tokens that will be predicted. we already ensure padding tokens are never targets
                hidden_states_to_predict = hidden_states_flat[labels_flat != -100]
                labels_to_predict = labels_flat[labels_flat != -100]

            output = self.output(hidden_states_to_predict)
            loss = cross_entropy( # pyright: ignore
                output,
                labels_to_predict,
                -100, 0.0, 'mean'
            )
            return loss


        # dont forget to add sparse token prediction (Only predict MASK tokens during training)
        if apply_lm_head:
            output = self.output(hidden_states).squeeze(0)  # (total_tokens, vocab_size)
        else:
            output = hidden_states

        # pad output
        print("output shape", output.shape)
        if self.use_flex_attention:
            output = pad_input(output, indices, orig_batch_size, orig_max_seqlen) # pyright: ignore
        return output
