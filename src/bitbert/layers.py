from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    flex_attention,
    BlockMask,
)
from transformers import BertConfig
from .quantization import activation_quant, weight_quant

# from torchao.prototype.quantized_training.int8_mixed_precision import (
#     Int8MixedPrecisionTrainingConfig,
#     Int8MixedPrecisionTrainingLinear
# )
try:
    from liger_kernel.transformers.rms_norm import LigerRMSNorm as RMSNorm  # pyright: ignore
    from liger_kernel.transformers.functional import (  # pyright: ignore
        liger_cross_entropy as cross_entropy,
        liger_swiglu as swiglu_mul_kernel,
    )
    # from cut_cross_entropy import linear_cross_entropy
except ImportError:
    print("liger kernel not available, falling back on pytorch RMSNorm")
    from torch.nn import RMSNorm

    cross_entropy = None

flex_attention = torch.compile(flex_attention)
config = BertConfig.from_pretrained("bert-base-uncased")


@dataclass
class ModelArgs:
    n_vocab: int = 32768
    d_model: int = 768
    mlp_hidden_dim: int = 3072
    n_heads: int = 12
    n_layers: int = 12


class LlamaRotary(nn.Module):
    """
    rotary emb meant to be shared by all layers. once at the start of training,
    we compute freqs_cis. once per batch of data, we compute the rotation tensor
    (based just on the position ids). then the rotary emb can be used for all layers.
    because of complex ops, this can't be compiled (i think).
    """

    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        """
        this is 'precompute_freqs_cis' from meta llama
        """
        super().__init__()
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(max_seq_len, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        self.register_buffer("freqs_cis", freqs_cis)
        self.rotation_tensor = None

    def compute_rotation_tensor(
        self,
        position_ids: torch.Tensor,
    ):
        """
        once per forward, we slice into freqs_cis to get the tensor to multiply with
        the hidden states to rotate them
        """
        assert position_ids.ndim == 1, "position_ids must be a 1D tensor"
        concat_seq_len = position_ids.shape[0]
        # first, use position ids to slice freqs_cis
        rotation_tensor = self.freqs_cis[position_ids, :].to(
            position_ids.device
        )  # packed_L x D
        # then, reshape to (1, L, 1, D)
        rotation_tensor = rotation_tensor.view(
            1, concat_seq_len, 1, self.freqs_cis.shape[-1]
        )
        self.rotation_tensor = rotation_tensor.to(self.freqs_cis.device)

    def forward(
        self, xq: torch.Tensor, xk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        From meta llama: https://github.com/meta-llama/llama/blob/main/llama/model.py
        Apply rotary embeddings to input tensors using the rotation tensor computed previously.
        Args:
            xq (torch.Tensor): Query tensor to apply rotary embeddings.
            xk (torch.Tensor): Key tensor to apply rotary embeddings.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
        """
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        xq_out = torch.view_as_real(xq_ * self.rotation_tensor).flatten(3)
        xk_out = torch.view_as_real(xk_ * self.rotation_tensor).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)


class CosSinRotary(torch.nn.Module):
    """
    As far as I know, complex ops can't be torch.compiled, so there's good reason
    to consider using a version of rotary that doesn't involve complex ops.
    """

    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        position_ids = torch.arange(max_seq_len)
        freq_tensor = torch.outer(position_ids, inv_freq)
        self.register_buffer("sin", freq_tensor.sin())  # shape: seq_len, dim // 2
        self.register_buffer("cos", freq_tensor.cos())  # shape: seq_len, dim // 2

    def compute_rotation_tensor(self, position_ids: torch.Tensor):
        assert position_ids.ndim == 1, "position_ids must be a 1D tensor"
        concat_seq_len = position_ids.shape[0]
        # first, use position ids to slice freqs_cis
        sin = (
            self.sin[position_ids, :]
            .to(position_ids.device)
            .view(1, concat_seq_len, 1, -1)
        )  # 1 x packed_L x 1 x D // 2
        cos = (
            self.cos[position_ids, :]
            .to(position_ids.device)
            .view(1, concat_seq_len, 1, -1)
        )  # 1 x packed_L x 1 x D // 2
        self.rotation_tensor = (sin, cos)

    @torch.compile
    def _rotate(self, x, sin, cos):
        assert x.ndim == 4  # multihead attention
        x1, x2 = x.chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], 3).type_as(x)

    def forward(
        self, xq: torch.Tensor, xk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sin, cos = self.rotation_tensor
        return self._rotate(xq, sin, cos), self._rotate(xk, sin, cos)


class BitLinear(nn.Linear):
    """
    This is only for training, and kernel optimization is needed for efficiency.
    """

    def __init__(self, in_features, out_features, bias=False):
        if bias:
            raise ValueError("bias not supported")
        super().__init__(in_features, out_features, bias=bias)

        # initialize self with small weights
        nn.init.trunc_normal_(self.weight, std=0.002)
        self.norm = RMSNorm(in_features, eps=1e-6)

    @torch.compile
    def forward(self, input):
        """
        Args:
        x: an input tensor with shape [n, d]
        Returns:
        y: an output tensor with shape [n, d]
        """
        # d = input.shape[-1]
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

    @torch.compile
    def forward(self, x, dropout_p: float = 0.0):
        up, gate = self.linear_in(x).chunk(2, dim=-1)
        up = F.dropout(up, dropout_p, training=self.training)
        # x = F.silu(gate) * up
        # return self.linear_out(x)
        return F.dropout(  # residual dropout
            # swiglu kernel handles the silu + multiplication
            self.linear_out(swiglu_mul_kernel(gate, up)),  # pyright: ignore
            dropout_p,
            training=self.training,
        )


# class BertFp8MLP(nn.Module):
#     def __init__(self, d_model, hidden_dim):
#       super().__init__()
#       self.d_model = d_model
#       self.hidden_dim = hidden_dim
#       self.linear_in = Int8MixedPrecisionTrainingLinear(
#           d_model, 2 * hidden_dim, bias=False,
#           config=Int8MixedPrecisionTrainingConfig()
#       )
#       self.linear_out = Int8MixedPrecisionTrainingLinear(
#           hidden_dim, d_model, bias=False,
#           config=Int8MixedPrecisionTrainingConfig()
#       )
#       nn.init.trunc_normal_(self.linear_in.weight, std=0.002)
#       nn.init.trunc_normal_(self.linear_out.weight, std=0.002)

#     @torch.compile
#     def forward(self, x, dropout_p: float = 0.0):
#       up, gate = self.linear_in(x).chunk(2, dim=-1)
#       up = F.dropout(up, dropout_p, training=self.training)
#       # x = F.silu(gate) * up
#       # return self.linear_out(x)
#       return F.dropout(  # residual dropout
#         # swiglu kernel handles the silu + multiplication
#         self.linear_out(swiglu_mul_kernel(gate, up)),  # pyright: ignore
#         dropout_p,
#         training=self.training,
#       )


class BitBertMLP(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.linear_in = BitLinear(d_model, 2 * hidden_dim, bias=False)
        self.linear_out = BitLinear(hidden_dim, d_model, bias=False)

    def forward(self, x, dropout_p: float = 0.0):
        up, gate = self.linear_in(x).chunk(2, dim=-1)
        up = F.dropout(up, dropout_p, training=self.training)
        # x = F.silu(gate) * up
        # return self.linear_out(x)
        # swiglu kernel handles the silu + multiplication
        return F.dropout(  # residual dropout
            # swiglu kernel handles the silu + multiplication
            self.linear_out(swiglu_mul_kernel(gate, up)),  # pyright: ignore
            dropout_p,
            training=self.training,
        )


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

    def forward(
        self,
        hidden_states,
        mask: BlockMask | torch.Tensor,
        rotary_emb: LlamaRotary | CosSinRotary,
        dropout_p: float = 0.0,
    ):
        # we don't use dropout in attention because it's not an arg for flexattention
        xq, xk, xv = self.qkv(hidden_states).chunk(3, dim=-1)  # [total_tokens, d_model]
        B, L, D = xq.shape  # unpadded, but we unsqueezed batch dim to be 1
        # split heads, but don't transpose until after rotary emb
        xq, xk, xv = map(
            lambda x: x.view(B, L, self.n_heads, D // self.n_heads), (xq, xk, xv)
        )  # B, L, nH, D/nH
        xq, xk = rotary_emb(xq, xk)
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
        return F.dropout(  # residual dropout
            self.o(attn_out), dropout_p, training=self.training
        )


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

    def forward(
        self,
        hidden_states,
        mask: BlockMask | torch.Tensor,
        rotary_emb: LlamaRotary | CosSinRotary,
        dropout_p: float = 0.0,
    ):
        xq, xk, xv = self.qkv(hidden_states).chunk(3, dim=-1)  # [total_tokens, d_model]
        B, L, D = xq.shape  # unpadded, but we unsqueezed batch dim to be 1
        # split heads, but don't transpose until after rotary emb
        xq, xk, xv = map(
            lambda x: x.view(B, L, self.n_heads, D // self.n_heads), (xq, xk, xv)
        )
        xq, xk = rotary_emb(xq, xk)
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
            ).to(input_dtype)
        attn_out = attn_out.transpose(1, 2).flatten(2)  # pyright:ignore
        return F.dropout(  # residual dropout
            self.o(attn_out), dropout_p, training=self.training
        )


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

    def forward(
        self, x, mask, rotary_emb: LlamaRotary | CosSinRotary, dropout_p: float = 0.0
    ):
        normed1 = self.norm1(x)
        attn_out = self.attention(normed1, mask, rotary_emb, dropout_p=dropout_p)
        x = x + attn_out
        normed2 = self.norm2(x)
        mlp_out = self.mlp(normed2, dropout_p=dropout_p)
        x = x + mlp_out
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

    def forward(
        self, x, mask, rotary_emb: LlamaRotary | CosSinRotary, dropout_p: float = 0.0
    ):
        x = x + self.attention(x, mask, rotary_emb, dropout_p=dropout_p)
        x = x + self.mlp(x, dropout_p=dropout_p)
        return x
