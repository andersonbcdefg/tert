import torch
import torch.nn.functional as F

try:
    from liger_kernel.transformers.functional import (
        liger_cross_entropy as liger_cross_entropy_fn,
        liger_fused_linear_cross_entropy as liger_fused_linear_cross_entropy_fn,
    )
except ImportError:
    liger_cross_entropy_fn = None
    liger_fused_linear_cross_entropy_fn = None

try:
    from cut_cross_entropy import linear_cross_entropy as cut_cross_entropy_fn
except ImportError:
    cut_cross_entropy_fn = None

# this file has a unified API for comparing many implementations of cross-entropy loss.
# - naive linear + cross-entropy
# - linear + liger cross-entropy
# - liger fused-linear-cross-entropy
# - torch-compiled linear-cross-entropy
# - cut cross-entropy
IGNORE_INDEX = -100


@torch.compile(fullgraph=True, dynamic=True)
def softcapping(logits: torch.Tensor, softcap: float) -> torch.Tensor:
    return torch.tanh(logits / softcap) * softcap


def naive_linear_cross_entropy(
    hidden_states: torch.Tensor,
    classifier_head: torch.Tensor,  # aka unembedding_matrix
    targets: torch.Tensor,
    softcap: float | None = None,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
) -> torch.Tensor:
    logits = hidden_states @ classifier_head.T  # bsz x seqlen x vocab_size
    if softcap is not None:
        logits = softcapping(logits, softcap)
    loss = F.cross_entropy(
        logits.float(), targets, ignore_index=ignore_index, reduction=reduction
    )
    return loss


compiled_linear_cross_entropy = torch.compile(
    naive_linear_cross_entropy, fullgraph=True, dynamic=True
)


def liger_cross_entropy(
    hidden_states: torch.Tensor,
    classifier_head: torch.Tensor,  # aka unembedding_matrix
    targets: torch.Tensor,
    softcap: float | None = None,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
):
    if liger_cross_entropy_fn is None:
        raise ImportError("liger cross-entropy not available")
    logits = hidden_states @ classifier_head.T  # bsz x seqlen x vocab_size
    return liger_cross_entropy_fn(
        logits, targets, ignore_index=ignore_index, reduction=reduction, softcap=softcap
    )


def liger_fused_linear_cross_entropy(
    hidden_states: torch.Tensor,
    classifier_head: torch.Tensor,  # aka unembedding_matrix
    targets: torch.Tensor,
    softcap: float | None = None,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
):
    if liger_fused_linear_cross_entropy_fn is None:
        raise ImportError("liger cross-entropy not available")

    return liger_fused_linear_cross_entropy_fn(
        hidden_states,
        classifier_head,
        targets,
        input,
        bias=None,
        ignore_index=ignore_index,
        lse_square_scale=0.0,
        label_smoothing=0.0,
        reduction=reduction,
        softcap=softcap,
    )


def cut_cross_entropy(
    hidden_states: torch.Tensor,
    classifier_head: torch.Tensor,  # aka unembedding_matrix
    targets: torch.Tensor,
    softcap: float | None = None,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
):
    if cut_cross_entropy_fn is None:
        raise ImportError("cut cross-entropy not available")
    return cut_cross_entropy_fn(
        hidden_states.to(torch.bfloat16),  # e: torch.Tensor,
        classifier_head.to(torch.bfloat16),  # c: torch.Tensor,
        targets,
        ignore_index=ignore_index,
        softcap=softcap,
        reduction=reduction,
    )


def linear_cross_entropy(
    hidden_states: torch.Tensor,
    classifier_head: torch.Tensor,  # aka unembedding_matrix
    targets: torch.Tensor,
    softcap: float | None = None,
    ignore_index: int = IGNORE_INDEX,
    reduction: str = "mean",
    backend: str = "cut_cross_entropy",
):
    if linear_cross_entropy is None:
        raise ImportError("cut cross-entropy not available")
    if backend == "cut_cross_entropy":
        return cut_cross_entropy(
            hidden_states, classifier_head, targets, softcap, ignore_index, reduction
        )
    elif backend == "liger_cross_entropy":
        return liger_cross_entropy(
            hidden_states, classifier_head, targets, softcap, ignore_index, reduction
        )
    elif backend == "liger_fused_linear_cross_entropy":
        return liger_fused_linear_cross_entropy(
            hidden_states, classifier_head, targets, softcap, ignore_index, reduction
        )
    elif backend == "compiled_linear_cross_entropy":
        return compiled_linear_cross_entropy(
            hidden_states, classifier_head, targets, softcap, ignore_index, reduction
        )
    elif backend == "naive_linear_cross_entropy":
        return naive_linear_cross_entropy(
            hidden_states, classifier_head, targets, softcap, ignore_index, reduction
        )
    else:
        raise ValueError(f"Invalid backend {backend}")
