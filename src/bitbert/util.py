import re
from itertools import accumulate
from typing import Callable, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

token_pattern_sklearn = r"(?u)\b\w\w+\b"
token_pattern_math = r"\b\w+\b|[^\w\s]"
token_pattern_math_2 = r"(?:\b\w+[-:.\w]*\b)|(?:[-—–])|(?:[^\w\s]+)"


def split_long_sentence(sentence, max_length=1000):
    if len(sentence) <= max_length:
        return [sentence]

    num_chunks = len(sentence) // 500 + 1
    words = sentence.split()

    # Distribute words into chunks
    chunks = []
    for i in range(num_chunks):
        chunk_size = len(words) // num_chunks + (i < len(words) % num_chunks)
        chunk = " ".join(words[:chunk_size])
        words = words[chunk_size:]
        chunks.append(chunk)

    return chunks


def combine_short_sentences(sentences, min_length=25):
    combined_sentences = []
    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        # Combine with next sentence if it's too short and not the last one
        while i + 1 < len(sentences) and len(sentence) < min_length:
            i += 1
            sentence += " " + sentences[i]
        combined_sentences.append(sentence)
        i += 1
    return combined_sentences


def split_sentences(text, max_length=1000, min_length=25):
    # First split into sentences
    # sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    # new regex also avoids splitting on initials
    sentences = re.split(
        r"(?<!\b[A-Za-z]\.)(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text
    )

    # Further split long sentences
    all_sentences = []
    for sentence in sentences:
        all_sentences.extend(split_long_sentence(sentence, max_length=max_length))

    # Combine short sentences
    all_sentences = combine_short_sentences(all_sentences, min_length=min_length)

    return all_sentences


def get_text_length(
    text: str,
    count_by: Literal["words", "chars", "bytes", "tokens"] = "words",
    tokenizer: Callable | None = None,
) -> int:
    if count_by == "words":
        return len(text.split())
    elif count_by == "chars":
        return len(text)
    elif count_by == "bytes":
        return len(text.encode())
    elif count_by == "tokens":
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided for token count")
        return len(tokenizer(text))
    else:
        raise ValueError(f"Invalid count_by: {count_by}")


def chunk_text(
    text: str,
    target_chunk_size: int = 450,
    count_by: Literal["words", "chars", "bytes"] = "words",
    overlap: int = 0,  # sentences to overlap; doesn't count towards chunk size
) -> list[str]:
    length = get_text_length(text, count_by=count_by)
    if length <= target_chunk_size:
        return [text]
    num_chunks = length // target_chunk_size + 1
    target_length = len(text) // num_chunks
    sentences = split_sentences(text)
    sentence_lengths = [
        get_text_length(sentence, count_by=count_by) for sentence in sentences
    ]
    cumsums = list(accumulate(sentence_lengths))
    split_points = []
    current_chunk = 0
    for i, cumsum in enumerate(cumsums):
        if cumsum - (current_chunk * target_length) >= target_length:
            split_points.append(i)
            current_chunk += 1

    # Split the sentences into chunks
    chunks = []
    start = 0
    for end in split_points:
        chunks.append(" ".join(sentences[start : end + overlap]))
        start = end
    chunks.append(" ".join(sentences[start:]))  # Add the last chunk

    return [chunk for chunk in chunks if chunk]


def dict_slice(d, start, end):
    return {k: v[start:end] for k, v in d.items()}


def send_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def scale_grads(model: nn.Module, scale_factor: float):
    # divide all grads by scale_factor (i.e. total tokens)
    scale_factor = 1.0 / scale_factor
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data.mul_(scale_factor)

# def pad_or_truncate(
#     tensor: torch.Tensor,
#     value: int,
#     dim: int,
#     max_length: int,
# ) -> torch.Tensor:
#     """
#     Pads or truncates a tensor to a specified maximum length along a given dimension.

#     Args:
#         tensor (torch.Tensor): Input tensor to be padded or truncated
#         value (int): Padding value to use when extending the tensor
#         dim (int): Dimension along which to pad or truncate
#         max_length (int): Target length for the specified dimension

#     Returns:
#         torch.Tensor: Padded or truncated tensor with the specified length along the given dimension
#     """
#     current_length = tensor.size(dim)

#     if current_length == max_length:
#         return tensor

#     if current_length > max_length:
#         # Truncate the tensor
#         if dim == 0:
#             return tensor[:max_length]
#         elif dim == 1:
#             return tensor[:, :max_length]
#         elif dim == 2:
#             return tensor[:, :, :max_length]
#         else:
#             raise ValueError(f"Invalid dimension: {dim}. must be 0, 1, or 2")

#     else:
#         # Pad the tensor
#         pad_size = [0] * (2 * tensor.ndim)
#         pad_size[2 * dim + 1] = max_length - current_length
#         return torch.nn.functional.pad(tensor, pad_size, mode='constant', value=value)
def pad_or_truncate(
    tensor: torch.Tensor,
    value: int,
    dim: int,
    max_length: int,
) -> torch.Tensor:
    current_length = tensor.size(dim)
    if current_length == max_length:
        return tensor
    if current_length > max_length:
        # Truncate the tensor
        if dim == 0:
            return tensor[:max_length]
        elif dim == 1:
            return tensor[:, :max_length]
        elif dim == 2:
            return tensor[:, :, :max_length]
        else:
            raise ValueError(f"Invalid dimension: {dim}. must be 0, 1, or 2")
    else:
        # Pad the tensor
        pad_size = [0] * (2 * tensor.ndim)
        # Fix: Pad from the end, so for dim=1 we want [..., 0, pad_amount]
        pad_size[2 * (tensor.ndim - 1 - dim) + 1] = max_length - current_length
        return torch.nn.functional.pad(tensor, pad_size, mode='constant', value=value)
