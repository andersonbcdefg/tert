import re
from itertools import accumulate
from collections import Counter
from typing import Callable, Literal
import torch

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
