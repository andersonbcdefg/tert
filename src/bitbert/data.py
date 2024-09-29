import torch
import datasets
from functools import partial
from torch.utils.data import DataLoader
from .tokenizer import Tokenizer

def send_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def collate_fn(batch, tokenizer: Tokenizer, max_length):
    tokenized = tokenizer.encode_batch(
        [x["text"] for x in batch],
        max_length=max_length,
        collate_strategy="longest"
    )
    tokenized["input_ids"] = torch.tensor(tokenized["input_ids"], dtype=torch.long)
    tokenized["attention_mask"] = torch.tensor(
        tokenized["attention_mask"], dtype=torch.long
    )

    # Randomly mask 15% of tokens
    mask_prob = 0.15
    masked_indices = torch.bernoulli(
        torch.full(tokenized["input_ids"].shape, mask_prob)
    ).bool()

    # do not allow padding tokens to be masked
    # use the attention_mask to zero those positions
    masked_indices = tokenized["attention_mask"].bool() & masked_indices

    # Create labels tensor
    labels = torch.full(tokenized["input_ids"].shape, -100, dtype=torch.long)
    labels[masked_indices] = tokenized["input_ids"][masked_indices]

    # Apply masking to input_ids
    tokenized["input_ids"][masked_indices] = tokenizer.mask_id

    tokenized["labels"] = labels

    return tokenized


def get_fw_dataloader(
    batch_size: int = 32,
    max_length=512,
    tokenizer=None
):
    NUM_SAMPLES = 9_672_200
    ds = datasets.load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True
    )
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_length=max_length),
        num_workers=1
    )
    return dataloader, NUM_SAMPLES


def get_wiki_dataloader(
    batch_size: int = 32,
    max_length=512,
    tokenizer=None
):
    ds = datasets.load_dataset("pszemraj/simple_wikipedia", split="train")
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_length=max_length),
        num_workers=1
    )
    return dataloader, 226_000

def get_tinystories_dataloader(
    batch_size: int = 32,
    max_length=512,
    tokenizer=None
):
    ds = datasets.load_dataset("roneneldan/TinyStories", split="train")
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_length=max_length),
        num_workers=1
    )
    return dataloader, 2_120_000
