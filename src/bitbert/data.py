import torch
import datasets
from functools import partial
from torch.utils.data import DataLoader


def send_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def collate_fn(batch, tokenizer, device="cuda"):
    tokenized = tokenizer(
        [x["text"] for x in batch], max_length=512, collate_strategy="longest"
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

    return send_to_device(tokenized, device)


def get_dataloader(batch_size: int = 32, tokenizer=None):
    ds = datasets.load_dataset("pszemraj/simple_wikipedia", split="train")
    dataloader = DataLoader(
        ds, batch_size=batch_size, collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )
    return dataloader
