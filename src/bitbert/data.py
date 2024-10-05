import os
import torch
import random
import datasets
from functools import partial
from torch.utils.data import DataLoader, IterableDataset
from collections import deque
import random
import pyarrow.parquet as pq
from typing import Any, Iterator
from .tokenizer import Tokenizer
from .glue import download_glue, load_task

def send_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch

class FineTuneDataset(torch.utils.data.Dataset):
    def __init__(self, sentence1s, sentence2s, labels, num_classes):
        self.sentence1s = sentence1s
        self.sentence2s = sentence2s
        self.labels = labels
        self.num_classes = num_classes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.sentence1s[idx],
            self.sentence2s[idx] if self.sentence2s is not None else None,
            self.labels[idx]
        )

class ShardedFileDataset(IterableDataset):
    def __init__(self, path):
        self.dir = path
        self.files = os.listdir(path)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """
        Iterates over the dataset, yielding one record at a time.

        Yields:
            Dict[str, Any]: A dictionary representing a single record from the dataset.
        """
        for file_path in self.files:
            if "parquet" not in file_path:
                continue
            try:
                full_path = os.path.join(self.dir, file_path)
                # Read the Parquet file using PyArrow with Zstandard compression
                table = pq.read_table(full_path, use_threads=True)
                # Convert the table to a list of dictionaries
                records = table.to_pylist()
                for record in records:
                    yield record
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

class InterleavedDataset(IterableDataset):
    def __init__(self, datasets: list[IterableDataset], lengths: list[int]):
        if len(datasets) != len(lengths):
            raise ValueError("The number of datasets must match the number of lengths provided.")

        self.datasets = datasets
        self.original_lengths = lengths.copy()
        self.total_length = sum(lengths)
        self.probs = [length / self.total_length for length in lengths]
        self.finished = [False] * len(self.datasets)

    def __iter__(self) -> Iterator[Any]:
        """
        Creates an iterator that interleaves the child datasets based on weighted probabilities.

        Yields:
            Any: Records from the child datasets.
        """
        iterators = [iter(dataset) for dataset in self.datasets]
        remaining_indices = list(range(len(self.datasets)))
        current_probs = self.probs.copy()

        while remaining_indices:
            # Select a dataset index based on current probabilities
            selected_index = random.choices(remaining_indices, weights=[current_probs[i] for i in remaining_indices], k=1)[0]

            try:
                # Attempt to get the next item from the selected dataset
                item = next(iterators[selected_index])
                yield item
            except StopIteration:
                # If the selected dataset is exhausted, mark it as finished
                self.finished[selected_index] = True
                remaining_indices.remove(selected_index)
                # Optionally, log or print the exhaustion
                # print(f"Dataset at index {selected_index} is exhausted.")
            except Exception as e:
                # Handle other potential exceptions
                print(f"Error reading from dataset at index {selected_index}: {e}")
                self.finished[selected_index] = True
                remaining_indices.remove(selected_index)


class StreamingShuffleDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, buffer_size: int):
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        buffer = deque(maxlen=self.buffer_size)
        iterator = iter(self.dataset)
        count = 0

        def get_item():
            nonlocal count
            try:
                item = next(iterator)
                if len(buffer) < self.buffer_size:
                    buffer.append(item)
                else:
                    j = random.randint(0, count)
                    if j < self.buffer_size:
                        buffer[j] = item
                count += 1
                return random.choice(buffer)
            except StopIteration:
                if buffer:
                    return buffer.popleft()
                raise StopIteration

        while True:
            try:
                yield get_item()
            except StopIteration:
                break

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

def collate_fn_finetuning(
    batch, tokenizer: Tokenizer, max_length
):
    sentence1s, sentence2s, labels = zip(*batch)
    tokenized = tokenizer.encode_batch_pairs(
        sentence1s,
        sentence2s,
        max_length=max_length
    )
    tokenized['input_ids'] = torch.tensor(tokenized['input_ids'], dtype=torch.long)
    tokenized['attention_mask'] = torch.tensor(tokenized['attention_mask'], dtype=torch.long)
    tokenized['labels'] = torch.tensor(labels, dtype=torch.long)
    return tokenized

def get_dclm_dataloader(
    data_dir: str,
    batch_size: int = 32,
    max_length=512,
    tokenizer=None
):
    if tokenizer is None:
        tokenizer = Tokenizer()
    NUM_SAMPLES = 10_000_000
    ds = ShardedFileDataset(data_dir)
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_length=max_length),
        num_workers=1
    )
    return dataloader, NUM_SAMPLES

def get_fw_dataloader(
    batch_size: int = 32,
    max_length=512,
    tokenizer=None
):
    if tokenizer is None:
        tokenizer = Tokenizer()
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

def get_dclm_fw_interleaved_dataloader(
    data_dir: str,
    batch_size: int = 32,
    max_length=512,
    tokenizer=None
):
    if tokenizer is None:
        tokenizer = Tokenizer()
    NUM_SAMPLES = 19_672_200
    fw = datasets.load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True
    )
    dclm = ShardedFileDataset(data_dir)
    interleaved = InterleavedDataset([fw, dclm], [9_672_200, 10_000_000])
    dataloader = DataLoader(
        interleaved,
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
    if tokenizer is None:
        tokenizer = Tokenizer()
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
    if tokenizer is None:
        tokenizer = Tokenizer()
    dataloader = DataLoader(
        ds, # pyright: ignore
        batch_size=batch_size,
        collate_fn=partial(collate_fn, tokenizer=tokenizer, max_length=max_length),
        num_workers=1
    )
    return dataloader, 2_120_000


def get_glue_dataloaders(
    glue_metadata: dict,
    task: str,
    batch_size: int,
    max_length=512,
    tokenizer=None
):
    if tokenizer is None:
        tokenizer = Tokenizer()
    # download glue if we haven't already
    if not os.path.exists("glue") or not os.path.exists("glue/CoLA"):
        download_glue(glue_metadata=glue_metadata, data_dir="glue")

    print(f"Getting {task} dataloaders...")
    train_sentence1s, train_sentence2s, train_labels = load_task(task, glue_metadata, split="train")
    train_dataset = FineTuneDataset(
        train_sentence1s, train_sentence2s, train_labels, glue_metadata['num_classes'][task]
    )
    collate_fn_partial = partial(collate_fn_finetuning, tokenizer=tokenizer, max_length=max_length)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_partial,
        num_workers=1,
        shuffle=True
    )
    if task != "MNLI":
        dev_sentence1s, dev_sentence2s, dev_labels = load_task(task, glue_metadata, split="dev")
        dev_dataset = FineTuneDataset(
            dev_sentence1s, dev_sentence2s, dev_labels, glue_metadata['num_classes'][task]
        )
        dev_dataloader = DataLoader(
            dev_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn_partial,
            num_workers=1,
            shuffle=False
        )
        return train_dataloader, (dev_dataloader,)
    else:
        dev_matched_sentence1s, dev_matched_sentence2s, dev_matched_labels = load_task(task, glue_metadata, split="dev_matched")
        dev_matched_dataset = FineTuneDataset(
            dev_matched_sentence1s, dev_matched_sentence2s, dev_matched_labels, glue_metadata['num_classes'][task]
        )
        dev_matched_dataloader = DataLoader(
            dev_matched_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn_partial,
            num_workers=1,
            shuffle=False
        )
        dev_mismatched_sentence1s, dev_mismatched_sentence2s, dev_mismatched_labels = load_task(
            task, glue_metadata, split="dev_mismatched"
        )
        dev_mismatched_dataset = FineTuneDataset(
            dev_mismatched_sentence1s, dev_mismatched_sentence2s, dev_mismatched_labels, glue_metadata['num_classes'][task]
        )
        dev_mismatched_dataloader = DataLoader(
            dev_mismatched_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn_partial,
            num_workers=1,
            shuffle=False
        )
        return train_dataloader, (dev_matched_dataloader, dev_mismatched_dataloader)
