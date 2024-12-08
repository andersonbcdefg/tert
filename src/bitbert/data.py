import os
import torch
import random
import datasets
from torch.utils.data import DataLoader, IterableDataset
from collections import deque
import pyarrow.parquet as pq
from typing import Any, Iterator
from .tokenizer import Tokenizer

# from .glue import download_glue, load_task
from .scheduler import MaskingScheduleCollator
from .util import chunk_text


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
            self.labels[idx],
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
                print(f"reading table {full_path}...")
                table = pq.read_table(full_path, use_threads=True)
                print(f"read table {full_path}!")
                # Convert the table to a list of dictionaries
                records = table.to_pylist()
                print(f"got {len(records)} records from {full_path}!")
                for record in records:
                    yield record
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue


class InterleavedDataset(IterableDataset):
    def __init__(
        self,
        datasets: list[IterableDataset],
        weights: list[float],  # can be lengths, or arbitrary sampling weights
        shuffle: bool = True,  # why not, it's already randomized because of sampling
        buffer_size: int = 1_000,
        split_text_at: int
        | None = None,  # how many words to split the text into chunks of
        infinite: bool = False,
    ):
        if len(datasets) != len(weights):
            raise ValueError(
                "The number of datasets must match the number of lengths provided."
            )
        self.buffer_size = buffer_size if shuffle else 1
        self.datasets = datasets
        total_weight = sum(weights)
        self.probs = [w / total_weight for w in weights]
        self.split_text_at = split_text_at
        self.infinite = infinite
        self.counts = [0] * len(datasets)

    def __iter__(self) -> Iterator[str]:
        """
        Creates an iterator that interleaves the child datasets based on weighted probabilities.
        Optionally shuffles the interleaved stream with an in-memory buffer.
        Yields:
          Any: Records from the child datasets.
        """
        buffer = deque()
        count = 0
        while True:
            iterators = [iter(dataset) for dataset in self.datasets]
            remaining_indices = list(range(len(self.datasets)))
            print("got iterators!")

            def get_next_interleaved_item():
                """Helper function to get the next item from the weighted interleaved streams"""
                while remaining_indices:
                    # print(f"Remaining indices: {remaining_indices}")
                    selected_index = random.choices(
                        remaining_indices,
                        weights=[self.probs[i] for i in remaining_indices],
                        k=1,
                    )[0]
                    # print(f"Selected index: {selected_index}")
                    try:
                        item = next(iterators[selected_index])
                        self.counts[selected_index] += 1
                        return item
                    except StopIteration:
                        # if infinite, just reset the iterator
                        if self.infinite:
                            print(
                                f"Resetting iterator {selected_index} after {self.counts[selected_index]} items..."
                            )
                            iterators[selected_index] = iter(
                                self.datasets[selected_index]
                            )
                            return next(iterators[selected_index])
                        else:
                            remaining_indices.remove(selected_index)
                    except Exception as e:
                        print(
                            f"Error reading from dataset at index {selected_index}: {e}"
                        )
                        remaining_indices.remove(selected_index)
                raise StopIteration

            try:
                while True:
                    item = get_next_interleaved_item()
                    # print("got item!")
                    # Split into chunks if needed
                    text_chunks = (
                        chunk_text(item["text"], target_chunk_size=self.split_text_at)
                        if self.split_text_at is not None
                        else [item]
                    )
                    chunks = [
                        {"text": chunk} for chunk in text_chunks if len(chunk) > 50
                    ]
                    # print("split into", len(chunks), "chunks")
                    for chunk in chunks:
                        if len(buffer) < self.buffer_size:
                            # print("adding to buffer")
                            buffer.append(chunk)
                        else:
                            # print("yielding from buffer")
                            # Choose random position and yield current item there
                            insert_pos = random.randrange(len(buffer))
                            yield buffer[insert_pos]
                            buffer[insert_pos] = chunk
                        count += 1
            except StopIteration:
                # Yield any remaining items in buffer
                while buffer:
                    yield buffer.popleft()
                print(
                    "After exhausting all datasets, the count of each is:", self.counts
                )
                break


def collate_fn(batch, tokenizer: Tokenizer, max_length, mask_prob=0.15):
    tokenized = tokenizer.encode_batch(
        [x["text"] for x in batch], max_length=max_length, collate_strategy="longest"
    )
    tokenized["input_ids"] = torch.tensor(tokenized["input_ids"], dtype=torch.long)  # pyright: ignore
    tokenized["attention_mask"] = torch.tensor(  # pyright: ignore
        tokenized["attention_mask"], dtype=torch.long
    )

    # Randomly mask 15% of tokens
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

    tokenized["labels"] = labels  # pyright: ignore

    return tokenized


def collate_fn_finetuning(batch, tokenizer: Tokenizer | Any, max_length):
    sentence1s, sentence2s, labels = zip(*batch)
    if isinstance(tokenizer, Tokenizer):
        tokenized = tokenizer.encode_batch_pairs(
            sentence1s, sentence2s, max_length=max_length
        )
        tokenized["input_ids"] = torch.tensor(tokenized["input_ids"], dtype=torch.long)
        tokenized["attention_mask"] = torch.tensor(
            tokenized["attention_mask"], dtype=torch.long
        )
        tokenized["labels"] = torch.tensor(labels, dtype=torch.long)
    else:
        # huggingface tokenizer
        tokenized = tokenizer.batch_encode_plus(
            list(zip(sentence1s, sentence2s)),
            max_length=max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        tokenized["labels"] = torch.tensor(labels, dtype=torch.long)
    return tokenized


def download_dclm_data(data_dir: str):
    import huggingface_hub

    local_dir = os.path.join(data_dir, "dclm")
    huggingface_hub.snapshot_download(
        repo_id="TaylorAI/dclm_subset_1pct", repo_type="dataset", local_dir=local_dir
    )


def get_dclm_dataset(data_dir: str):
    # NUM_SAMPLES = 10_000_000
    local_dir = os.path.join(data_dir, "dclm")
    ds = ShardedFileDataset(local_dir)
    return ds  # , NUM_SAMPLES


def download_fineweb_data(data_dir: str):
    import huggingface_hub

    local_dir = os.path.join(data_dir, "fineweb")
    huggingface_hub.snapshot_download(
        repo_id="HuggingFaceFW/fineweb-edu",
        allow_patterns="sample/10BT/*.parquet",
        repo_type="dataset",
        local_dir=local_dir,
    )


def get_fineweb_dataset(data_dir: str):
    # there is no reason to stream this when we can just download once at build time
    # NUM_SAMPLES = 10_000_000
    local_dir = os.path.join(data_dir, "fineweb/sample/10BT")
    ds = ShardedFileDataset(local_dir)
    return ds  # , NUM_SAMPLES


def download_wiki_data(data_dir: str):
    import huggingface_hub

    local_dir = os.path.join(data_dir, "wikipedia_en")
    huggingface_hub.snapshot_download(
        repo_id="wikimedia/wikipedia",
        allow_patterns="20231101.en/*.parquet",
        repo_type="dataset",
        local_dir=local_dir,
    )


def get_wiki_dataset(data_dir: str):
    # 7m articles, unclear how many chunks. probably like 30m?
    local_dir = os.path.join(data_dir, "wikipedia_en/20231101.en")
    ds = ShardedFileDataset(local_dir)
    return ds  # , num_samples


# def get_gutenberg_dataset(data_dir: str):
#   import huggingface_hub

#   local_dir = os.path.join(data_dir, "gutenberg")
#   ds = datasets.load_dataset(
#     "eminorhan/gutenberg_en", "chunk_size_2048", split="train", cache_dir=local_dir
#   )
#   num_samples = len(ds)  # pyright: ignore
#   return ds, num_samples

# def get_tinystories_dataset(data_dir: str):
#   local_dir = os.path.join(data_dir, "tinystories")
#   ds = datasets.load_dataset(
#     "roneneldan/TinyStories", split="train", cache_dir=local_dir
#   )
#   num_samples = len(ds)  # pyright: ignore
#   return ds, num_samples


dataset_name_to_getter = {
    "dclm": get_dclm_dataset,
    # "gutenberg": get_gutenberg_dataset,
    "fineweb": get_fineweb_dataset,
    "wiki": get_wiki_dataset,
    # "tinystories": get_tinystories_dataset,
}


def get_combined_dataset(
    data_dir: str,
    splits: list[str] = ["dclm"],
    weights: list[float] = [1.0],
    chunk_size: int = 400,  # words, approx 512 tokens hopefully
):
    if len(splits) == 0:
        raise ValueError("At least one split must be specified.")
    assert len(splits) == len(weights), "Length of splits and weights must be the same."
    datasets = []
    for split in splits:
        dataset = dataset_name_to_getter[split](data_dir)
        datasets.append(dataset)
    # print("Created interleaved dataset with {} samples".format(sum(lengths)))
    return InterleavedDataset(
        datasets, weights, split_text_at=chunk_size, infinite=True
    )


def get_interleaved_dataloader(
    data_dir: str, batch_size: int = 32, max_length=512, tokenizer=None
):
    if tokenizer is None:
        tokenizer = Tokenizer()
    NUM_SAMPLES = 28_000_000  # 19_672_200
    fw = datasets.load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True
    )
    dclm = ShardedFileDataset(data_dir)
    gutenberg = datasets.load_dataset(
        "eminorhan/gutenberg_en", "chunk_size_2048", split="train", streaming=False
    )
    wiki = datasets.load_dataset("pszemraj/simple_wikipedia", split="train")
    interleaved = InterleavedDataset(
        [fw, dclm, gutenberg, wiki],
        [9_672_200, 10_000_000, 8_000_000, 226_000],  # pyright: ignore
    )
    collator = MaskingScheduleCollator(
        total_steps=NUM_SAMPLES // batch_size + 1,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    dataloader = DataLoader(
        interleaved, batch_size=batch_size, collate_fn=collator, num_workers=1
    )
    return dataloader, NUM_SAMPLES


# def get_glue_dataloaders(
#   glue_metadata: dict, task: str, batch_size: int, max_length=512, tokenizer=None
# ):
#   if tokenizer is None:
#     tokenizer = Tokenizer()
#   # download glue if we haven't already
#   if not os.path.exists("glue") or not os.path.exists("glue/CoLA"):
#     download_glue(glue_metadata=glue_metadata, data_dir="glue")

#   print(f"Getting {task} dataloaders...")
#   train_sentence1s, train_sentence2s, train_labels = load_task(
#     task, glue_metadata, split="train"
#   )
#   train_dataset = FineTuneDataset(
#     train_sentence1s,
#     train_sentence2s,
#     train_labels,
#     glue_metadata["num_classes"][task],
#   )
#   collate_fn_partial = partial(
#     collate_fn_finetuning, tokenizer=tokenizer, max_length=max_length
#   )
#   train_dataloader = DataLoader(
#     train_dataset,
#     batch_size=batch_size,
#     collate_fn=collate_fn_partial,
#     num_workers=1,
#     shuffle=True,
#   )
#   if task != "MNLI":
#     dev_sentence1s, dev_sentence2s, dev_labels = load_task(
#       task, glue_metadata, split="dev"
#     )
#     dev_dataset = FineTuneDataset(
#       dev_sentence1s,
#       dev_sentence2s,
#       dev_labels,
#       glue_metadata["num_classes"][task],
#     )
#     dev_dataloader = DataLoader(
#       dev_dataset,
#       batch_size=batch_size,
#       collate_fn=collate_fn_partial,
#       num_workers=1,
#       shuffle=False,
#     )
#     return train_dataloader, (dev_dataloader,)
#   else:
#     dev_matched_sentence1s, dev_matched_sentence2s, dev_matched_labels = load_task(
#       task, glue_metadata, split="dev_matched"
#     )
#     dev_matched_dataset = FineTuneDataset(
#       dev_matched_sentence1s,
#       dev_matched_sentence2s,
#       dev_matched_labels,
#       glue_metadata["num_classes"][task],
#     )
#     dev_matched_dataloader = DataLoader(
#       dev_matched_dataset,
#       batch_size=batch_size,
#       collate_fn=collate_fn_partial,
#       num_workers=1,
#       shuffle=False,
#     )
#     (
#       dev_mismatched_sentence1s,
#       dev_mismatched_sentence2s,
#       dev_mismatched_labels,
#     ) = load_task(task, glue_metadata, split="dev_mismatched")
#     dev_mismatched_dataset = FineTuneDataset(
#       dev_mismatched_sentence1s,
#       dev_mismatched_sentence2s,
#       dev_mismatched_labels,
#       glue_metadata["num_classes"][task],
#     )
#     dev_mismatched_dataloader = DataLoader(
#       dev_mismatched_dataset,
#       batch_size=batch_size,
#       collate_fn=collate_fn_partial,
#       num_workers=1,
#       shuffle=False,
#     )
#     return train_dataloader, (dev_matched_dataloader, dev_mismatched_dataloader)
