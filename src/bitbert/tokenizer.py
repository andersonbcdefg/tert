from tiktoken import Encoding
from tiktoken.load import load_tiktoken_bpe, dump_tiktoken_bpe
import numpy as np
import awkward as ak
from pathlib import Path
from typing import Literal, Union

VOCAB_SIZE = 32768
STARTOFTEXT = "<|startoftext|>"
ENDOFTEXT = "<|endoftext|>"
MASK = "<|mask|>"
SEP = "<|sep|>"
PAD = "<|pad|>"

special_tokens = {
    STARTOFTEXT: 32764,
    ENDOFTEXT: 32765,
    MASK: 32766,
    SEP: 32767,
    # PAD: -1 # not included in the tokenizer; manage it manually
}


def get_tiktoken_encoding():
    mergeable_ranks = load_tiktoken_bpe(
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
        expected_hash="446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
    )
    # keep the first 32764 tokens; leaving 4 for:
    # - startoftext (aka CLS)
    # - endoftext
    # - mask
    # - sep
    mergeable_ranks = {k: v for k, v in mergeable_ranks.items() if v < 32764}

    # This regex could be made more efficient
    pat_str = "|".join(
        [
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
            r"""\p{N}{1,3}""",
            r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
            r"""\s*[\r\n]+""",
            r"""\s+(?!\S)""",
            r"""\s+""",
        ]
    )
    args = {
        "name": "tokenizer",
        "pat_str": pat_str,
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": special_tokens,
        "explicit_n_vocab": VOCAB_SIZE,
    }
    return Encoding(**args)


class Tokenizer:
    def __init__(self):
        super().__init__()
        self.tokenizer = get_tiktoken_encoding()

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    @property
    def bos_id(self) -> int:
        return special_tokens[STARTOFTEXT]

    @property
    def eos_id(self) -> int:
        return special_tokens[ENDOFTEXT]

    @property
    def pad_id(self) -> int:
        return -1

    @property
    def mask_id(self) -> int:
        return special_tokens[MASK]

    def sep_id(self) -> int:
        return special_tokens[SEP]

    def __call__(self, t: Union[str, list[str]], **kwargs) -> np.ndarray:
        if isinstance(t, str):
            t = [t]
        return self.encode_batch(t, **kwargs)

    def _pad(
        self, tokens: ak.Array | list, max_length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(tokens, ak.Array):
            tokens = ak.Array(tokens)
        sequence_lengths = ak.num(tokens, axis=1)
        tokens = ak.pad_none(tokens, target=max_length, axis=-1, clip=True)
        tokens_padded = np.array(ak.fill_none(tokens, self.pad_id))
        # mask is 0 where pad token is present, 1 otherwise
        mask = (tokens_padded != self.pad_id).astype(np.int32)
        return tokens_padded, mask

    def _unpad(self, tokens: ak.Array):
        tokens = ak.from_regular(tokens)
        is_pad = tokens == self.pad_id
        return tokens[~is_pad]

    def encode_batch(
        self,
        batch: list[str],
        max_length: int,
        collate_strategy: Literal["max_length", "longest", "pack", "jagged"],
        use_bos: bool = True,
        use_eos: bool = True,
        eos_after_truncation: bool = False,
    ) -> dict[str, np.ndarray]:
        # encoding is jagged
        tokens = ak.Array(self.tokenizer.encode_ordinary_batch(batch))
        sequence_lengths = ak.num(tokens, axis=1).tolist()
        # manually add bos/eos tokens
        if use_bos:
            tokens = ak.concatenate(
                [ak.full_like(tokens[:, :1], self.bos_id), tokens], axis=1
            )
        if use_eos and not eos_after_truncation:
            tokens = ak.concatenate(
                [tokens, ak.full_like(tokens[:, :1], self.eos_id)], axis=1
            )

        input_ids = None
        attention_mask = None
        sequence_ids = None  # only if packing
        if collate_strategy == "pack":
            # pack into batches of max_length, throw away the leftovers
            sequence_ids = ak.zeros_like(tokens) + range(len(tokens))
            tokens = ak.flatten(tokens, axis=None)
            sequence_ids = ak.flatten(sequence_ids, axis=None)
            n_to_truncate = len(tokens) % max_length
            input_ids = np.asarray(tokens)[:-n_to_truncate].reshape(-1, max_length)
            attention_mask = np.ones_like(tokens)
            sequence_ids = np.asarray(sequence_ids)[:-n_to_truncate].reshape(
                -1, max_length
            )
        elif collate_strategy in ["max_length", "longest"]:
            if collate_strategy == "longest":
                longest = max(len(t) for t in tokens)  # pyright: ignore
                max_length = min(longest, max_length)

            # keep sequences separate; pad or truncate to max_length
            input_ids, attention_mask = self._pad(tokens, max_length)
            if eos_after_truncation:
                # get last idxs by summing mask
                last_idxs = np.sum(attention_mask, axis=-1) - 1
                input_ids[np.arange(len(input_ids)), last_idxs] = self.eos_id

            # replace -1 with eos to avoid indexing errors
            input_ids = np.where(input_ids == -1, self.eos_id, input_ids)

        elif collate_strategy == "jagged":
            input_ids = tokens.tolist()
        else:
            raise ValueError(f"Invalid collate strategy {collate_strategy}")

        return {  # pyright: ignore
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sequence_ids": sequence_ids,
            "sequence_lengths": sequence_lengths,
        }

    def encode_batch_pairs(
        self,
        batch: list[tuple[str, str]],
        max_length: int,
        truncation_strategy: Literal["longest_first", "only_first", "only_second", "do_not_truncate"] = "longest_first",
        use_bos: bool = True,
        use_eos: bool = True,
        eos_after_truncation: bool = True,
    ):
        """
        Encodes a batch of pairs, separated by SEP token.
        """
        sentences1 = [x[0] for x in batch]
        sentences2 = [x[1] for x in batch]

        tokens1 = ak.Array(self.tokenizer.encode_ordinary_batch(sentences1))
        tokens2 = ak.Array(self.tokenizer.encode_ordinary_batch(sentences2))
        sequence_lengths1 = ak.num(tokens1, axis=1).tolist()
        sequence_lengths2 = ak.num(tokens2, axis=1).tolist()

        # if bos, concatenate to front of tokens1
        if use_bos:
            tokens1 = ak.concatenate(
                [ak.full_like(tokens1[:, :1], self.bos_id), tokens1], axis=1
            )

        # concatenate sep to end of tokens1
        tokens1 = ak.concatenate(
            [tokens1, ak.full_like(tokens1[:, :1], self.sep_id)], axis=1
        )

        # concatenate tokens2 to the end of tokens1
        tokens1 = ak.concatenate([tokens1, tokens2], axis=1)

        if use_eos:
            tokens1 = ak.concatenate(
                [tokens1, ak.full_like(tokens1[:, :1], self.eos_id)], axis=1
            )

        # for fine-tuning with pairs, always pad to max length



    def decode_batch(self, batch: Union[np.ndarray, list]) -> list[str]:
        tokens = self._unpad(ak.Array(np.array(batch))).tolist()
        return self.tokenizer.decode_batch(tokens)
