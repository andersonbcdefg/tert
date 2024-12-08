from tiktoken import Encoding
from tiktoken.load import load_tiktoken_bpe
import numpy as np
import awkward as ak
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


def calc_truncation(len1: int, len2: int, available_length: int):
    if available_length < 0:
        raise ValueError("available_length must be non-negative.")

    combined_length = len1 + len2
    overflow = combined_length - available_length
    length_diff = abs(len1 - len2)

    if overflow <= 0:
        return len1, len2
    trunc_longer = min(length_diff, overflow)
    remaining_overflow = overflow - trunc_longer
    trunc_both = (
        remaining_overflow // 2
        if remaining_overflow % 2 == 0
        else (remaining_overflow // 2) + 1
    )

    if len1 > len2:
        num_to_truncate1 = trunc_longer + trunc_both
        num_to_truncate2 = trunc_both
    else:
        num_to_truncate1 = trunc_both
        num_to_truncate2 = trunc_longer + trunc_both

    return len1 - num_to_truncate1, len2 - num_to_truncate2


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

    @property
    def sep_id(self) -> int:
        return special_tokens[SEP]

    def __call__(self, t: Union[str, list[str]], **kwargs):
        if isinstance(t, str):
            t = [t]
        return self.encode_batch(t, **kwargs)

    def _pad(
        self, tokens: ak.Array | list, max_length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(tokens, ak.Array):
            tokens = ak.Array(tokens)
        tokens = ak.pad_none(tokens, target=max_length, axis=-1, clip=True)
        tokens_padded = np.array(ak.fill_none(tokens, self.pad_id))
        # mask is 0 where pad token is present, 1 otherwise
        mask = (tokens_padded != self.pad_id).astype(np.int32)
        return tokens_padded, mask

    def _unpad(self, tokens: ak.Array) -> ak.Array:
        tokens = ak.from_regular(tokens)
        is_pad = tokens == self.pad_id
        return tokens[~is_pad]  # pyright: ignore

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
        sentence1s: list[str],
        sentence2s: list[str] | list[None],
        max_length: int,
        truncation_strategy: Literal[
            "longest_first"
        ] = "longest_first",  # do other ones later if we actually need them
        use_bos: bool = True,
        use_eos: bool = True,
        eos_after_truncation: bool = True,
        collate_strategy: Literal["max_length", "longest", "jagged"] = "jagged",
    ):
        """
        Encodes a batch of pairs, separated by SEP token.
        """
        assert len(sentence1s) == len(
            sentence2s
        ), "sentence1s and sentence2s must be the same length"
        tokens1 = ak.Array(self.tokenizer.encode_ordinary_batch(sentence1s))
        tokens2 = ak.Array(self.tokenizer.encode_ordinary_batch(sentence2s))  # pyright: ignore
        # print(tokens1.type.show())
        sequence_lengths1 = ak.num(tokens1, axis=1).tolist()
        sequence_lengths2 = ak.num(tokens2, axis=1).tolist()
        # print(sequence_lengths1, sequence_lengths2)
        available_length = max_length - 3
        new_lens = [
            calc_truncation(len1, len2, available_length)
            for len1, len2 in zip(sequence_lengths1, sequence_lengths2)
        ]
        # print(new_lens1, new_lens2)
        tokens1 = tokens1.tolist()
        tokens2 = tokens2.tolist()
        # print(tokens1, tokens2)
        for i, (len1, len2) in enumerate(new_lens):
            tokens1[i] = tokens1[i][:len1]
            tokens2[i] = tokens2[i][:len2]
        tokens1 = ak.Array(tokens1)
        # print(tokens1.type.show())
        tokens2 = ak.Array(tokens2)
        # print("tokens", tokens1, tokens2)
        # combine with sep token between
        # print("tokens1 shape:", tokens1.type.show())
        # print("tokens2 shape:", tokens2.type.show())
        sep = ak.full_like(tokens1[:, :1], self.sep_id)
        # print("sep shape:", sep.type.show())
        tokens = ak.concatenate([tokens1, sep, tokens2], axis=1)

        # print("concatenated with sep type:")
        # tokens.type.show()

        if use_bos:
            tokens = ak.concatenate(
                [ak.full_like(tokens1[:, :1], self.bos_id), tokens], axis=1
            )
            # print("concatenated with bos type:")
            # tokens.type.show()
        if use_eos:
            tokens = ak.concatenate(
                [tokens, ak.full_like(tokens1[:, :1], self.eos_id)], axis=1
            )
            # print("concatenated with bos type:")
            # tokens.type.show()

        # for fine-tuning with pairs, always pad to longest sequence length
        # print("tokens", tokens.tolist())

        # print("pad length:", pad_length)
        if collate_strategy in ["max_length", "longest"]:
            pad_length = (
                max(ak.num(tokens, axis=1).tolist())
                if collate_strategy == "longest"
                else max_length
            )
            input_ids, attention_mask = self._pad(tokens, pad_length)
            input_ids = np.where(input_ids == -1, self.eos_id, input_ids)
        else:
            input_ids = tokens
            attention_mask = ak.ones_like(tokens)

        return {
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
        }

    def decode_batch(self, batch: Union[np.ndarray, list]) -> list[str]:
        tokens = self._unpad(ak.Array(np.array(batch))).tolist()
        return self.tokenizer.decode_batch(tokens)


FAKE_SENTENCE_PAIRS = [
    ("This is a sentence.", "This is another sentence."),
    ("This is a sentence.", "This is another sentence."),
    ("And this is a sentence.", "And this is another sentence."),
    ("And this is a sentence.", "And this is another sentence."),
    ("A sentence.", "Another sentence."),
    ("A sentence.", "Another sentence."),
]
