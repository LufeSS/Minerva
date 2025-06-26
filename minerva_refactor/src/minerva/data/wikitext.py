from __future__ import annotations

from typing import Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def _tokenize_function(examples, tokenizer, block_size):
    # examples["text"] is a list of strings
    return tokenizer(examples["text"], return_special_tokens_mask=False)


def _group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated[list(examples.keys())[0]])
    # We drop the remainder, we could add padding if needed.
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def load_wikitext(
    split: str = "train",
    tokenizer_name: str = "gpt2",
    block_size: int = 128,
    dataset_variant: str = "wikitext-2-raw-v1",
) -> Tuple[torch.utils.data.Dataset, int, "transformers.PreTrainedTokenizerBase"]:
    """Load WikiText subset (2 or 103) and return HF dataset in PyTorch format.

    Parameters
    ----------
    split: str
        One of "train", "validation", "test".
    tokenizer_name: str
        Name or path of a HF tokenizer (defaults to GPT-2).
    block_size: int
        Length of each training chunk after grouping.
    dataset_variant: str
        Either "wikitext-2-raw-v1" (default) or "wikitext-103-raw-v1".
    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", dataset_variant, split=split)

    tokenize_fn = lambda x: _tokenize_function(x, tokenizer, block_size)
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    grouped = tokenized.map(
        lambda x: _group_texts(x, block_size),
        batched=True,
    )
    grouped.set_format(type="torch")
    return grouped, tokenizer.vocab_size, tokenizer


def build_dataloader(
    dataset: torch.utils.data.Dataset, *, batch_size: int, shuffle: bool = True
) -> DataLoader:
    """Return a PyTorch ``DataLoader`` with simple stacking collate_fn."""

    def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = torch.stack([item["input_ids"] for item in batch])
        return {"input_ids": input_ids}

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    ) 