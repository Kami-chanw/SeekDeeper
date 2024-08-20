import os
from math import ceil
from typing import Optional, Callable

import datasets
from datasets import DownloadManager
import torch
from torch.utils.data import Dataset, DataLoader

import config
from modules.bpe import BPETokenizer


class TokenIDDataset(Dataset):

    def __init__(
        self, data_source, num_token_per_item: int, extract_token_id_fn: Callable
    ):
        """
        This class is used to read continuous max_len tokens from data_source.

        Args:
            data_source:
                A dataset where each item contains token ids of a sentence.
            num_token_per_item (int):
                The number of token ids of an item.
            extract_token_id_fn (Callable):
                A function that takes an item in data_source and returns token ids.
        """
        super().__init__()
        self.data_source = data_source
        self.num_token_per_item = num_token_per_item
        self.extract_token_id_fn = extract_token_id_fn

    def __getitem__(self, index):
        rest_length = self.num_token_per_item
        sequence = []
        while rest_length > 0:
            token_ids = list(self.extract_token_id_fn(self.data_source[index]))
            sequence.extend(token_ids[:rest_length])
            rest_length -= len(token_ids)
            index += 1
        return torch.tensor(sequence, dtype=torch.long)

    def __len__(self):
        accumulate_length = 0
        for i, example in enumerate(reversed(self.data_source)):
            accumulate_length += len(self.extract_token_id_fn(example))
            if accumulate_length >= self.num_token_per_item:
                return len(self.data_source) - i
        return 0


def _load_bookcorpus(tokenizer, loading_ratio, num_proc):
    def tokenize(example):
        example["text"] = tokenizer.encode(example["text"], verbose=False)
        return example

    num_parquet_files = ceil(loading_ratio * 10)
    # 10 files in total, but we may just use part of them
    URLS = [
        f"https://hf-mirror.com/datasets/bookcorpus/bookcorpus/resolve/refs%2Fconvert%2Fparquet/plain_text/train/000{i}.parquet?download=true"
        for i in range(num_parquet_files)
    ]

    dl_manager = DownloadManager("bookcorpus")
    paths = dl_manager.download(URLS)
    print("Downloaded at ", paths)

    dataset = datasets.load_dataset(
        "parquet", data_files=paths, split="train", num_proc=num_proc
    )

    num_per_file = len(dataset) // num_parquet_files

    subset = dataset.select(range(int(loading_ratio * 10 * num_per_file))).map(
        tokenize, load_from_cache_file=True, num_proc=num_proc, batched=True
    )

    return DataLoader(
        TokenIDDataset(
            subset,
            num_token_per_item=config.max_len,
            extract_token_id_fn=lambda example: example["text"],
        ),
        batch_size=config.TrainConfig.batch_size,
        shuffle=True,
    )


def _load_cola(tokenizer, loading_ratio, num_proc):
    pass


def load_data(name: str, loading_ratio: float = 1, num_proc: Optional[int] = None):
    dispatch = {
        "bookcorpus": _load_bookcorpus,
        "cola": _load_cola,
    }
    assert name.lower() in dispatch
    assert 0 < loading_ratio <= 1

    tokenizer = BPETokenizer(
        os.path.join(config.bookcorpus_dir, "encoder_bpe_40000.json"),
        os.path.join(config.bookcorpus_dir, "vocab_40000.bpe"),
    )

    return dispatch[name.lower()](tokenizer, loading_ratio, num_proc)