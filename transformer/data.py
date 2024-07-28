import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import config
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import datasets
from typing import Sequence, Optional

SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"

special_tokens = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN]


def build_tokenizer(dataset, lang, force_reload):
    tokenizer_path = os.path.join(config.dataset_dir, f"tokenizer-{lang}.json")
    if os.path.exists(tokenizer_path) and not force_reload:
        tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN)).from_file(tokenizer_path)
    else:
        tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            special_tokens=special_tokens, show_progress=True, min_frequency=2
        )
        print(f"Training tokenizer for {lang}...")

        def batch_iterator():
            for i in range(0, dataset["train"].num_rows, config.batch_size):
                yield dataset["train"][i : i + config.batch_size][lang]

        tokenizer.train_from_iterator(batch_iterator(), trainer)
        tokenizer.enable_padding(
            pad_id=tokenizer.token_to_id(PAD_TOKEN), pad_token=PAD_TOKEN
        )
        tokenizer.enable_truncation(max_length=config.max_len)
        tokenizer.post_processor = TemplateProcessing(
            single=f"{SOS_TOKEN} $A {EOS_TOKEN}",
            pair=f"{SOS_TOKEN} $A {EOS_TOKEN} $B:1 {EOS_TOKEN}:1",  # not used
            special_tokens=[
                (SOS_TOKEN, tokenizer.token_to_id(SOS_TOKEN)),
                (EOS_TOKEN, tokenizer.token_to_id(EOS_TOKEN)),
            ],
        )
        tokenizer.save(tokenizer_path)
    return tokenizer


def load_dataset(splits):
    # data format in parquet, e.g. en -> de
    #                               "translation"
    # 1:     {"en" : <an English sentence>, "de" : <a German sentence>}
    # 2:     {"en" : <an English sentence>, "de" : <a German sentence>}
    # 3:                                ...
    parquet_filenames = {
        split: [
            os.path.join(config.dataset_dir, split, f)
            for f in os.listdir(os.path.join(config.dataset_dir, split))
            if f.endswith(".parquet")
        ]
        for split in splits
    }

    #          "translation.en"           "translation.de"
    # 1:     <an English sentence>      <a German sentence>
    # 2:     <an English sentence>      <a German sentence>
    # 3:              ...                      ...
    dataset = datasets.load_dataset("parquet", data_files=parquet_filenames).flatten()

    # rename columns
    #                 "en"                     "de"
    # 1:     <an English sentence>      <a German sentence>
    # 2:     <an English sentence>      <a German sentence>
    # 3:              ...                      ...
    for split in splits:
        columns = dataset[split].column_names
        lang_suffix = [col.split(".")[-1] for col in columns]
        dataset[split] = dataset[split].rename_columns(
            {col: suffix for col, suffix in zip(columns, lang_suffix)}
        )
    return dataset.with_format("torch")


def load_data(
    src_lang, tgt_lang, splits: Optional[Sequence[str]] = None, force_reload=False
):
    """
    Load IWSLT 2017 dataset. The raw data should be downloaded at `config.dataset_dir` by running download.py first.
    Args:
        src_lang (str): Source language, which depends on which language pair you download.
        tgt_lang (str): Target language, which depends on which language pair you download.
        splits (`Sequence[str]` or `None`, defaults to `None`): The splits you want to load. It can be arbitrary combination of "train", "test" and "valid".
        force_reload (`bool`, defaults to `False`): If set to `True`, it will re-train a new tokenizer with BPE.
    """
    if sorted((src_lang, tgt_lang)) != ["de", "en"]:
        raise ValueError("Available language options are ('de','en') and ('en', 'de')")
    all_splits = ["train", "valid", "test"]
    if not set(splits).issubset(all_splits):
        raise ValueError(f"Splits should only contain some of {all_splits}")

    dataset = load_dataset(splits)

    src_tokenizer = build_tokenizer(dataset, src_lang, force_reload)
    tgt_tokenizer = build_tokenizer(dataset, tgt_lang, force_reload)

    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for item in batch:
            src_batch.append(item[src_lang])
            tgt_batch.append(item[tgt_lang])

        src_batch = src_tokenizer.encode_batch(src_batch)
        tgt_batch = tgt_tokenizer.encode_batch(tgt_batch)

        src_tensor = torch.LongTensor([item.ids for item in src_batch])
        tgt_tensor = torch.LongTensor([item.ids for item in tgt_batch])

        if src_tensor.shape[-1] < tgt_tensor.shape[-1]:
            src_tensor = F.pad(
                src_tensor,
                [0, tgt_tensor.shape[-1] - src_tensor.shape[-1]],
                value=src_tokenizer.token_to_id(PAD_TOKEN),
            )
        else:
            tgt_tensor = F.pad(
                tgt_tensor,
                [0, src_tensor.shape[-1] - tgt_tensor.shape[-1]],
                value=tgt_tokenizer.token_to_id(PAD_TOKEN),
            )

        return src_tensor, tgt_tensor

    dataloaders = [
        DataLoader(
            dataset[split],
            batch_size=config.batch_size,
            collate_fn=collate_fn,
            shuffle=split == "train",
        )
        for split in splits
    ]

    return (src_tokenizer, tgt_tokenizer, *dataloaders)
