import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import spacy
import config
import pickle


class TranslationDataset(Dataset):
    def __init__(
        self, src_file, tgt_file, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer
    ):
        self.src_sentences = open(src_file, "r", encoding="utf-8").readlines()
        self.tgt_sentences = open(tgt_file, "r", encoding="utf-8").readlines()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx].strip()
        tgt_sentence = self.tgt_sentences[idx].strip()
        src_tensor = torch.tensor(
            [
                self.src_vocab.get(token, self.src_vocab["<unk>"])
                for token in self.src_tokenizer(src_sentence)
            ],
            dtype=torch.long,
        )
        tgt_tensor = torch.tensor(
            [
                self.tgt_vocab.get(token, self.tgt_vocab["<unk>"])
                for token in self.tgt_tokenizer(tgt_sentence)
            ],
            dtype=torch.long,
        )
        return src_tensor, tgt_tensor


def build_vocab(language, data_path, tokenizer):
    specials = ["<unk>", "<pad>", "<sos>", "<eos>"]
    save_path = os.path.join(config.dataset_dir, f"vocab.{language}")
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            vocab = pickle.load(f)
    else:
        counter = Counter()
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                counter.update(tokenizer(line.strip()))
        vocab = {token: idx for idx, token in enumerate(specials)}
        vocab.update(
            {
                token: idx + len(specials)
                for idx, (token, _) in enumerate(counter.items())
            }
        )
        with open(save_path, "wb") as f:
            pickle.dump(vocab, f)
    return vocab


def get_tokenizer(language):
    spacy_langs = {
        "en": "en_core_web_sm",
        "de": "de_core_news_sm",
    }
    # Please download `en_core_web_sm` and `de_core_news_sm` first
    # for example, to download `en_core_web_sm`, using `pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-x.y.z/en_core_web_sm-x.y.z.tar.gz --no-deps`
    # for the URL in the command, find a compatible release here: https://github.com/explosion/spacy-models/releases/

    spacy_tokenizer = spacy.load(spacy_langs[language])
    return lambda text: [tok.text for tok in spacy_tokenizer(text)]


def load_data(src_lang="en", tgt_lang="de"):
    if sorted((src_lang, tgt_lang)) != ["de", "en"]:
        raise ValueError("Available language options are ('de','en') and ('en', 'de')")

    splits = ["train", "valid", "test"]
    prefix = {"train": "train", "valid": "val", "test": "test"}
    data_files = {
        split: {
            "src": os.path.join(
                config.dataset_dir, split, f"{prefix[split]}.{src_lang}"
            ),
            "tgt": os.path.join(
                config.dataset_dir, split, f"{prefix[split]}.{tgt_lang}"
            ),
        }
        for split in splits
    }

    source_tokenizer = get_tokenizer(src_lang)
    target_tokenizer = get_tokenizer(tgt_lang)

    src_vocab = build_vocab(src_lang, data_files["train"]["src"], source_tokenizer)
    tgt_vocab = build_vocab(tgt_lang, data_files["train"]["tgt"], target_tokenizer)

    train_dataset = TranslationDataset(
        data_files["train"]["src"],
        data_files["train"]["tgt"],
        src_vocab,
        tgt_vocab,
        source_tokenizer,
        target_tokenizer,
    )
    valid_dataset = TranslationDataset(
        data_files["valid"]["src"],
        data_files["valid"]["tgt"],
        src_vocab,
        tgt_vocab,
        source_tokenizer,
        target_tokenizer,
    )
    test_dataset = TranslationDataset(
        data_files["test"]["src"],
        data_files["test"]["tgt"],
        src_vocab,
        tgt_vocab,
        source_tokenizer,
        target_tokenizer,
    )

    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_item, tgt_item in batch:
            src_batch.append(
                torch.cat(
                    [
                        torch.tensor([src_vocab["<sos>"]]),
                        src_item,
                        torch.tensor([src_vocab["<eos>"]]),
                    ],
                    dim=0,
                )
            )
            tgt_batch.append(
                torch.cat(
                    [
                        torch.tensor([tgt_vocab["<sos>"]]),
                        tgt_item,
                        torch.tensor([tgt_vocab["<eos>"]]),
                    ],
                    dim=0,
                )
            )
        src_batch = pad_sequence(src_batch, padding_value=src_vocab["<pad>"])
        tgt_batch = pad_sequence(tgt_batch, padding_value=tgt_vocab["<pad>"])
        return src_batch, tgt_batch

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=config.batch_size, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batch_size, collate_fn=collate_fn
    )

    return (
        src_vocab,
        tgt_vocab,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
    )
