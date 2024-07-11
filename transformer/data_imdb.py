import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from config import max_len

data_dir = "./datasets/aclImdb"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")
vocab_path = os.path.join(data_dir, "imdb.vocab")

CLS_TOKEN = "<cls>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"


class Tokenizer:
    def __init__(self, vocab_path, max_len):
        self.vocab = self.load_vocab(vocab_path)
        self.inv_vocab = {idx: word for word, idx in self.vocab.items()}
        self.max_len = max_len

    def load_vocab(self, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = f.read().splitlines()
        vocab = [CLS_TOKEN, UNK_TOKEN, PAD_TOKEN] + vocab
        return {word: idx for idx, word in enumerate(vocab)}

    def clean_text(self, text):
        return text.lower()

    def encode(self, text):
        tokens = self.clean_text(text).split()
        token_ids = [self.vocab[CLS_TOKEN]] + [
            self.vocab.get(word, self.vocab[UNK_TOKEN]) for word in tokens
        ]
        if len(token_ids) > self.max_len:
            token_ids = token_ids[: self.max_len]
        return token_ids

    def decode(self, token_ids):
        tokens = [self.inv_vocab.get(id, UNK_TOKEN) for id in token_ids]
        return " ".join(tokens)


tokenizer = Tokenizer(vocab_path, max_len)
vocab = tokenizer.vocab


def read_file(file_path, tokenizer, label_type):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        token_ids = tokenizer.encode(content)
        label = 1 if label_type == "pos" else 0
    return token_ids, label


class IMDBDataset(Dataset):
    def __init__(self, data_directory, tokenizer, label_type):
        self.data = []
        self.labels = []
        dir_name = os.path.join(data_directory, label_type)
        file_list = [
            os.path.join(dir_name, fname)
            for fname in os.listdir(dir_name)
            if fname.endswith(".txt")
        ]

        with ThreadPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(
                        lambda f: read_file(f, tokenizer, label_type), file_list
                    ),
                    total=len(file_list),
                    desc=f"Loading {label_type} data",
                )
            )

        for token_ids, label in results:
            self.data.append(token_ids)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(
            self.labels[idx], dtype=torch.long
        )


train_pos_dataset = IMDBDataset(train_dir, tokenizer, "pos")
train_neg_dataset = IMDBDataset(train_dir, tokenizer, "neg")
test_pos_dataset = IMDBDataset(test_dir, tokenizer, "pos")
test_neg_dataset = IMDBDataset(test_dir, tokenizer, "neg")

train_dataset = ConcatDataset([train_pos_dataset, train_neg_dataset])
test_dataset = ConcatDataset([test_pos_dataset, test_neg_dataset])


def collate_batch(batch):
    data, labels = zip(*batch)
    data = torch.nn.utils.rnn.pad_sequence(
        data, batch_first=True, padding_value=vocab[PAD_TOKEN]
    )
    labels = torch.stack(labels)
    return data, labels


train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch
)
