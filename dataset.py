import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from vocab import tokenize_input, tokenize_target, END_IDX


class SeqDataset(Dataset):
    def __init__(self, inputs: list[str], targets: list[str]):
        self.X = [torch.tensor(tokenize_input(s),  dtype=torch.long) for s in inputs]
        self.Y = [torch.tensor(tokenize_target(s), dtype=torch.long) for s in targets]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    X_batch, Y_batch = zip(*batch)
    X_lens = torch.tensor([len(x) for x in X_batch])
    Y_lens = torch.tensor([len(y) for y in Y_batch])
    X_pad = pad_sequence(X_batch, batch_first=True, padding_value=END_IDX)
    Y_pad = pad_sequence(Y_batch, batch_first=True, padding_value=END_IDX)
    return X_pad, X_lens, Y_pad, Y_lens


def load_split(path: str) -> tuple[list[str], list[str]]:
    inputs, targets = [], []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            inputs.append(row['input'])
            targets.append(row['target'])
    return inputs, targets


def make_loader(path: str, batch_size: int, shuffle: bool = True) -> DataLoader:
    inputs, targets = load_split(path)
    dataset = SeqDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
