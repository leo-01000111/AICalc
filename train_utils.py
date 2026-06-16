import json
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from vocab import VOCAB_SIZE, START_IDX, END_IDX, PAD_IDX, idx_to_char


def read_setup(path: str) -> dict:
    config = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, value = line.split('=', 1)
            value = value.strip()
            config[key.strip()] = float(value) if '.' in value else int(value)
    return config


def compute_loss(
    logits: torch.Tensor,    # (batch, seq_len, vocab_size)
    targets: torch.Tensor,   # (batch, seq_len)  — this is Y_pad[:, 1:]
    y_lens: torch.Tensor,    # (batch,)           — lengths of full Y sequences (including START+END)
    end_loss_weight: float,
) -> torch.Tensor:
    batch, seq_len, _ = logits.shape
    # Clone and relabel padding positions (beyond the real <END>) to -100
    # y_lens[b] is the length of the full Y sequence [<START>, d1, ..., dN, <END>]
    # After the [:, 1:] shift, real tokens occupy positions 0 .. y_lens[b]-2
    # Positions y_lens[b]-1 and beyond are padding
    labels = targets.clone()
    for b in range(batch):
        real_end = y_lens[b].item() - 1  # last valid position in labels (the real <END>)
        if real_end < seq_len:
            labels[b, real_end:] = -100  # everything after real <END> is padding

    flat_logits  = logits.reshape(-1, VOCAB_SIZE)
    flat_labels  = labels.reshape(-1)
    loss_per_tok = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction='none')

    # Extra weight on the real <END> positions
    end_weight = torch.ones(batch, seq_len, device=logits.device)
    for b in range(batch):
        real_end_pos = y_lens[b].item() - 2  # position of real <END> in labels (0-indexed)
        if 0 <= real_end_pos < seq_len:
            end_weight[b, real_end_pos] = end_loss_weight
    loss_per_tok = loss_per_tok * end_weight.reshape(-1)

    valid = (flat_labels != -100).sum().clamp(min=1)
    return loss_per_tok.sum() / valid


def sequence_accuracy(
    logits: torch.Tensor,    # (batch, seq_len, vocab_size)
    targets: torch.Tensor,   # (batch, seq_len)
    y_lens: torch.Tensor,    # (batch,)
) -> float:
    preds = logits.argmax(dim=-1)
    correct = 0
    batch = logits.size(0)
    for b in range(batch):
        real_len = y_lens[b].item() - 1  # number of real output tokens in labels
        pred_seq = preds[b, :real_len].tolist()
        tgt_seq  = targets[b, :real_len].tolist()
        if pred_seq == tgt_seq:
            correct += 1
    return correct / batch


def save_model(
    model: nn.Module,
    config: dict,
    model_type: str,
    resume_data: dict | None = None,
) -> str:
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    folder = os.path.join('models', model_type, timestamp)
    os.makedirs(folder, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(folder, 'model.pt'))
    with open(os.path.join(folder, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    if resume_data is not None:
        torch.save(resume_data, os.path.join(folder, 'resume.pt'))
    return folder


def ask_resume(model_type: str):
    """
    Scan models/<model_type>/ for checkpoints that contain resume.pt, show a
    numbered menu, and return a 5-tuple:
        (start_epoch, best_val_acc, model_state, optimizer_state, best_model_state)
    All state dicts are None when the user picks "start fresh".
    """
    folder = os.path.join('models', model_type)
    entries = []
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name, 'resume.pt')
            if os.path.isfile(path):
                data = torch.load(path, map_location='cpu')
                entries.append((name, data))

    if not entries:
        return 0, -1.0, None, None, None

    print(f"\nFound {len(entries)} resumable checkpoint(s) for {model_type}:")
    print("  [0] Start fresh")
    for i, (name, data) in enumerate(entries, 1):
        print(f"  [{i}] {name}  —  epoch {data['epoch']}, best val {data['best_val_acc']:.2%}")

    while True:
        choice = input("Choice: ").strip()
        if choice == '0':
            return 0, -1.0, None, None, None
        if choice.isdigit() and 1 <= int(choice) <= len(entries):
            name, data = entries[int(choice) - 1]
            print(f"Resuming from epoch {data['epoch']}, best val {data['best_val_acc']:.2%}")
            return (
                data['epoch'],
                data['best_val_acc'],
                data['model'],
                data['optimizer'],
                data.get('best_model'),
            )
        print(f"Enter 0–{len(entries)}.")
