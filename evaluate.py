import json
import os
import random
import torch

from dataset import make_loader, load_split
from model_seq2seq import Seq2SeqAttention
from model_transformer import TransformerCalc
from train_utils import compute_loss, sequence_accuracy
from vocab import VOCAB_SIZE, START_IDX, decode_output, tokenize_input


def list_checkpoints(model_type: str) -> list[str]:
    folder = os.path.join('models', model_type)
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, f))
    )


def load_seq2seq(folder: str, device) -> tuple[Seq2SeqAttention, dict]:
    with open(os.path.join(folder, 'config.json')) as f:
        cfg = json.load(f)
    model = Seq2SeqAttention(
        vocab_size=VOCAB_SIZE,
        embed_size=int(cfg['embed_size']),
        hidden_size=int(cfg['hidden_size']),
        num_layers=int(cfg['num_layers']),
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(folder, 'model.pt'), map_location=device))
    model.eval()
    return model, cfg


def load_transformer(folder: str, device) -> tuple[TransformerCalc, dict]:
    with open(os.path.join(folder, 'config.json')) as f:
        cfg = json.load(f)
    model = TransformerCalc(
        vocab_size=VOCAB_SIZE,
        d_model=int(cfg['d_model']),
        nhead=int(cfg['nhead']),
        num_encoder_layers=int(cfg['num_encoder_layers']),
        num_decoder_layers=int(cfg['num_decoder_layers']),
        dim_feedforward=int(cfg['dim_feedforward']),
        dropout=float(cfg['dropout']),
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(folder, 'model.pt'), map_location=device))
    model.eval()
    return model, cfg


def _pick_checkpoint(model_type: str) -> str | None:
    checkpoints = list_checkpoints(model_type)
    if not checkpoints:
        print(f"No saved {model_type} checkpoints found in models/{model_type}/")
        return None
    print(f"\nAvailable {model_type} checkpoints:")
    for i, name in enumerate(checkpoints):
        print(f"  [{i}] {name}")
    while True:
        choice = input(f"Select {model_type} checkpoint index: ").strip()
        if choice.isdigit() and int(choice) < len(checkpoints):
            return os.path.join('models', model_type, checkpoints[int(choice)])
        print("Invalid choice.")


@torch.no_grad()
def evaluate_seq2seq(model, loader, device, end_loss_weight) -> dict:
    total_loss, total_acc, n = 0.0, 0.0, 0
    for X, X_lens, Y, Y_lens in loader:
        X, X_lens = X.to(device), X_lens.to(device)
        Y, Y_lens = Y.to(device), Y_lens.to(device)
        logits = model(X, X_lens, Y, teacher_forcing_prob=0.0)
        labels = Y[:, 1:]
        total_loss += compute_loss(logits, labels, Y_lens, end_loss_weight).item() * X.size(0)
        total_acc  += sequence_accuracy(logits, labels, Y_lens) * X.size(0)
        n += X.size(0)
    return {'loss': total_loss / n, 'seq_acc': total_acc / n, 'n': n}


@torch.no_grad()
def evaluate_transformer(model, loader, device, end_loss_weight) -> dict:
    total_loss, total_acc, n = 0.0, 0.0, 0
    for X, X_lens, Y, Y_lens in loader:
        X = X.to(device)
        Y, Y_lens = Y.to(device), Y_lens.to(device)
        logits = model(X, Y)
        labels = Y[:, 1:]
        total_loss += compute_loss(logits, labels, Y_lens, end_loss_weight).item() * X.size(0)
        total_acc  += sequence_accuracy(logits, labels, Y_lens) * X.size(0)
        n += X.size(0)
    return {'loss': total_loss / n, 'seq_acc': total_acc / n, 'n': n}


def predict_seq2seq(model, expr: str, device) -> str:
    toks = torch.tensor([tokenize_input(expr)], dtype=torch.long, device=device)
    lens = torch.tensor([toks.size(1)])
    result = model.inference(toks, lens, max_len=10)
    return decode_output(result[0])


def predict_transformer(model, expr: str, device) -> str:
    toks = torch.tensor([tokenize_input(expr)], dtype=torch.long, device=device)
    result = model.inference(toks, max_len=10)
    return decode_output(result[0])


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load both models
    seq2seq_folder = _pick_checkpoint('seq2seq')
    transformer_folder = _pick_checkpoint('transformer')

    if not seq2seq_folder or not transformer_folder:
        print("Both models must be saved before running evaluate.py.")
        return

    seq2seq_model, seq2seq_cfg   = load_seq2seq(seq2seq_folder, device)
    transformer_model, trans_cfg = load_transformer(transformer_folder, device)

    # Evaluate on test set
    test_loader = make_loader('data/test.csv', batch_size=256, shuffle=False)

    print("\nEvaluating on test set...")
    s2s_results   = evaluate_seq2seq(seq2seq_model, test_loader, device, seq2seq_cfg['end_loss_weight'])
    trans_results = evaluate_transformer(transformer_model, test_loader, device, trans_cfg['end_loss_weight'])

    # Comparison table
    print("\n" + "=" * 60)
    print(f"{'Model':<20} {'Test Loss':>12} {'Seq Acc':>10} {'N':>8}")
    print("-" * 60)
    print(f"{'Seq2Seq+Attention':<20} {s2s_results['loss']:>12.4f} {s2s_results['seq_acc']:>10.2%} {s2s_results['n']:>8}")
    print(f"{'Transformer':<20} {trans_results['loss']:>12.4f} {trans_results['seq_acc']:>10.2%} {trans_results['n']:>8}")
    print("=" * 60)

    # Show 10 random examples
    inputs, targets = load_split('data/test.csv')
    sample_idx = random.sample(range(len(inputs)), min(10, len(inputs)))

    print("\nRandom examples:")
    print(f"{'Input':<12} {'Ground Truth':>14} {'Seq2Seq':>10} {'Transformer':>12}")
    print("-" * 52)
    for i in sample_idx:
        expr = inputs[i]
        gt   = targets[i]
        s2s_pred   = predict_seq2seq(seq2seq_model, expr, device)
        trans_pred = predict_transformer(transformer_model, expr, device)
        print(f"{expr:<12} {gt:>14} {s2s_pred:>10} {trans_pred:>12}")


if __name__ == '__main__':
    main()
