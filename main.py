import json
import os
import re
import torch

from model_seq2seq import Seq2SeqAttention
from model_transformer import TransformerCalc
from vocab import VOCAB_SIZE, tokenize_input, decode_output

INPUT_RE = re.compile(r'^(\d+)([+\-])(\d+)$')


def list_checkpoints(model_type: str) -> list[str]:
    folder = os.path.join('models', model_type)
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, f))
    )


def load_seq2seq(folder: str, device) -> Seq2SeqAttention:
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
    return model


def load_transformer(folder: str, device) -> TransformerCalc:
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
    return model


def pick_checkpoint(model_type: str) -> str | None:
    checkpoints = list_checkpoints(model_type)
    if not checkpoints:
        print(f"No saved checkpoints found in models/{model_type}/")
        print(f"Run train_{'seq2seq' if model_type == 'seq2seq' else 'transformer'}.py first.")
        return None
    print(f"\nAvailable checkpoints:")
    for i, name in enumerate(checkpoints):
        print(f"  [{i}] {name}")
    while True:
        choice = input("Select checkpoint index: ").strip()
        if choice.isdigit() and int(choice) < len(checkpoints):
            return os.path.join('models', model_type, checkpoints[int(choice)])
        print("Invalid choice.")


def ground_truth(expr: str) -> str:
    m = INPUT_RE.match(expr)
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    result = (a + b) % 128 if op == '+' else (a - b) % 128
    return str(result)


def predict(model, expr: str, model_type: str, device) -> str:
    toks = torch.tensor([tokenize_input(expr)], dtype=torch.long, device=device)
    if model_type == 'seq2seq':
        lens = torch.tensor([toks.size(1)])
        indices = model.inference(toks, lens, max_len=10)[0]
    else:
        indices = model.inference(toks, max_len=10)[0]
    return decode_output(indices)


def main():
    print("=" * 45)
    print("  AICalc — Neural Arithmetic Calculator")
    print("=" * 45)
    print("Choose model:")
    print("  [1] Seq2Seq + Bahdanau Attention")
    print("  [2] Transformer")

    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice in ('1', '2'):
            break
        print("Please enter 1 or 2.")

    model_type = 'seq2seq' if choice == '1' else 'transformer'
    folder = pick_checkpoint(model_type)
    if folder is None:
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model_type == 'seq2seq':
        model = load_seq2seq(folder, device)
    else:
        model = load_transformer(folder, device)

    print(f"\nLoaded {model_type} model from {folder}")
    print("Enter expressions like  125+42  or  20-100")
    print("Numbers must be in [0, 127]. Type 'quit' to exit.\n")

    while True:
        expr = input(">> ").strip().replace(' ', '')
        if expr.lower() in ('quit', 'exit', 'q'):
            break

        m = INPUT_RE.match(expr)
        if not m:
            print("Invalid format. Use digits and + or -, e.g. 125+42")
            continue

        a, b = int(m.group(1)), int(m.group(3))
        if not (0 <= a <= 127 and 0 <= b <= 127):
            print("Both numbers must be in [0, 127].")
            continue

        prediction = predict(model, expr, model_type, device)
        gt = ground_truth(expr)
        match = "OK" if prediction == gt else "WRONG"
        print(f"= {prediction}  (ground truth: {gt}) [{match}]")


if __name__ == '__main__':
    main()
