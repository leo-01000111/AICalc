import torch
import torch.optim as optim

from dataset import make_loader
from model_transformer import TransformerCalc
from train_utils import read_setup, compute_loss, sequence_accuracy, save_model, ask_resume
from vocab import VOCAB_SIZE


def train_epoch(model, loader, optimizer, device, config):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for X, X_lens, Y, Y_lens in loader:
        X = X.to(device)
        Y, Y_lens = Y.to(device), Y_lens.to(device)

        optimizer.zero_grad()
        logits = model(X, Y)          # (batch, tgt_len-1, vocab_size)
        labels = Y[:, 1:]             # (batch, tgt_len-1)
        loss = compute_loss(logits, labels, Y_lens, config['end_loss_weight'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        optimizer.step()

        bs = X.size(0)
        total_loss += loss.item() * bs
        total_acc  += sequence_accuracy(logits, labels, Y_lens) * bs
        n += bs

    return total_loss / n, total_acc / n


@torch.no_grad()
def validate(model, loader, device, config):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for X, X_lens, Y, Y_lens in loader:
        X = X.to(device)
        Y, Y_lens = Y.to(device), Y_lens.to(device)

        logits = model(X, Y)
        labels = Y[:, 1:]
        loss = compute_loss(logits, labels, Y_lens, config['end_loss_weight'])

        bs = X.size(0)
        total_loss += loss.item() * bs
        total_acc  += sequence_accuracy(logits, labels, Y_lens) * bs
        n += bs

    return total_loss / n, total_acc / n


def main():
    config = read_setup('setup_transformer.txt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    torch.manual_seed(int(config['seed']))

    train_loader = make_loader('data/train.csv', int(config['batch_size']), shuffle=True)
    val_loader   = make_loader('data/val.csv',   int(config['batch_size']), shuffle=False)

    model = TransformerCalc(
        vocab_size=VOCAB_SIZE,
        d_model=int(config['d_model']),
        nhead=int(config['nhead']),
        num_encoder_layers=int(config['num_encoder_layers']),
        num_decoder_layers=int(config['num_decoder_layers']),
        dim_feedforward=int(config['dim_feedforward']),
        dropout=float(config['dropout']),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],
                            weight_decay=config.get('weight_decay', 0.0))

    start_epoch, best_val_acc, model_state, opt_state, best_model_state = ask_resume('transformer')
    if model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in model_state.items()})
        optimizer.load_state_dict(opt_state)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    best_state = best_model_state  # None on fresh start; set on first improving epoch

    num_epochs = int(config['num_epochs'])
    if start_epoch >= num_epochs:
        print(f"Already at epoch {start_epoch}; increase num_epochs (currently {num_epochs}) "
              f"in setup_transformer.txt to continue training.")
        return

    for epoch in range(start_epoch, num_epochs):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, device, config)
        vl_loss, vl_acc = validate(model, val_loader, device, config)

        print(
            f"Epoch {epoch+1:3d}/{num_epochs} | "
            f"Train Loss={tr_loss:.4f} Acc={tr_acc:.2%} | "
            f"Val Loss={vl_loss:.4f} Acc={vl_acc:.2%}"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest val accuracy: {best_val_acc:.2%}")
    answer = input("Save model? [S/N]: ").strip().upper()
    if answer == 'S':
        resume_data = {
            'model':      {k: v.cpu().clone() for k, v in model.state_dict().items()},
            'best_model': best_state,
            'optimizer':  optimizer.state_dict(),
            'epoch':      num_epochs,
            'best_val_acc': best_val_acc,
        }
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        config['model_type'] = 'transformer'
        folder = save_model(model, config, 'transformer', resume_data=resume_data)
        print(f"Model saved to {folder}")


if __name__ == '__main__':
    main()
