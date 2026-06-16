import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from vocab import VOCAB_SIZE, START_IDX, END_IDX


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v   = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        s_prev: torch.Tensor,  # (batch, hidden)
        H: torch.Tensor,       # (batch, src_len, hidden)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # e_ij = v^T · tanh( W_s·s + W_h·h_j )
        s = self.W_s(s_prev).unsqueeze(1)          # (batch, 1, hidden)
        h = self.W_h(H)                             # (batch, src_len, hidden)
        scores = self.v(torch.tanh(s + h))          # (batch, src_len, 1)
        alpha = torch.softmax(scores, dim=1)        # (batch, src_len, 1)
        context = (alpha * H).sum(dim=1)            # (batch, hidden)
        return context, alpha


class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=END_IDX)
        self.gru  = nn.GRU(embed_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        # project concatenated fwd+bwd outputs down to hidden_size for attention
        self.proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,       # (batch, src_len)
        x_lens: torch.Tensor,  # (batch,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, x_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, h_n = self.gru(packed)

        H_bidir, _ = pad_packed_sequence(packed_out, batch_first=True)  # (batch, src_len, hidden*2)
        H = self.proj(H_bidir)                                           # (batch, src_len, hidden)

        # h_n: (num_layers*2, batch, hidden) — layout: [l0_fwd, l0_bwd, l1_fwd, l1_bwd, ...]
        # reshape → (num_layers, 2, batch, hidden), average fwd and bwd for decoder h_0
        h_n = h_n.view(self.num_layers, 2, -1, self.hidden_size)
        h_dec = (h_n[:, 0] + h_n[:, 1]) / 2                             # (num_layers, batch, hidden)
        return H, h_dec


class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=END_IDX)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward_step(
        self,
        y_prev: torch.Tensor,   # (batch,)
        h_prev: torch.Tensor,   # (num_layers, batch, hidden)
        H: torch.Tensor,        # (batch, src_len, hidden)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s_prev = h_prev[-1]                                          # (batch, hidden) — top layer
        context, alpha = self.attention(s_prev, H)                   # (batch, hidden), (batch, src_len, 1)
        emb = self.embedding(y_prev).unsqueeze(1)                    # (batch, 1, embed)
        gru_input = torch.cat([emb, context.unsqueeze(1)], dim=-1)  # (batch, 1, embed+hidden)
        out, h_new = self.gru(gru_input, h_prev)
        logits = self.fc_out(out.squeeze(1))                         # (batch, vocab_size)
        return logits, h_new, alpha


class Seq2SeqAttention(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.encoder = Seq2SeqEncoder(vocab_size, embed_size, hidden_size, num_layers)
        self.decoder = Seq2SeqDecoder(vocab_size, embed_size, hidden_size, num_layers)

    def forward(
        self,
        x: torch.Tensor,               # (batch, src_len)
        x_lens: torch.Tensor,          # (batch,)
        y: torch.Tensor,               # (batch, tgt_len)
        teacher_forcing_prob: float = 1.0,
    ) -> torch.Tensor:                 # (batch, tgt_len-1, vocab_size)
        H, h_enc = self.encoder(x, x_lens)
        h_dec = h_enc
        tgt_len = y.size(1)
        outputs = []
        y_t = y[:, 0]  # always feed <START> as first input

        for t in range(1, tgt_len):
            logit, h_dec, _ = self.decoder.forward_step(y_t, h_dec, H)
            outputs.append(logit)
            if random.random() < teacher_forcing_prob:
                y_t = y[:, t]
            else:
                y_t = logit.argmax(dim=-1)

        return torch.stack(outputs, dim=1)

    @torch.no_grad()
    def inference(
        self,
        x: torch.Tensor,       # (batch, src_len)
        x_lens: torch.Tensor,  # (batch,)
        max_len: int = 10,
    ) -> list[list[int]]:
        H, h_enc = self.encoder(x, x_lens)
        h_dec = h_enc
        batch_size = x.size(0)
        device = x.device

        y_t = torch.full((batch_size,), START_IDX, dtype=torch.long, device=device)
        results = [[] for _ in range(batch_size)]
        done = [False] * batch_size

        for _ in range(max_len):
            logit, h_dec, _ = self.decoder.forward_step(y_t, h_dec, H)
            y_t = logit.argmax(dim=-1)
            for b in range(batch_size):
                if not done[b]:
                    tok = y_t[b].item()
                    if tok == END_IDX:
                        done[b] = True
                    else:
                        results[b].append(tok)
            if all(done):
                break

        return results
