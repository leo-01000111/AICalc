import math
import torch
import torch.nn as nn

from vocab import VOCAB_SIZE, START_IDX, END_IDX, PAD_IDX


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerCalc(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
            for _ in range(num_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, sz: int, device) -> torch.Tensor:
        # Upper-triangular -inf mask so each position only attends to past tokens
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def _padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        # True where token == PAD_IDX (PyTorch: True = ignore)
        return seq == PAD_IDX

    def encode(
        self,
        src: torch.Tensor,                          # (batch, src_len)
        src_key_padding_mask: torch.Tensor | None = None,  # (batch, src_len)
    ) -> torch.Tensor:
        x = self.pos_enc(self.src_embedding(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return self.encoder_norm(x)  # (batch, src_len, d_model)

    def decode(
        self,
        tgt: torch.Tensor,                               # (batch, tgt_len)
        memory: torch.Tensor,                            # (batch, src_len, d_model)
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.pos_enc(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder_layers:
            x = layer(
                x, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.decoder_norm(x)  # (batch, tgt_len, d_model)

    def forward(
        self,
        src: torch.Tensor,  # (batch, src_len)
        tgt: torch.Tensor,  # (batch, tgt_len)
    ) -> torch.Tensor:      # (batch, tgt_len-1, vocab_size)
        src_pad_mask = self._padding_mask(src)
        tgt_in = tgt[:, :-1]   # feed [<START>, d1, ..., dN]
        tgt_pad_mask = self._padding_mask(tgt_in)
        causal = self._causal_mask(tgt_in.size(1), src.device)

        memory = self.encode(src, src_key_padding_mask=src_pad_mask)
        dec_out = self.decode(
            tgt_in, memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.fc_out(dec_out)  # (batch, tgt_len-1, vocab_size)

    @torch.no_grad()
    def inference(
        self,
        src: torch.Tensor,  # (batch, src_len)
        max_len: int = 10,
    ) -> list[list[int]]:
        device = src.device
        batch_size = src.size(0)
        src_pad_mask = self._padding_mask(src)
        memory = self.encode(src, src_key_padding_mask=src_pad_mask)

        tgt = torch.full((batch_size, 1), START_IDX, dtype=torch.long, device=device)
        results = [[] for _ in range(batch_size)]
        done = [False] * batch_size

        for _ in range(max_len):
            causal = self._causal_mask(tgt.size(1), device)
            dec_out = self.decode(tgt, memory, tgt_mask=causal,
                                  memory_key_padding_mask=src_pad_mask)
            logits = self.fc_out(dec_out[:, -1, :])   # (batch, vocab_size)
            next_tok = logits.argmax(dim=-1)           # (batch,)
            tgt = torch.cat([tgt, next_tok.unsqueeze(1)], dim=1)

            for b in range(batch_size):
                if not done[b]:
                    tok = next_tok[b].item()
                    if tok == END_IDX:
                        done[b] = True
                    else:
                        results[b].append(tok)
            if all(done):
                break

        return results
