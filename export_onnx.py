"""Export the AICalc models to ONNX for the in-browser demo, and check parity.

    python export_onnx.py            # export + write meta.json/split.bin/wrong.bin
    python export_onnx.py --check    # greedy-decode all 32768 inputs with onnxruntime
                                     # and compare with PyTorch model.inference
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from dataset import load_split
from evaluate import load_seq2seq, load_transformer
from vocab import VOCAB, START_IDX, END_IDX, decode_output, tokenize_input

SITE = 'C:/TuttoNelMondo/PROJECTS/0_Website/leo-01000111.github.io'
OUT = os.path.join(SITE, 'assets', 'aicalc')
S2S_DIR = 'models/seq2seq/2026-06-16_17-48-06'
TR_DIR = 'models/transformer/2026-06-16_14-53-38'
MAX_LEN = 10
OPSET = 17
PAD_S = 9   # '127+127' -> <START> + 7 chars + <END>


# ---------- wrappers ----------
class S2SEncoder(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, src):
        e = self.enc
        emb = e.embedding(src)
        out, h_n = e.gru(emb)              # batch 1, no padding: packing is a no-op
        H = e.proj(out)
        # The trained model attends over zero-padded H (batches are padded to the longest
        # input, 9 tokens in practice, and attention has no mask). Reproduce that: pad H
        # with zeros to PAD_S so batch-1 inference equals the evaluated behaviour.
        H = torch.nn.functional.pad(H, (0, 0, 0, PAD_S - H.size(1)))
        h_n = h_n.view(e.num_layers, 2, -1, e.hidden_size)
        h0 = (h_n[:, 0] + h_n[:, 1]) / 2
        return H, h0


class S2SDecoderStep(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, y_prev, h_prev, H):
        logits, h_new, alpha = self.dec.forward_step(y_prev, h_prev, H)
        return logits, h_new, alpha.squeeze(-1)   # alpha [1,S]


class TrEncoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, src):
        # The model uses PAD == END, so the END token that ends every input is masked as a
        # key even at batch size 1 (model.inference does the same). Keep that.
        return self.m.encode(src, src_key_padding_mask=(src == END_IDX))


class TrDecoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, tgt, memory):
        t = tgt.size(1)
        ar = torch.arange(t)
        mask = torch.zeros(t, t).masked_fill(ar[None, :] > ar[:, None], float('-inf'))
        s = memory.size(1)
        mem_pad = (torch.arange(s) == s - 1).unsqueeze(0)   # last memory slot = END token
        x = self.m.decode(tgt, memory, tgt_mask=mask, memory_key_padding_mask=mem_pad)
        return self.m.fc_out(x[:, -1, :])


def export_dynamo(module, args, path, in_names, out_names, dyn_shapes):
    # nn.MultiheadAttention bakes the traced sequence length into Reshape nodes with the legacy
    # tracer, so the transformer uses the torch.export based exporter with real dynamic axes.
    module.eval()
    torch.onnx.export(module, args, path, input_names=in_names, output_names=out_names,
                      dynamic_shapes=dyn_shapes, opset_version=OPSET, dynamo=True,
                      external_data=False, report=False, verbose=False)


def export(module, args, path, in_names, out_names, dyn):
    module.eval()
    # grad stays enabled on purpose: this disables nn.Transformer's fused fast path,
    # so the traced graph is the plain eval-mode math.
    torch.onnx.export(module, args, path, input_names=in_names, output_names=out_names,
                      dynamic_axes=dyn, opset_version=OPSET, dynamo=False)


# ---------- helpers ----------
def run_torch(model, kind, inputs):
    out = []
    for i in range(0, len(inputs), 512):
        chunk = list(inputs[i:i + 512])
        if kind == 'seq2seq':
            chunk.append('127+127')   # sentinel: every batch is padded to 9 like in training/eval
        t = [torch.tensor(tokenize_input(s)) for s in chunk]
        lens = torch.tensor([len(x) for x in t])
        X = pad_sequence(t, batch_first=True, padding_value=END_IDX)
        r = model.inference(X, lens, MAX_LEN) if kind == 'seq2seq' else model.inference(X, MAX_LEN)
        out += [decode_output(x) for x in r][:len(inputs[i:i + 512])]
    return out


def all_inputs():
    # index = a*256 + b*2 + (op=='-')
    return [f"{a}{'-' if o else '+'}{b}" for a in range(128) for b in range(128) for o in (0, 1)]


def truth(expr):
    for op in '+-':
        if op in expr:
            a, b = expr.split(op)
            a, b = int(a), int(b)
            return str((a + b) % 128 if op == '+' else (a - b) % 128)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def main_export():
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device('cpu')
    s2s, s2s_cfg = load_seq2seq(S2S_DIR, dev)
    tr, tr_cfg = load_transformer(TR_DIR, dev)

    src = torch.tensor([tokenize_input('88-93')])
    export(S2SEncoder(s2s.encoder), (src,), f'{OUT}/seq2seq_encoder.onnx',
           ['src'], ['H', 'h0'], {'src': {1: 'S'}, 'H': {1: 'S'}})
    Hs, h0 = S2SEncoder(s2s.encoder)(src)
    export(S2SDecoderStep(s2s.decoder), (torch.tensor([START_IDX]), h0, Hs),
           f'{OUT}/seq2seq_decoder_step.onnx', ['y_prev', 'h_prev', 'H'],
           ['logits', 'h_new', 'alpha'],
           {'H': {1: 'S'}, 'alpha': {1: 'S'}})
    from torch.export import Dim
    dS, dT = Dim('S', min=3, max=9), Dim('T', min=1, max=MAX_LEN + 1)
    export_dynamo(TrEncoder(tr), (src,), f'{OUT}/transformer_encoder.onnx',
                  ['src'], ['memory'], {'src': {1: dS}})
    mem = TrEncoder(tr)(src)
    tgt = torch.tensor([[START_IDX, 5, 6]])
    export_dynamo(TrDecoder(tr), (tgt, mem), f'{OUT}/transformer_decoder.onnx',
                  ['tgt', 'memory'], ['logits'], {'tgt': {1: dT}, 'memory': {1: dS}})

    # split.bin
    ins = all_inputs()
    split_of = {}
    for k, sp in enumerate(('train', 'val', 'test')):
        for e in load_split(f'data/{sp}.csv')[0]:
            split_of[e] = k
    assert len(split_of) == 32768
    split = np.array([split_of[e] for e in ins], dtype=np.uint8)
    packed = np.zeros(32768 // 4, dtype=np.uint8)
    for i in range(32768):
        packed[i >> 2] |= int(split[i]) << ((i & 3) * 2)
    packed.tofile(f'{OUT}/split.bin')

    # accuracies + wrong bitsets from PyTorch
    meta_models = {}
    wrong_bits = []
    for name, model, d, cfg in (('seq2seq', s2s, S2S_DIR, s2s_cfg),
                                ('transformer', tr, TR_DIR, tr_cfg)):
        preds = run_torch(model, name, ins)
        wrong = np.array([p != truth(e) for p, e in zip(preds, ins)])
        wrong_bits.append(np.packbits(wrong, bitorder='little'))
        sp = {}
        for k, s in enumerate(('train', 'val', 'test')):
            m = split == k
            sp[s] = {'n': int(m.sum()), 'accuracy_pct': round(float(100 * (1 - wrong[m].mean())), 2),
                     'wrong': int(wrong[m].sum())}
        meta_models[name] = {'checkpoint': d, 'config': cfg, 'params': n_params(model),
                             'accuracy': sp, 'wrong_total': int(wrong.sum())}
    np.concatenate(wrong_bits).tofile(f'{OUT}/wrong.bin')

    files = ['seq2seq_encoder.onnx', 'seq2seq_decoder_step.onnx',
             'transformer_encoder.onnx', 'transformer_decoder.onnx']
    meta = {
        'seq2seq_pad_to': PAD_S, 'vocab': VOCAB, 'start_idx': START_IDX, 'end_idx': END_IDX, 'max_len': MAX_LEN,
        'opset': OPSET,
        'onnx_bytes': {f: os.path.getsize(f'{OUT}/{f}') for f in files},
        'models': meta_models,
        'split_bin': {
            'file': 'split.bin', 'entries': 32768, 'bytes': 8192,
            'layout': 'index = a*256 + b*2 + (op=="-"); 2 bits per entry, 4 per byte; '
                      'entry i is in byte i>>2 at bit offset (i&3)*2; 0=train 1=val 2=test'},
        'wrong_bin': {
            'file': 'wrong.bin', 'bytes': 8192,
            'layout': 'two 4096-byte halves: first seq2seq, second transformer. Same entry index as '
                      'split.bin. Within a half: bit i = (byte[i>>3] >> (i&7)) & 1; '
                      '1 = greedy output differs from the true result'},
    }
    with open(f'{OUT}/meta.json', 'w') as f:
        json.dump(meta, f, indent=1)
    print(meta['onnx_bytes'])
    for n, m in meta_models.items():
        print(n, m['params'], m['accuracy'], 'wrong', m['wrong_total'])


# ---------- parity check ----------
def ort_s2s(sessions, expr):
    enc, dec = sessions
    src = np.array([tokenize_input(expr)], dtype=np.int64)
    H, h = enc.run(None, {'src': src})
    y = np.array([START_IDX], dtype=np.int64)
    out = []
    first = None
    for _ in range(MAX_LEN):
        logits, h, alpha = dec.run(None, {'y_prev': y, 'h_prev': h, 'H': H})
        if first is None:
            first = logits
        t = int(logits.argmax())
        y = np.array([t], dtype=np.int64)
        if t == END_IDX:
            break
        out.append(t)
    return decode_output(out), first


def ort_tr(sessions, expr):
    enc, dec = sessions
    src = np.array([tokenize_input(expr)], dtype=np.int64)
    mem = enc.run(None, {'src': src})[0]
    tgt = [START_IDX]
    out = []
    first = None
    for _ in range(MAX_LEN):
        logits = dec.run(None, {'tgt': np.array([tgt], dtype=np.int64), 'memory': mem})[0]
        if first is None:
            first = logits
        t = int(logits.argmax())
        tgt.append(t)
        if t == END_IDX:
            break
        out.append(t)
    return decode_output(out), first


def main_check():
    import onnxruntime as ort
    dev = torch.device('cpu')
    s2s, _ = load_seq2seq(S2S_DIR, dev)
    tr, _ = load_transformer(TR_DIR, dev)

    def mk(f):
        return ort.InferenceSession(f'{OUT}/{f}', providers=['CPUExecutionProvider'])
    sess = {'seq2seq': (mk('seq2seq_encoder.onnx'), mk('seq2seq_decoder_step.onnx')),
            'transformer': (mk('transformer_encoder.onnx'), mk('transformer_decoder.onnx'))}
    ins = all_inputs()
    bad_total = 0
    for kind, model, fn in (('seq2seq', s2s, ort_s2s), ('transformer', tr, ort_tr)):
        tp = run_torch(model, kind, ins)
        mism = 0
        maxdiff = 0.0
        nsamp = 0
        for k, e in enumerate(ins):
            p, lg = fn(sess[kind], e)
            if p != tp[k]:
                mism += 1
            if k % 97 == 0:   # logit diff vs unbatched torch, first decode step
                nsamp += 1
                t = torch.tensor([tokenize_input(e)])
                with torch.no_grad():
                    if kind == 'seq2seq':
                        H, h0 = model.encoder(t, torch.tensor([t.size(1)]))
                        H = torch.nn.functional.pad(H, (0, 0, 0, PAD_S - H.size(1)))
                        ref, _, _ = model.decoder.forward_step(torch.tensor([START_IDX]), h0, H)
                    else:
                        pm = t == END_IDX
                        mem = model.encode(t, src_key_padding_mask=pm)
                        ref = model.fc_out(model.decode(torch.tensor([[START_IDX]]), mem,
                                           tgt_mask=model._causal_mask(1, 'cpu'),
                                           memory_key_padding_mask=pm)[:, -1, :])
                maxdiff = max(maxdiff, float(np.abs(ref.numpy() - lg).max()))
        print(f'{kind}: {len(ins)} inputs, mismatches vs PyTorch = {mism}, '
              f'max |logit diff| (first step, {nsamp} samples) = {maxdiff:.3e}')
        bad_total += mism
    sys.exit(1 if bad_total else 0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    main_check() if a.check else main_export()
