# AICalc

A calculator where the "compute" button is a neural network.

You type `125+42` or `20-100`, and instead of evaluating the expression arithmetically, the model predicts the answer character by character — the same way a language model generates text. Two architectures are implemented and compared: a GRU-based seq2seq with Bahdanau attention, and a Transformer encoder-decoder. Both operate on addition and subtraction mod 128 (so all operands and results live in [0, 127]).

The interesting part isn't the calculator. It's watching the training curves. Both models exhibit **grokking** — they memorise the training set first (train accuracy climbs, validation stays flat), then at some point validation accuracy suddenly jumps to match. The transformer took ~500 epochs to generalise; the seq2seq did it faster but needed a bidirectional encoder to break past ~79%.

---

## Setup

```
pip install torch
python generate_data.py
```

That writes `data/train.csv`, `data/val.csv`, and `data/test.csv` — all 32,768 combinations of operands × operations, shuffled once with a fixed seed.

---

## Training

```
python train_seq2seq.py
python train_transformer.py
```

Hyperparameters live in `setup_seq2seq.txt` and `setup_transformer.txt` — edit them directly, no flags needed. Both scripts track the best validation checkpoint throughout and prompt to save at the end.

If you've already run a training session and want to continue it (say, another 500 epochs on top of a saved run), the script will detect the existing checkpoint and offer to resume from it. `num_epochs` in the setup file is the total epoch count, so bump it before resuming.

---

## Running

**GUI:**
```
python ui.py
```

Pick a model type, select a checkpoint from the dropdown, hit Load, then use the numpad or your keyboard. The right panel keeps a scrollable history with correct/wrong colour-coded.

**Console:**
```
python main.py
```

Prompts you to pick a model and checkpoint, then loops on expressions until you type `quit`.

**Evaluation** (side-by-side test-set comparison of both models):
```
python evaluate.py
```

---

## Architecture notes

**Seq2seq** — bidirectional GRU encoder, single-direction GRU decoder with additive (Bahdanau) attention. The encoder reads the input string both forwards and backwards; the two final hidden states are averaged to seed the decoder. Teacher forcing decays linearly from 100% to 0% over the course of training.

**Transformer** — standard encoder-decoder with sinusoidal positional encoding. No teacher forcing schedule; full teacher forcing throughout training, greedy autoregressive decoding at inference. AdamW with weight decay is what eventually triggers generalisation — without it the model memorises indefinitely.

Both models use character-level tokenisation over a 14-token vocabulary: `<START>`, `<END>` (doubled as padding), digits 0–9, `+`, `-`.

---

## Results

| Model | Val acc |
|---|---|
| Seq2seq (bidirectional) | 92.66% |
| Transformer | 96.66% |

The transformer number is after 500 epochs. The seq2seq number is preliminary — the bidirectional encoder pushed it from a ~79% ceiling to 70% by epoch 40, trajectory still climbing.
