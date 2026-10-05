# AICalc

A calculator where the "compute" button is a neural network.

You type `125+42` or `20-100`. Instead of evaluating the expression, the model predicts the answer one character at a time, the way a language model generates text. There are two architectures: a GRU seq2seq with Bahdanau attention, and a Transformer encoder-decoder. Both work on addition and subtraction mod 128, so every operand and result is in [0, 127].

What I wanted to see was the training curves. Both models grok: they memorise the training set first (train accuracy climbs while validation stays flat), and then validation accuracy jumps up to match. The Transformer took about 500 epochs to generalise. The seq2seq got there faster, but only after I made its encoder bidirectional; before that it stalled at around 79%.

## Setup

You need Python 3.11+ and, for the GUI, tkinter.

```
pip install torch
python generate_data.py
```

`pyproject.toml` and `uv.lock` are also in the repo, so `uv sync` works as well.

`generate_data.py` writes `data/train.csv`, `data/val.csv` and `data/test.csv`. Together they hold all 32,768 operand/operation combinations, shuffled once with a fixed seed and split 70/15/15.

## Training

```
python train_seq2seq.py
python train_transformer.py
```

Hyperparameters are in `setup_seq2seq.txt` and `setup_transformer.txt`. Edit the files directly; there are no command-line flags. Both scripts keep the best validation checkpoint and ask whether to save it at the end. Checkpoints go to `models/<model_type>/<timestamp>/`, which is git-ignored, so a fresh clone has to train before the GUI or console has anything to load.

To continue a run (say, another 500 epochs on top of a saved one), start the script again. It finds the existing checkpoint and offers to resume. `num_epochs` in the setup file is the total epoch count, so raise it before resuming.

## Running

GUI:

```
python ui.py
```

Pick a model type, choose a checkpoint from the dropdown, press Load, then use the keypad or your keyboard. The right panel keeps a scrollable history with correct and wrong answers colour-coded.

Console:

```
python main.py
```

It asks for a model and checkpoint, then reads expressions until you type `quit`. Operands must be in [0, 127].

Evaluation, a side-by-side comparison of both models on the test split (needs a saved checkpoint of each):

```
python evaluate.py
```

## Architecture notes

Seq2seq: a bidirectional GRU encoder and a single-direction GRU decoder with additive (Bahdanau) attention. The encoder reads the input forwards and backwards, and the two final hidden states are averaged to start the decoder. Teacher forcing falls linearly from 100% to 0% over training. The optimiser is Adam with weight decay 1e-4.

Transformer: a standard encoder-decoder with sinusoidal positional encoding. It uses full teacher forcing throughout training and greedy autoregressive decoding at inference. It is trained with AdamW, and the weight decay is what eventually made it generalise; without it the model kept memorising.

Both models tokenise by character over a 14-token vocabulary: `<START>`, `<END>` (also used as padding), the digits 0 to 9, `+` and `-`.

## Results

Accuracy is exact match on the whole answer string, measured on the validation split.

| Model | Val acc |
|---|---|
| Seq2seq (bidirectional encoder, 300 epochs) | 92.66% |
| Transformer (about 500 epochs) | 96.66% |

The bidirectional encoder is what lifted the seq2seq from its ~79% plateau to the number above.
