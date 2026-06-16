VOCAB = ['<START>', '<END>'] + [str(i) for i in range(10)] + ['+', '-']
VOCAB_SIZE = len(VOCAB)   # 14

START_IDX = VOCAB.index('<START>')  # 0
END_IDX   = VOCAB.index('<END>')    # 1
PAD_IDX   = END_IDX                 # reuse <END> as padding

char_to_idx = {ch: i for i, ch in enumerate(VOCAB)}
idx_to_char = {i: ch for i, ch in enumerate(VOCAB)}


def tokenize_input(s: str) -> list[int]:
    return [START_IDX] + [char_to_idx[c] for c in s] + [END_IDX]


def tokenize_target(s: str) -> list[int]:
    return [START_IDX] + [char_to_idx[c] for c in s] + [END_IDX]


def decode_output(indices: list[int]) -> str:
    result = []
    for idx in indices:
        if idx == END_IDX:
            break
        if idx == START_IDX:
            continue
        ch = idx_to_char.get(idx, '')
        if ch and ch not in ('<START>', '<END>'):
            result.append(ch)
    return ''.join(result)
