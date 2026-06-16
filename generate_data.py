import csv
import os
import random


DATA_FOLDER = 'data'
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
SEED        = 42


def generate_all_samples() -> list[tuple[str, str]]:
    samples = []
    for a in range(128):
        for b in range(128):
            for op in ['+', '-']:
                input_str = f"{a}{op}{b}"
                result = (a + b) % 128 if op == '+' else (a - b) % 128
                samples.append((input_str, str(result)))
    return samples


def save_splits(samples: list[tuple[str, str]]) -> None:
    rng = random.Random(SEED)
    rng.shuffle(samples)

    n = len(samples)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        'train.csv': samples[:n_train],
        'val.csv':   samples[n_train:n_train + n_val],
        'test.csv':  samples[n_train + n_val:],
    }

    os.makedirs(DATA_FOLDER, exist_ok=True)
    for filename, rows in splits.items():
        path = os.path.join(DATA_FOLDER, filename)
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['input', 'target'])
            writer.writerows(rows)
        print(f"  {path}: {len(rows)} samples")


if __name__ == '__main__':
    samples = generate_all_samples()
    print(f"Total samples: {len(samples)}")
    save_splits(samples)
    print("Done.")
