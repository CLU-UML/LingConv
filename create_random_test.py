import argparse
import os

import numpy as np
from datasets import DatasetDict, load_from_disk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--input", default="ling_conversion")
    parser.add_argument("--output", default="ling_conversion_random0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def main():
    args = parse_args()
    base_data = load_from_disk(os.path.join(args.data_dir, args.input))
    if args.split not in base_data:
        raise ValueError(f"Split {args.split!r} not found in {args.input!r}.")

    test_data = base_data[args.split]
    rng = np.random.default_rng(args.seed)
    shuffled_indices = rng.permutation(len(test_data))

    target_columns = [column for column in ["sentence2_lftk+", "sentence2_discr", "sentence2_ling"] if column in test_data.column_names]

    def replace_targets(_, idx):
        replacement_idx = int(shuffled_indices[idx])
        return {column: test_data[replacement_idx][column] for column in target_columns}

    shuffled_test = test_data.map(replace_targets, with_indices=True)
    DatasetDict({args.split: shuffled_test}).save_to_disk(os.path.join(args.data_dir, args.output))


if __name__ == "__main__":
    main()
