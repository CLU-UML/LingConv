import argparse
import os

import numpy as np
from datasets import load_from_disk

from const import used_indices


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("--output_path")
    return parser.parse_args()


def filter_ids(rows):
    return {
        "sentence1_ling": np.array(rows["sentence1_lftk+"])[:, used_indices],
        "sentence2_ling": np.array(rows["sentence2_lftk+"])[:, used_indices],
    }


def main():
    args = parse_args()
    output_path = args.output_path or args.input_path.rstrip("/") + "_filtered"

    data = load_from_disk(args.input_path)
    data = data.map(filter_ids, batched=True, batch_size=1000, num_proc=min(8, os.cpu_count() or 1))
    data.save_to_disk(output_path)


if __name__ == "__main__":
    main()
