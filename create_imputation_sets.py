import argparse
import os

import numpy as np
from datasets import load_from_disk


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", help="Optional save_to_disk dataset path used to infer test set size.")
    parser.add_argument("--n_samples", type=int, default=1887)
    parser.add_argument("--n_features", type=int, default=40)
    parser.add_argument("--percentages", default="20,40,60,80")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--output_dir", default="imputation_sets")
    return parser.parse_args()


def infer_sample_count(dataset_path, fallback_count):
    if not dataset_path:
        return fallback_count
    dataset = load_from_disk(dataset_path)
    return len(dataset["test"])


def main():
    args = parse_args()
    n_samples = infer_sample_count(args.dataset_path, args.n_samples)
    percentages = [int(value) for value in args.percentages.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]

    os.makedirs(args.output_dir, exist_ok=True)
    for percentage in percentages:
        n_impute = int(args.n_features * (percentage / 100.0))
        for seed in seeds:
            rng = np.random.default_rng(seed)
            imputation_indices = [
                rng.choice(args.n_features, size=n_impute, replace=False)
                for _ in range(n_samples)
            ]
            filename = f"imputation_p{percentage}_s{seed}.npy"
            np.save(os.path.join(args.output_dir, filename), np.array(imputation_indices))


if __name__ == "__main__":
    main()
