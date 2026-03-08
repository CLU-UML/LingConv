import argparse
import json
import multiprocessing as mp
import os

import joblib
import numpy as np
import pandas as pd
from datasets import load_from_disk
from evaluate import load

from compute_lng import compute_lng
from const import eval_indices, lftkplus_names, name_map, type_map, used_indices


SCALER = joblib.load("assets/scaler.bin")
BERTSCORE = load("bertscore")
BLEU = load("bleu")
LNG_NAMES = lftkplus_names


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to .txt, .csv, or .json predictions file.")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--data", default="ling_conversion")
    parser.add_argument("--split", default="test")
    parser.add_argument("--reference", choices=["target", "source"], default="target")
    parser.add_argument("--approximate", action="store_true")
    parser.add_argument("--disc_ckpt", help="HF id or local checkpoint for approximate linguistic scoring.")
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--sample_size", type=int)
    parser.add_argument("--save_breakdown")
    return parser.parse_args()


def load_predictions(path):
    _, extension = os.path.splitext(path)
    if extension == ".txt":
        with open(path) as handle:
            return [line.strip() for line in handle if line.strip()]
    if extension == ".csv":
        frame = pd.read_csv(path)
        if "prediction" not in frame.columns:
            raise ValueError("CSV predictions must contain a 'prediction' column.")
        return frame["prediction"].fillna("").tolist()
    if extension == ".json":
        with open(path) as handle:
            return json.load(handle)
    raise ValueError(f"Unsupported prediction file format: {extension}")


def get_lng(texts, n_jobs):
    worker_count = max(1, min(n_jobs, os.cpu_count() or 1))
    with mp.Pool(worker_count) as pool:
        return list(pool.imap(compute_lng, texts, chunksize=5))


def estimate_ling_preds(texts, disc_ckpt):
    from model import LingDiscPipeline

    pipe = LingDiscPipeline(disc_ckpt=disc_ckpt)
    outputs = []
    for text in texts:
        outputs.append(pipe(text).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def compute_ling_metrics(predictions, references_ling_raw, source_ling_raw, source_dataset, approximate=False, disc_ckpt=None, n_jobs=8):
    references_ling = SCALER.transform(references_ling_raw)
    source_ling = SCALER.transform(source_ling_raw)

    if approximate:
        if not disc_ckpt:
            raise ValueError("--disc_ckpt is required with --approximate.")
        predictions_ling = estimate_ling_preds(predictions, disc_ckpt=disc_ckpt)
        predictions_ling_raw = SCALER.inverse_transform(predictions_ling)
    else:
        predictions_ling_raw = np.array(get_lng(predictions, n_jobs=n_jobs))[:, used_indices]
        predictions_ling = SCALER.transform(predictions_ling_raw)

    errors_raw = np.abs(predictions_ling_raw - references_ling_raw)
    errors_target = (predictions_ling - references_ling) ** 2
    errors_source = (predictions_ling - source_ling) ** 2

    metrics = {name_map[LNG_NAMES[idx]]: errors_raw[:, idx].mean() for idx in eval_indices}
    metrics["mse_t"] = errors_target.mean()
    metrics["mse_s"] = errors_source.mean()

    index_errors = errors_target.mean(axis=0)
    types = np.array([type_map[name] for name in LNG_NAMES])
    for feature_type in np.unique(types):
        metrics[feature_type] = index_errors[types == feature_type].mean()

    for dataset_name in np.unique(source_dataset):
        mask = source_dataset == dataset_name
        metrics[f"mse_t_{dataset_name}"] = errors_target[mask].mean()
        metrics[f"mse_s_{dataset_name}"] = errors_source[mask].mean()

    breakdown = pd.DataFrame(
        {
            "prediction": predictions,
            "mse_t": errors_target.mean(axis=1),
            "mse_s": errors_source.mean(axis=1),
            "source": source_dataset,
        }
    )
    return metrics, breakdown


def main():
    args = parse_args()
    predictions = load_predictions(args.predictions)
    dataset = load_from_disk(os.path.join(args.data_dir, args.data))[args.split]

    n_examples = min(len(predictions), len(dataset))
    predictions = predictions[:n_examples]
    dataset = dataset.select(range(n_examples))

    if args.sample_size is not None:
        n_examples = min(args.sample_size, len(dataset))
        indices = np.random.default_rng(0).choice(len(dataset), size=n_examples, replace=False)
        predictions = list(np.array(predictions)[indices])
        dataset = dataset.select(indices.tolist())

    references = dataset["sentence2"]
    sources = dataset["sentence1"]
    source_dataset = np.array(dataset["source"])

    ling_metrics, breakdown = compute_ling_metrics(
        predictions=predictions,
        references_ling_raw=np.array(dataset["sentence2_ling"]),
        source_ling_raw=np.array(dataset["sentence1_ling"]),
        source_dataset=source_dataset,
        approximate=args.approximate,
        disc_ckpt=args.disc_ckpt,
        n_jobs=args.n_jobs,
    )

    bleu_references = [[reference] for reference in references]
    ling_metrics["bleu"] = BLEU.compute(predictions=predictions, references=bleu_references)["bleu"]

    bert_references = references if args.reference == "target" else sources
    ling_metrics["berts"] = np.mean(BERTSCORE.compute(predictions=predictions, references=bert_references, lang="en")["f1"])

    prediction_array = np.array(predictions)
    bert_reference_array = np.array(bert_references)
    for dataset_name in np.unique(source_dataset):
        mask = source_dataset == dataset_name
        ling_metrics[f"berts_{dataset_name}"] = np.mean(
            BERTSCORE.compute(
                predictions=prediction_array[mask].tolist(),
                references=bert_reference_array[mask].tolist(),
                lang="en",
            )["f1"]
        )

    if args.save_breakdown:
        breakdown_dir = os.path.dirname(args.save_breakdown)
        if breakdown_dir:
            os.makedirs(breakdown_dir, exist_ok=True)
        breakdown.to_csv(args.save_breakdown, index=False)

    print("\n".join(f"{key:20s}: {value:.4f}" for key, value in sorted(ling_metrics.items())))


if __name__ == "__main__":
    main()
