import multiprocessing as mp
import os
from functools import partial

import joblib
import numpy as np
import torch
from datasets import Sequence, Value, concatenate_datasets, load_from_disk
from sklearn.decomposition import FastICA
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from transformers import DataCollatorForSeq2Seq

from const import used_indices


def _num_proc():
    return max(1, min(os.cpu_count() or 1, 8))


def load_imputation_indices(percentage, seed):
    return np.load(f"imputation_sets/imputation_p{percentage}_s{seed}.npy")


def _fit_scaler(train_values, quantize, n_bins):
    if not quantize:
        scaler = StandardScaler()
    else:
        scaler = Pipeline(
            [
                ("kbins", KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="kmeans")),
                ("standard", StandardScaler()),
            ]
        )
    scaler.fit(train_values)
    return scaler


def _impute_single_feature(sample_tuple, ling_collection_scaled):
    feature, sample_indices = sample_tuple
    feature_mod = feature.copy()[used_indices]
    feature_mod[sample_indices] = np.nan
    imputer = IterativeImputer(
        estimator=Ridge(alpha=1e3, fit_intercept=False),
        imputation_order="random",
        max_iter=100,
        random_state=0,
    )
    combined_matrix = np.vstack([ling_collection_scaled, feature_mod.reshape(1, -1)])
    interpolated_matrix = imputer.fit_transform(combined_matrix)
    feature[used_indices] = interpolated_matrix[-1]
    return feature


def impute_features(features, imputation_indices, ling_collection_scaled):
    ling_collection_scaled = ling_collection_scaled[:, used_indices]
    sample_list = [(features[i], imputation_indices[i]) for i in range(len(imputation_indices))]
    n_jobs = _num_proc()
    with mp.Pool(n_jobs) as pool:
        imputed_results = list(
            pool.imap(
                partial(_impute_single_feature, ling_collection_scaled=ling_collection_scaled),
                sample_list,
                chunksize=5,
            )
        )
    return np.array(imputed_results)


def _rename_ling_columns(data, src_lng):
    target_column = f"sentence2_{src_lng}"
    if src_lng == "ling":
        return data

    if "test" not in data or target_column not in data["test"].column_names:
        raise ValueError(f"Could not find linguistic columns for src_lng={src_lng!r}.")

    rename_map = {target_column: "sentence2_ling"}
    source_column = f"sentence1_{src_lng}"
    if source_column in data["test"].column_names:
        rename_map[source_column] = "sentence1_ling"

    for split in data.keys():
        split_columns = data[split].column_names
        removable = [column for column in ["sentence1_ling", "sentence2_ling"] if column in split_columns]
        if removable:
            data[split] = data[split].remove_columns(removable)
        split_rename_map = {old: new for old, new in rename_map.items() if old in split_columns}
        data[split] = data[split].rename_columns(split_rename_map)

    return data


def prepare_ling(
    data,
    lng_ids=None,
    quantize=True,
    n_bins=20,
    src_lng="ling",
    do_imputation=False,
    imputation_percentage=None,
    imputation_seed=None,
):
    data = _rename_ling_columns(data, src_lng)

    def fix_nan(rows):
        output = {"sentence2_ling": np.nan_to_num(rows["sentence2_ling"])}
        if "sentence1_ling" in rows:
            output["sentence1_ling"] = np.nan_to_num(rows["sentence1_ling"])
        return output

    data = data.map(fix_nan, batched=True)

    if lng_ids is not None:
        def select_ids(rows):
            output = {"sentence2_ling": np.array(rows["sentence2_ling"])[:, lng_ids]}
            if "sentence1_ling" in rows:
                output["sentence1_ling"] = np.array(rows["sentence1_ling"])[:, lng_ids]
            return output

        data = data.map(select_ids, batched=True)

    lng_dim = len(data["test"][0]["sentence2_ling"])
    scaler_path = "assets/scaler.bin" if lng_dim == 40 else None
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        if "train" not in data:
            raise FileNotFoundError("No train split available to fit the linguistic scaler.")
        train_key = "sentence1_ling" if "sentence1_ling" in data["train"].column_names else "sentence2_ling"
        scaler = _fit_scaler(data["train"][train_key], quantize, n_bins)

    def scale(rows):
        output = {"sentence2_ling": scaler.transform(rows["sentence2_ling"])}
        if "sentence1_ling" in rows:
            output["sentence1_ling"] = scaler.transform(rows["sentence1_ling"])
        return output

    data = data.map(scale, batched=True)

    if do_imputation and "train" in data and "test" in data:
        imputation_indices = load_imputation_indices(imputation_percentage, imputation_seed)
        sample_size = min(1000, len(data["train"]))
        sampled_ids = np.random.default_rng(0).choice(len(data["train"]), size=sample_size, replace=False)
        ling_collection_scaled = np.array(data["train"].select(sampled_ids)["sentence2_ling"])
        test_ling = np.array(data["test"]["sentence2_ling"])
        imputed_ling = impute_features(test_ling, imputation_indices, ling_collection_scaled)
        data["test"] = data["test"].remove_columns(["sentence2_ling"])
        data["test"] = data["test"].add_column("sentence2_ling", imputed_ling.tolist())

    return data, scaler


def swap(name):
    swap_mapping = {
        "input_ids": "labels",
        "labels": "input_ids",
        "attention_mask": "labels_attention_mask",
        "labels_attention_mask": "attention_mask",
        "sentence1": "sentence2",
        "sentence2": "sentence1",
        "sentence1_ling": "sentence2_ling",
        "sentence2_ling": "sentence1_ling",
    }
    return swap_mapping.get(name, name)


def augment_reverse(data):
    swap_cols = {
        column: swap(column)
        for column in data.column_names
        if column in {
            "input_ids",
            "labels",
            "attention_mask",
            "labels_attention_mask",
            "sentence1",
            "sentence2",
            "sentence1_ling",
            "sentence2_ling",
        }
    }
    return concatenate_datasets([data, data.rename_columns(swap_cols)])


def augment_same(data):
    data1 = data.remove_columns(
        [column for column in data.column_names if column in {"sentence2", "sentence2_ling", "labels", "labels_attention_mask"}]
    )
    data1 = concatenate_datasets(
        [
            data1.rename_columns({column: swap(column) for column in data1.column_names if column in {"input_ids", "attention_mask", "sentence1", "sentence1_ling"}})
        ],
        axis=1,
    ).shuffle()
    sample_size = max(1, int(0.25 * len(data)))
    return concatenate_datasets([data, data1.select(range(sample_size))])


def replace_ling(args, data, scaler):
    lingpred_data = load_from_disk(os.path.join(args.data_dir, "lingpred", args.data))
    for split in data.keys():
        data[split] = data[split].remove_columns(["sentence1_ling", "sentence2_ling"])
        data[split] = data[split].add_column("sentence1_ling", lingpred_data[split]["lingpred1"])
        data[split] = data[split].add_column("sentence2_ling", lingpred_data[split]["lingpred2"])

    def scale(rows):
        return {
            "sentence1_ling": scaler.transform(rows["sentence1_ling"]),
            "sentence2_ling": scaler.transform(rows["sentence2_ling"]),
        }

    return data.map(scale, batched=True)


def _tokenize_rows(rows, tokenizer, max_length, prepend_prompt, prompt_text):
    input_text = rows["sentence1"]
    if prepend_prompt:
        input_text = [prompt_text + text for text in input_text]

    model_inputs = tokenizer(
        input_text,
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    label_inputs = tokenizer(
        text_target=rows["sentence2"],
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = label_inputs["input_ids"]
    model_inputs["labels_attention_mask"] = label_inputs["attention_mask"]
    return model_inputs


def _build_dev_ood_split(data):
    if "dev" not in data:
        return data

    rng = np.random.default_rng(0)
    original_targets = data["dev"]["sentence2_ling"]
    shuffled_targets = [original_targets[i] for i in rng.permutation(len(original_targets))]
    data["dev_ood"] = data["dev"].map(lambda _, idx: {"sentence2_ling": shuffled_targets[idx]}, with_indices=True)
    return data


def load_data(args, tokenizer, return_data=False):
    data = load_from_disk(os.path.join(args.data_dir, args.data))

    if "train" in data and args.data_sources is not None and "source" in data["train"].column_names:
        allowed_sources = set(args.data_sources)
        data = data.filter(lambda row: row["source"] in allowed_sources)

    data, scaler = prepare_ling(
        data,
        lng_ids=args.lng_ids,
        quantize=args.quantize_lng,
        n_bins=args.quant_nbins,
        src_lng=args.src_lng,
        do_imputation=args.do_imputation,
        imputation_percentage=args.imputation_percentage,
        imputation_seed=args.imputation_seed,
    )

    ica = None
    if args.use_ica and "train" in data:
        ica = FastICA(n_components=args.n_ica, random_state=args.seed)
        ica.fit(np.array(data["train"]["sentence1_ling"]))

        def reduce(rows):
            output = {"sentence2_ling": ica.transform(rows["sentence2_ling"])}
            if "sentence1_ling" in rows:
                output["sentence1_ling"] = ica.transform(rows["sentence1_ling"])
            return output

        data = data.map(reduce, batched=True)

    data = data.map(
        partial(
            _tokenize_rows,
            tokenizer=tokenizer,
            max_length=args.max_length,
            prepend_prompt=args.prepend_prompt,
            prompt_text=args.prompt_text,
        ),
        batched=True,
        batch_size=1000,
        num_proc=_num_proc(),
    )
    data = data.cast_column("labels", Sequence(feature=Value("int32")))
    data = data.cast_column("labels_attention_mask", Sequence(feature=Value("int8")))

    if args.use_lingpred:
        data = replace_ling(args, data, scaler)

    if "train" in data and args.aug_same:
        data["train"] = augment_same(augment_reverse(data["train"]))

    data = _build_dev_ood_split(data)

    if "dev" in data:
        sample_size = min(args.max_eval_samples, len(data["dev"]))
        sampled_ids = np.random.default_rng(args.seed).choice(len(data["dev"]), size=sample_size, replace=False)
        data["dev"] = data["dev"].select(sampled_ids)
        data["dev_ood"] = data["dev_ood"].select(sampled_ids)

    keep_cols = {
        "input_ids",
        "attention_mask",
        "labels",
        "labels_attention_mask",
        "sentence1",
        "sentence2",
        "sentence1_ling",
        "sentence2_ling",
        "sentence1_ling_40",
        "sentence2_ling_40",
        "source",
    }
    for split in data.keys():
        remove_cols = set(data[split].column_names) - keep_cols
        if remove_cols:
            data[split] = data[split].remove_columns(sorted(remove_cols))

    return data, scaler, ica


class LingDataCollator:
    def __init__(self, tokenizer):
        self.base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")

    def __call__(self, features):
        seq2seq_features = [
            {key: value for key, value in feature.items() if key in {"input_ids", "attention_mask", "labels"}}
            for feature in features
        ]
        batch = self.base_collator(seq2seq_features)

        for key in ["sentence1_ling", "sentence2_ling", "sentence1_ling_40", "sentence2_ling_40"]:
            if key in features[0]:
                batch[key] = torch.tensor([feature[key] for feature in features], dtype=torch.float32)

        return batch
