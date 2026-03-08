import argparse
import os

from datasets import DatasetDict, concatenate_datasets, load_from_disk


SOURCE_SPECS = {
    "qqp": {"sentence1": "question1", "sentence2": "question2", "keep": lambda label: label == 1},
    "mrpc": {"sentence1": "sentence1", "sentence2": "sentence2", "keep": lambda label: label == 1},
    "stsb": {"sentence1": "sentence1", "sentence2": "sentence2", "keep": lambda label: label >= 3},
    "rte": {"sentence1": "sentence1", "sentence2": "sentence2", "keep": lambda label: label == 1},
    "anli": {"sentence1": "sentence1", "sentence2": "sentence2", "keep": lambda label: label == 0},
    "smf": {"sentence1": "sentence1", "sentence2": "sentence2", "keep": lambda label: label == 0},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--sources", default="qqp,mrpc,stsb")
    parser.add_argument("--output", default="ling_conversion")
    return parser.parse_args()


def rename_dev_split(data):
    if "dev" not in data and "validation" in data:
        data["dev"] = data["validation"]
        del data["validation"]
    return data


def normalize_columns(data, source_name, sentence1_name, sentence2_name):
    for split in data.keys():
        split_data = data[split]
        rename_map = {}
        if sentence1_name in split_data.column_names and sentence1_name != "sentence1":
            rename_map[sentence1_name] = "sentence1"
        if sentence2_name in split_data.column_names and sentence2_name != "sentence2":
            rename_map[sentence2_name] = "sentence2"
        if rename_map:
            split_data = split_data.rename_columns(rename_map)
        if "source" not in split_data.column_names:
            split_data = split_data.add_column("source", [source_name] * len(split_data))
        keep_columns = [column for column in split_data.column_names if column.startswith("sentence") or column == "source"]
        data[split] = split_data.remove_columns(sorted(set(split_data.column_names) - set(keep_columns)))
    return data


def main():
    args = parse_args()
    sources = [source.strip() for source in args.sources.split(",") if source.strip()]

    datasets = {}
    for source_name in sources:
        if source_name not in SOURCE_SPECS:
            raise ValueError(f"Unsupported source dataset: {source_name}")

        spec = SOURCE_SPECS[source_name]
        dataset = load_from_disk(os.path.join(args.data_dir, source_name))
        dataset = rename_dev_split(dataset)
        dataset = dataset.filter(lambda row: spec["keep"](row["label"]))
        dataset = normalize_columns(dataset, source_name, spec["sentence1"], spec["sentence2"])
        datasets[source_name] = dataset

    merged = DatasetDict(
        {
            "train": concatenate_datasets([dataset["train"] for dataset in datasets.values() if "train" in dataset]),
            "dev": concatenate_datasets([dataset["dev"] for dataset in datasets.values() if "dev" in dataset]),
            "test": concatenate_datasets([dataset["test"] for dataset in datasets.values() if "test" in dataset]),
        }
    )
    merged.save_to_disk(os.path.join(args.data_dir, args.output))


if __name__ == "__main__":
    main()
