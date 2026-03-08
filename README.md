# LingConv

Official code release for the EMNLP 2025 paper [Linguistically-Controlled Paraphrase Generation](https://aclanthology.org/2025.findings-emnlp.1137/).

LingConv is a controllable paraphrase generation model that conditions a T5 decoder on target linguistic attributes. This release includes the paper-facing training, inference, evaluation, and dataset preprocessing code.

## Repository Layout

- `main.py`: train, validate, and generate with the core LingConv model
- `predict.py`: standalone generation, including quality-controlled inference
- `compute_metrics.py`: automatic evaluation for the paper settings
- `create_dataset.py`: build the main `ling_conversion` dataset
- `filter_ids.py`: derive the paper's 40-dimensional control vectors from full `lftk+` features
- `create_random_test.py`: build the novel-target evaluation split
- `create_imputation_sets.py`: create feature-imputation index sets for ablations
- `compute_lng.py`: optional exact linguistic feature extraction wrapper

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Download the NLTK resources required by the lexical analyzer:

```
import nltk
for resource in ["punkt", "averaged_perceptron_tagger", "wordnet"]:
    nltk.download(resource)
```

## Data

The released processed datasets are available on Hugging Face:

- [`mohdelgaar/ling_conversion`](https://huggingface.co/datasets/mohdelgaar/ling_conversion)
- [`mohdelgaar/ling_conversion_random0`](https://huggingface.co/datasets/mohdelgaar/ling_conversion_random0)

The training and evaluation code expects Hugging Face `save_to_disk` datasets under `DATA_DIR/DATASET_NAME`.

The main paper dataset is `ling_conversion` with `train`, `dev`, and `test` splits. The novel-target evaluation dataset is `ling_conversion_random0` with a `test` split containing shuffled target linguistic attributes.

Each example includes:

- `sentence1`
- `sentence2`
- `source`
- `sentence1_lftk+`
- `sentence2_lftk+`
- `sentence1_discr`
- `sentence2_discr`
- `sentence1_ling`
- `sentence2_ling`

The preprocessing scripts used to build these datasets are included in this repo and mirrored in the Hugging Face dataset repos under `preprocessing/`.

### Build The Core Dataset

```bash
python create_dataset.py --data_dir /path/to/data --output ling_conversion
```

By default, this reproduces the paper setting over `qqp`, `mrpc`, and `stsb`.

### Build The Novel-Target Split

```bash
python create_random_test.py \
  --data_dir /path/to/data \
  --input ling_conversion \
  --output ling_conversion_random0 \
  --seed 0
```

## Train LingConv

```bash
python main.py \
  --do_train \
  --data_dir /path/to/data \
  --data ling_conversion \
  --combine_method decoder_add_first \
  --ling2_only
```

Checkpoints are saved under `./checkpoints/` by default.

## Generate Predictions

Standard generation:

```bash
python predict.py \
  --ckpt ./checkpoints/<run-or-checkpoint-dir> \
  --data_dir /path/to/data \
  --data ling_conversion \
  --split test \
  --predict_fn preds/lingconv_test.txt
```

Quality-controlled generation requires the auxiliary linguistic predictor and semantic model:

```bash
python predict.py \
  --ckpt ./checkpoints/<run-or-checkpoint-dir> \
  --disc_ckpt <linguistic-predictor-checkpoint-or-hf-id> \
  --sem_ckpt <semantic-model-checkpoint-or-hf-id> \
  --predict_with_feedback \
  --data_dir /path/to/data \
  --data ling_conversion_random0 \
  --split test \
  --predict_fn preds/lingconv_qc_ood.txt
```

## Evaluate

Paper-style automatic metrics:

```bash
python compute_metrics.py \
  --predictions preds/lingconv_test.txt \
  --data_dir /path/to/data \
  --data ling_conversion
```

Novel-target evaluation:

```bash
python compute_metrics.py \
  --predictions preds/lingconv_qc_ood.txt \
  --data_dir /path/to/data \
  --data ling_conversion_random0 \
  --reference source
```

Use `--approximate` to score generations with the learned linguistic predictor instead of the exact extractor.

## Exact Feature Extraction

This repo intentionally does not version your local `lng/` directory. If you have a local copy of the legacy analyzers, `compute_lng.py` can use it for exact paper-style lexical and syntactic metrics. Otherwise, run `compute_metrics.py --approximate` instead.

## License

The repository is released under the MIT license in `LICENSE`.

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{elgaar-amiri-2025-linguistically,
    title = "Linguistically-Controlled Paraphrase Generation",
    author = "Elgaar, Mohamed  and
      Amiri, Hadi",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.1137/",
    doi = "10.18653/v1/2025.findings-emnlp.1137",
    pages = "20842--20864",
    ISBN = "979-8-89176-335-7",
}
```
