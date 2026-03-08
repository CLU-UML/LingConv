import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer, set_seed

from data import LingDataCollator, load_data
from model import get_model
from options import parse_args


def move_to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def ensure_parent_dir(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def write_feedback_log(log_path, feedback_traces):
    ensure_parent_dir(log_path)
    with open(log_path, "w") as handle:
        for idx, trace in enumerate(feedback_traces):
            handle.write(f"Example {idx}\n")
            handle.write(f"Final: {trace['final']}\n")
            for step, interpolation in enumerate(trace["interpolations"]):
                handle.write(f"  [{step}] {interpolation}\n")
            handle.write("\n")


def write_predictions(path, predictions):
    ensure_parent_dir(path)
    with open(path, "w") as handle:
        handle.write("\n".join(predictions))
        handle.write("\n")


def build_feedback_batch(batch):
    feedback_batch = {
        "sentence1_input_ids": batch["input_ids"],
        "sentence1_attention_mask": batch["attention_mask"],
        "sentence2_ling": batch["sentence2_ling"],
    }
    if "sentence1_ling" in batch:
        feedback_batch["sentence1_ling"] = batch["sentence1_ling"]
    return feedback_batch


def main():
    args, _, _ = parse_args()
    if not args.ckpt:
        raise ValueError("--ckpt is required for prediction.")

    set_seed(args.seed)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    data, _, _ = load_data(args, tokenizer, return_data=True)
    if args.split not in data:
        raise ValueError(f"Split {args.split!r} not found in dataset.")

    if args.predict_with_feedback and (not args.disc_ckpt or not args.sem_ckpt):
        raise ValueError("Quality-controlled prediction requires both --disc_ckpt and --sem_ckpt.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ling_disc, sem_emb = get_model(args, tokenizer, device)
    model.eval()
    if ling_disc is not None:
        ling_disc.eval()
    if sem_emb is not None:
        sem_emb.eval()

    collator = LingDataCollator(tokenizer)
    batch_size = 1 if args.predict_with_feedback else args.eval_batch_size
    dataloader = DataLoader(data[args.split], batch_size=batch_size, shuffle=False, collate_fn=collator)

    predictions = []
    feedback_traces = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        batch = move_to_device(batch, device)
        if args.predict_with_feedback:
            prediction_ids, feedback_trace = model.infer_with_feedback_BP(
                ling_disc=ling_disc,
                sem_emb=sem_emb,
                batch=build_feedback_batch(batch),
                tokenizer=tokenizer,
            )
            decoded = tokenizer.batch_decode(prediction_ids.cpu(), skip_special_tokens=True)
            predictions.extend(text.strip() for text in decoded)
            feedback_traces.append(
                {
                    "final": feedback_trace[0],
                    "interpolations": feedback_trace[1],
                }
            )
        else:
            with torch.no_grad():
                prediction_ids = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    sentence1_ling=batch.get("sentence1_ling"),
                    sentence2_ling=batch["sentence2_ling"],
                )
                decoded = tokenizer.batch_decode(prediction_ids.cpu(), skip_special_tokens=True)
                predictions.extend(text.strip() for text in decoded)

    write_predictions(args.predict_fn, predictions)
    if args.predict_with_feedback:
        write_feedback_log(args.fb_log, feedback_traces)

    print(f"Saved predictions to {args.predict_fn}")
    if args.predict_with_feedback:
        print(f"Saved feedback traces to {args.fb_log}")


if __name__ == "__main__":
    main()
