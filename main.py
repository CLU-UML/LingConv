import os

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, T5Tokenizer, set_seed

from data import LingDataCollator, load_data
from model import get_model
from options import parse_args


def resolve_output_dir(args):
    if args.ckpt and os.path.isdir(args.ckpt):
        if os.path.basename(args.ckpt).startswith("checkpoint-"):
            return os.path.dirname(args.ckpt)
        return args.ckpt
    return os.path.join(args.ckpt_dir, args.name)


def save_predictions(tokenizer, predictions, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    decoded = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded = [text.strip() for text in decoded]
    with open(output_path, "w") as handle:
        handle.write("\n".join(decoded))
        handle.write("\n")
    return decoded


def main():
    args, _, _ = parse_args()
    if not any([args.do_train, args.do_eval, args.do_predict]):
        args.do_train = True

    set_seed(args.seed)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    data, _, _ = load_data(args, tokenizer)

    output_dir = resolve_output_dir(args)
    os.makedirs(output_dir, exist_ok=True)

    has_eval = "dev" in data
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        predict_with_generate=True,
        generation_max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        logging_steps=50,
        evaluation_strategy="steps" if args.do_train and has_eval else "no",
        eval_steps=500 if args.do_train and has_eval else None,
        save_strategy="steps" if args.do_train and has_eval else ("epoch" if args.do_train else "no"),
        save_steps=500 if args.do_train and has_eval else None,
        save_total_limit=2,
        load_best_model_at_end=bool(args.do_train and has_eval),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
    )

    model, _, _ = get_model(args, tokenizer, training_args.device)
    collator = LingDataCollator(tokenizer)
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collator,
        train_dataset=data.get("train"),
        eval_dataset=data.get("dev"),
    )

    if args.do_train:
        trainer.train(resume_from_checkpoint=args.ckpt if args.ckpt else None)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

    if args.do_eval:
        if "dev" not in data:
            raise ValueError("No dev split found for evaluation.")
        dev_metrics = trainer.evaluate(eval_dataset=data["dev"], metric_key_prefix="dev")
        print(dev_metrics)
        if "dev_ood" in data:
            dev_ood_metrics = trainer.evaluate(eval_dataset=data["dev_ood"], metric_key_prefix="dev_ood")
            print(dev_ood_metrics)

    if args.do_predict:
        if args.split not in data:
            raise ValueError(f"Split {args.split!r} not found in dataset.")
        predictions = trainer.predict(data[args.split], metric_key_prefix=args.split).predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        save_predictions(tokenizer, predictions, args.predict_fn)
        print(f"Saved predictions to {args.predict_fn}")


if __name__ == "__main__":
    main()
