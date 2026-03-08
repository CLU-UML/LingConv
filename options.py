import argparse
import json
import os
from copy import deepcopy
from datetime import datetime

import numpy as np

from const import lca_names, lingfeat_names, sca_names


def str2bool(value):
    if isinstance(value, bool):
        return value

    lowered = value.lower()
    if lowered in {"yes", "true", "t", "y", "1"}:
        return True
    if lowered in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _load_saved_args(args, restore_keys):
    if not args.ckpt:
        return [args]

    ckpts = args.ckpt.split(",") if "," in args.ckpt else [args.ckpt]
    restored_args = [deepcopy(args) for _ in ckpts]

    for restored, ckpt in zip(restored_args, ckpts):
        ckpt = ckpt.rstrip("/")
        candidate_paths = [
            os.path.join(os.path.dirname(ckpt), "trainer_state.json"),
            f"{ckpt}.json",
            os.path.dirname(ckpt) + ".json",
        ]

        saved_args_path = next((path for path in candidate_paths if os.path.exists(path)), None)
        if saved_args_path and saved_args_path.endswith(".json") and "trainer_state.json" not in saved_args_path:
            with open(saved_args_path) as handle:
                restored.__dict__.update(json.load(handle))

        restored.__dict__.update(restore_keys)
        restored.ckpt = ckpt

    return restored_args


def parse_args(ckpt=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--predict_with_feedback", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--data", default="ling_conversion")
    parser.add_argument("--data_sources", default="qqp,mrpc,stsb")
    parser.add_argument("--split", default="test")
    parser.add_argument("--src_lng", default="ling")
    parser.add_argument("--lng_ids")
    parser.add_argument("--lng_ids_idx", type=int)
    parser.add_argument("--lng_ids_path", default="./indices")
    parser.add_argument("--do_imputation", action="store_true")
    parser.add_argument("--imputation_percentage", type=int, default=20)
    parser.add_argument("--imputation_seed", type=int, default=0)
    parser.add_argument("--quantize_lng", action="store_true")
    parser.add_argument("--quant_nbins", type=int, default=20)
    parser.add_argument("--use_ica", action="store_true")
    parser.add_argument("--n_ica", type=int, default=10)
    parser.add_argument("--prepend_prompt", action="store_true")
    parser.add_argument("--prompt_text", default="generate a paraphrase: ")

    parser.add_argument("--ckpt_dir", default="./checkpoints")
    parser.add_argument("--ckpt")
    parser.add_argument("--predict_fn", default="preds/predictions.txt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--grad_accumulation", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_eval_samples", type=int, default=3000)
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--model_name", default="google/flan-t5-base")
    parser.add_argument("--sem_model_path", default="google/flan-t5-base")
    parser.add_argument("--disc_type", default="deberta")
    parser.add_argument("--disc_ckpt")
    parser.add_argument("--sem_ckpt")
    parser.add_argument("--sem_loss_type", default="dedicated")
    parser.add_argument("--combine_method", default="decoder_add_first")
    parser.add_argument("--injection_type", default="first")
    parser.add_argument("--injection_layer", type=int, default=1)
    parser.add_argument("--ling_embed_type", default="one-layer")
    parser.add_argument("--combine_weight", type=float, default=1.0)
    parser.add_argument("--hidden_dim", type=int, default=500)
    parser.add_argument("--latent_dim", type=int, default=150)
    parser.add_argument("--lng_dim", type=int, default=40)
    parser.add_argument("--disc_lng_dim", type=int, default=40)
    parser.add_argument("--ling_dropout", type=float, default=0.1)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--ling2_only", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--ling_vae", action="store_true")

    parser.add_argument("--feedback_param", default="s")
    parser.add_argument("--fb_log", default="feedback_logs/default.txt")

    parser.add_argument("--pretrain_gen", action="store_true")
    parser.add_argument("--pretrain_sem", action="store_true")
    parser.add_argument("--pretrain_disc", action="store_true")
    parser.add_argument("--disc_loss", action="store_true")
    parser.add_argument("--sem_loss", action="store_true")
    parser.add_argument("--sim_loss", action="store_true")
    parser.add_argument("--use_lingpred", action="store_true")
    parser.add_argument("--process_lingpred", action="store_true")
    parser.add_argument("--aug_same", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--linggen_type", default="none")
    parser.add_argument("--linggen_input", default="s+l")
    parser.add_argument("--freeze_lm", action="store_true")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--major_arg", type=int, default=0)
    parser.add_argument("--to_restore", nargs="+", default=[])

    args, _ = parser.parse_known_args()
    args.name = f"{datetime.now().strftime('%m%d_%H-%M-%S')}-{args.data}-{args.combine_method}"

    if ckpt is not None:
        args.ckpt = ckpt

    if args.data_sources == "all":
        args.data_sources = None
    elif args.data_sources:
        args.data_sources = args.data_sources.split(",")

    restore_keys = {
        key: args.__dict__[key]
        for key in [
            "do_train",
            "do_eval",
            "do_predict",
            "predict_with_feedback",
            "eval_only",
            "predict_fn",
            "split",
            "data_dir",
            "data",
            "data_sources",
            "disc_ckpt",
            "disc_type",
            "sem_ckpt",
            "feedback_param",
            "fb_log",
            "eval_batch_size",
            "test_batch_size",
            "max_eval_samples",
            "do_imputation",
            "imputation_percentage",
            "imputation_seed",
        ]
        + args.to_restore
    }

    args_list = _load_saved_args(args, restore_keys)
    lng_names = lca_names + sca_names + lingfeat_names

    for parsed_args in args_list:
        if parsed_args.lng_ids or parsed_args.lng_ids_idx:
            if parsed_args.lng_ids_idx is not None:
                lng_ids = np.load(os.path.join(parsed_args.lng_ids_path, f"{parsed_args.lng_ids_idx}.npy"))
            elif parsed_args.lng_ids[0].isnumeric():
                lng_ids = np.array([int(x) for x in parsed_args.lng_ids.split(",")])
            elif "," in parsed_args.lng_ids:
                lng_ids = np.array([lng_names.index(x) for x in parsed_args.lng_ids.split(",")])
            else:
                lng_ids = np.load(parsed_args.lng_ids)
            parsed_args.lng_dim = len(lng_ids)
            parsed_args.lng_ids = lng_ids.tolist()
        elif parsed_args.use_ica:
            parsed_args.lng_dim = parsed_args.n_ica

        if parsed_args.disc_lng_dim is None:
            parsed_args.disc_lng_dim = parsed_args.lng_dim

    major_arg = args.major_arg
    selected_args = args_list[major_arg]

    if not selected_args.ckpt and not selected_args.eval_only:
        os.makedirs(selected_args.ckpt_dir, exist_ok=True)
        with open(os.path.join(selected_args.ckpt_dir, f"{selected_args.name}.json"), "w") as handle:
            json.dump(selected_args.__dict__, handle)

    return selected_args, args_list, lng_names
