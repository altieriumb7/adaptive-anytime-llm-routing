#!/usr/bin/env python3
import argparse

from src.train.sft_build import build_sft_examples, save_sft_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--budgets", default="1,2,3,4", help="comma-separated budgets, e.g. 1,2,3,4")
    ap.add_argument("--max_per_uid", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conf_target", default="teacher", choices=["teacher", "label", "smooth", "calibrated_teacher"])
    ap.add_argument("--conf_pos", type=float, default=0.90)
    ap.add_argument("--conf_neg", type=float, default=0.10)
    ap.add_argument("--calibrator_path", default=None)
    ap.add_argument("--allow_missing_answer", action="store_true", help="keep rows even if no parsed answer")
    args = ap.parse_args()

    budgets = tuple(int(x.strip()) for x in args.budgets.split(",") if x.strip())
    exs = build_sft_examples(
        args.traj_jsonl,
        budgets=budgets,
        keep_only_if_answer_present=not args.allow_missing_answer,
        seed=args.seed,
        max_per_uid=args.max_per_uid,
        conf_target=args.conf_target,
        conf_pos=args.conf_pos,
        conf_neg=args.conf_neg,
        calibrator_path=args.calibrator_path,
    )
    save_sft_jsonl(exs, args.out_jsonl)
    print(f"Wrote {len(exs)} SFT examples to {args.out_jsonl}")


if __name__ == "__main__":
    main()
