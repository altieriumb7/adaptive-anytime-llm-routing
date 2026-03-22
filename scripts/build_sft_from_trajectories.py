import argparse
from src.train.sft_build import build_sft_examples, save_sft_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--budgets", default="1,2,3,4", help="comma-separated budgets, e.g. 1,2,3,4")
    ap.add_argument("--max_per_uid", type=int, default=None)
    ap.add_argument("--conf_target", required=True)
    ap.add_argument("--conf_pos", required=True)
    ap.add_argument("--conf_neg", required=True)
    ap.add_argument("--calibrator_path", required=True)


    args = ap.parse_args()

    budgets = tuple(int(x.strip()) for x in args.budgets.split(",") if x.strip())
    exs = build_sft_examples(args.traj_jsonl, budgets=budgets, max_per_uid=args.max_per_uid)
    save_sft_jsonl(exs, args.out_jsonl)

    print(f"Wrote {len(exs)} SFT examples to {args.out_jsonl}")

if __name__ == "__main__":
    main()
