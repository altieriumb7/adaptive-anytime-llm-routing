import argparse
from dotenv import load_dotenv

load_dotenv()

from src.data.load_datasets import load_gsm8k
from src.trajectory.generate_trajectories import generate_jsonl
from src.teachers.openai_teacher import OpenAITeacher


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--teacher_backend", choices=["openai", "hf"], default="openai")
    ap.add_argument("--openai_model", default="gpt-4o-mini")
    ap.add_argument("--teacher_model", default=None, help="HF model name if teacher_backend=hf")

    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--out", required=True)

    ap.add_argument("--load_in_4bit", action="store_true", help="Cheaper inference for HF teacher")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)

    args = ap.parse_args()

    if args.teacher_backend == "openai":
        teacher = OpenAITeacher(model=args.openai_model)
    else:
        if not args.teacher_model:
            raise SystemExit("--teacher_model is required when --teacher_backend=hf")
        from src.models import load_llm  # <-- IMPORT ONLY HERE
        teacher = load_llm(args.teacher_model, load_in_4bit=args.load_in_4bit)

    examples = load_gsm8k(split=args.split)
    generate_jsonl(
        teacher=teacher,
        examples=examples,
        out_path=args.out,
        resume=args.resume,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()