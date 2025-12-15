import argparse, os, time
from datetime import timedelta
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI


def fmt_eta(seconds: float) -> str:
    if seconds <= 0 or seconds == float("inf"):
        return "?"
    return str(timedelta(seconds=int(seconds)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_id", required=True)
    ap.add_argument("--out", default="data/batch_anytime_train_out.jsonl")
    ap.add_argument("--err", default="data/batch_anytime_train_err.jsonl")
    ap.add_argument("--every", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    client = OpenAI()

    prev_completed = None
    prev_ts = None

    while True:
        b = client.batches.retrieve(args.batch_id)
        rc = getattr(b, "request_counts", None)
        completed = getattr(rc, "completed", None) if rc else None
        failed = getattr(rc, "failed", None) if rc else None
        total = getattr(rc, "total", None) if rc else None

        now = time.time()
        rate = None
        eta = None
        if completed is not None:
            if prev_completed is not None and prev_ts is not None and now > prev_ts:
                d = completed - prev_completed
                rate = (d / (now - prev_ts)) * 60.0  # per minute
                if total is not None:
                    remaining = max(0, total - completed - (failed or 0))
                    eta = (remaining / rate) * 60.0 if rate and rate > 0 else float("inf")
            prev_completed, prev_ts = completed, now

        print(
            f"status={b.status} "
            f"completed={completed}/{total} failed={failed} "
            f"rate={rate:.2f}/min ETA={fmt_eta(eta) if eta is not None else '?'} "
            f"output_file_id={getattr(b,'output_file_id',None)} error_file_id={getattr(b,'error_file_id',None)}"
        )

        # Output/error files are meant to be fetched via Files API using output_file_id/error_file_id
        # once the batch is completed/finalizing/cancelled. :contentReference[oaicite:0]{index=0}
        if getattr(b, "error_file_id", None):
            content = client.files.content(file_id=b.error_file_id)
            with open(args.err, "wb") as f:
                f.write(content.read())
            print(f"saved err -> {args.err}")

        if getattr(b, "output_file_id", None):
            content = client.files.content(file_id=b.output_file_id)
            with open(args.out, "wb") as f:
                f.write(content.read())
            print(f"saved out -> {args.out}")
            break

        if b.status in ("failed", "expired"):
            print("Batch ended without output. Check error_file_id / dashboard.")
            break

        time.sleep(args.every)


if __name__ == "__main__":
    main()
