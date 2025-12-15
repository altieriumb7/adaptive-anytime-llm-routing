import argparse
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_id", required=True)
    ap.add_argument("--out", default="data/batch_out.jsonl")
    ap.add_argument("--err", default="data/batch_err.jsonl")
    args = ap.parse_args()

    client = OpenAI()
    b = client.batches.retrieve(args.batch_id)
    print("status:", b.status)
    print("request_counts:", getattr(b, "request_counts", None))
    print("output_file_id:", getattr(b, "output_file_id", None))
    print("error_file_id:", getattr(b, "error_file_id", None))

    # Download error file ASAP (even if in_progress)
    if getattr(b, "error_file_id", None):
        content = client.files.content(file_id=b.error_file_id)
        with open(args.err, "wb") as f:
            f.write(content.read())
        print("Saved errors:", args.err)

    # Download output if available (usually only when completed/cancelled/expired)
    if getattr(b, "output_file_id", None):
        content = client.files.content(file_id=b.output_file_id)
        with open(args.out, "wb") as f:
            f.write(content.read())
        print("Saved output:", args.out)

    if b.status != "completed":
        print("Not completed yet.")
        return


if __name__ == "__main__":
    main()
