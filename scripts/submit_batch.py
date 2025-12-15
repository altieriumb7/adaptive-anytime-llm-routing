import argparse
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--endpoint", default="/v1/responses")
    ap.add_argument("--window", default="24h")
    args = ap.parse_args()

    client = OpenAI()

    with open(args.infile, "rb") as f:
        up = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=up.id,
        endpoint=args.endpoint,
        completion_window=args.window,
    )

    print("input_file_id:", up.id)
    print("batch_id:", batch.id)
    print("status:", batch.status)

if __name__ == "__main__":
    main()
