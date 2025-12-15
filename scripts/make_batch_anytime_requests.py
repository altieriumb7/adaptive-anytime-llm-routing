import argparse, json
from pathlib import Path
from datasets import load_dataset

SYSTEM = (
    "You are a math solver. Produce 4 progressively improved attempts for the same problem. "
    "Each attempt must include: a short explanation, a final numeric answer, and a confidence in [0,1]. "
    "Keep explanations concise."
)

USER_TMPL = """Solve this GSM8K problem.

Problem:
{question}

Return ONLY a JSON object matching the schema.
"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "b1": {"type": "object", "additionalProperties": False,
               "properties": {"explanation": {"type": "string"},
                              "final_answer": {"type": "string"},
                              "conf": {"type": "number"}},
               "required": ["explanation", "final_answer", "conf"]},
        "b2": {"type": "object", "additionalProperties": False,
               "properties": {"explanation": {"type": "string"},
                              "final_answer": {"type": "string"},
                              "conf": {"type": "number"}},
               "required": ["explanation", "final_answer", "conf"]},
        "b3": {"type": "object", "additionalProperties": False,
               "properties": {"explanation": {"type": "string"},
                              "final_answer": {"type": "string"},
                              "conf": {"type": "number"}},
               "required": ["explanation", "final_answer", "conf"]},
        "b4": {"type": "object", "additionalProperties": False,
               "properties": {"explanation": {"type": "string"},
                              "final_answer": {"type": "string"},
                              "conf": {"type": "number"}},
               "required": ["explanation", "final_answer", "conf"]},
    },
    "required": ["b1", "b2", "b3", "b4"],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--max_output_tokens", type=int, default=700)
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    ds = load_dataset("gsm8k", "main", split=args.split)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            body = {
                "model": args.model,
                "input": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER_TMPL.format(question=ex["question"])},
                ],
                "temperature": args.temperature,
                "max_output_tokens": args.max_output_tokens,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "anytime_gsm8k_v2",
                        "strict": True,
                        "schema": SCHEMA,
                    }
                },
            }

            req = {
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    print(f"Wrote batch requests: {args.out} ({len(ds)} lines)")

if __name__ == "__main__":
    main()