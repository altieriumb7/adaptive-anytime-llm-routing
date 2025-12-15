import argparse, json
from datasets import load_dataset
from pathlib import Path

def extract_output_text(resp_body: dict) -> str:
    # Responses payload contains output[] with content[] (type: "output_text") :contentReference[oaicite:3]{index=3}
    out = []
    for item in resp_body.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    out.append(c.get("text", ""))
    return "".join(out).strip()

def make_raw(step: dict) -> str:
    return f"{step['explanation']}\n#### {step['final_answer']}\nCONF: {step['conf']}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--batch_out", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ds = load_dataset("gsm8k", "main", split=args.split)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(args.batch_out, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            cid = int(obj["custom_id"])

            resp = obj.get("response")
            if not resp or resp.get("status_code") != 200:
                continue

            body = resp["body"]
            text = extract_output_text(body)
            data = json.loads(text)

            ex = ds[cid]
            rec = {
                "id": cid,
                "question": ex["question"],
                "answer": ex["answer"],
                "checkpoints": [
                    {"budget": 1, "raw": make_raw(data["b1"])},
                    {"budget": 2, "raw": make_raw(data["b2"])},
                    {"budget": 3, "raw": make_raw(data["b3"])},
                    {"budget": 4, "raw": make_raw(data["b4"])},
                ],
                "meta": {"teacher": "gpt-4o-mini", "format": "batch_structured_anytime_v2"},
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    print("Wrote:", args.out, "lines:", written)

if __name__ == "__main__":
    main()
