import argparse, json, os
from typing import Any, Dict, Optional

from datasets import load_dataset
from src.data.load_datasets import normalize_answer
from src.data.judging import is_correct


def gsm8k_gold_from_answer(answer_text: str) -> str:
    # gsm8k answers usually include "#### <final>"
    import re
    m = re.search(r"####\s*(.+?)\s*$", answer_text, flags=re.MULTILINE)
    return normalize_answer(m.group(1) if m else answer_text)


def extract_output_text(resp_body: Dict[str, Any]) -> str:
    parts = []
    for item in resp_body.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    parts.append(c.get("text", ""))
    return "".join(parts).strip()


def mk_raw(expl: str, ans: str, conf: float) -> str:
    return f"{expl.strip()}\n#### {str(ans).strip()}\nCONF: {float(conf):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--batch_out", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", default="gpt-4o-mini")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ds = load_dataset("gsm8k", "main", split=args.split)

    ok = 0
    bad = 0
    with open(args.batch_out, "r", encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            cid = int(obj["custom_id"])
            resp = obj.get("response", {})
            if resp.get("status_code") != 200:
                bad += 1
                continue

            body = resp["body"]
            out_text = extract_output_text(body)
            data = json.loads(out_text)  # {"b1":..., "b2":..., "b3":..., "b4":...}

            q = ds[cid]["question"]
            gold = gsm8k_gold_from_answer(ds[cid]["answer"])
            uid = f"gsm8k_{args.split}_{cid}"

            cps = []
            for t in (1, 2, 3, 4):
                b = data[f"b{t}"]
                ans = str(b["final_answer"])
                conf = float(b["conf"])
                raw = mk_raw(b["explanation"], ans, conf)
                cps.append({
                    "t": t,
                    "mode": {1:"draft", 2:"resolve", 3:"verify_text", 4:"repair"}[t],
                    "raw": raw,
                    "answer": ans,
                    "conf": conf,
                    "correct": bool(is_correct(ans, gold)),
                })

            rec = {
                "uid": uid,
                "problem": q,
                "gold": gold,
                "meta": {"dataset": "gsm8k", "split": args.split, "teacher": args.teacher, "source": "openai_batch"},
                "checkpoints": cps,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ok += 1

    print(f"Wrote {ok} trajectories to {args.out}. Skipped {bad} non-200 lines.")


if __name__ == "__main__":
    main()
