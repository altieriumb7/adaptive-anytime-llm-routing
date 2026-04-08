import json, sys

def loadj(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def write_hist(metrics_json, out_hist):
    j = loadj(metrics_json)
    hist = j.get("stop_histogram", [])
    with open(out_hist, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)

def append_csv(csv_path, split, budget_tag, metrics_json):
    j = loadj(metrics_json)
    row = [
        split,
        j.get("policy",""),
        budget_tag,
        f"\"k={j.get('k','')};thr={j.get('threshold','')};m={j.get('m','')};min_step={j.get('min_step','')}\"",
        str(j.get("acc","")),
        str(j.get("mean_steps","")),
        str(j.get("p95_steps","")),
        str(j.get("mean_tokens","")),
        str(j.get("p95_tokens","")),
        str(j.get("flip_rate","")),
        str(j.get("regress_at_stop_rate","")),
    ]
    with open(csv_path, "a", encoding="utf-8") as f:
        f.write(",".join(row) + "\n")

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "write_hist":
        write_hist(sys.argv[2], sys.argv[3])
    elif cmd == "append_csv":
        append_csv(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        raise SystemExit("unknown cmd")
