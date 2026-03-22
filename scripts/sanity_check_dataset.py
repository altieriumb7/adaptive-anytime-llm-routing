#!/usr/bin/env python3
"""Sanity checks for JSONL datasets used in this repo.

This script is meant to be run BEFORE training/eval to catch:
  - JSON parse errors
  - missing fields / missing budgets
  - non-parseable answers / missing confidence
  - confidence out of range
  - incorrect stored 'correct' flags (optional recompute)

It auto-detects three common schemas in this repo:
  1) Trajectories (anytime dataset)
     {uid, problem, gold, checkpoints:[{t, raw, answer?, conf?, correct?}, ...]}
  2) SFT rows
     {uid, budget_t, messages, response, gold, ...}
  3) Router rows (one row per uid,t)
     {uid, t, raw_text, answer, conf, correct, ...}

Examples
  python scripts/sanity_check_dataset.py --path data/anytime_gsm8k_train_v2.jsonl
  python scripts/sanity_check_dataset.py --path data/sft_gsm8k_anytime_v2.jsonl
  python scripts/sanity_check_dataset.py --path data/router_splits/dev.jsonl --expected_ts 1,2,3,4 --check_correct

Exit code
  - 0 if OK
  - 1 if issues were found and --strict was provided
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Allow `python scripts/...` from repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.utils.parsing import parse_answer_and_conf


# NOTE: We intentionally implement a tiny correctness checker here instead of
# importing `src.data.judging`, because `src.data.load_datasets` imports
# HuggingFace `datasets`, which might not be installed in lightweight envs.
def _normalize_answer(ans: str) -> str:
    ans = ans.strip()
    ans = ans.replace(",", "")
    ans = ans.rstrip(".")
    return ans


def is_correct(pred: Optional[str], gold: str) -> bool:
    """A minimal GSM8K-style judge (exact after normalization + numeric equality)."""
    if pred is None:
        return False
    p = _normalize_answer(str(pred))
    g = _normalize_answer(str(gold))
    if p == g:
        return True
    # numeric match tolerant to "42.0" vs "42"
    try:
        return float(p) == float(g)
    except Exception:
        return False


@dataclass
class Stats:
    n_rows: int = 0
    n_json_errors: int = 0
    n_missing_required: int = 0
    n_missing_answer: int = 0
    n_missing_conf: int = 0
    n_conf_oob: int = 0
    n_nonparseable: int = 0
    n_correct_mismatch: int = 0
    n_missing_expected_ts: int = 0
    n_duplicate_uids: int = 0
    n_duplicate_uid_t: int = 0


def iter_jsonl(path: str, *, max_lines: Optional[int] = None) -> Iterable[Tuple[int, str]]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if max_lines is not None and i > max_lines:
                return
            line = line.rstrip("\n")
            if not line.strip():
                continue
            yield i, line


def detect_schema(obj: Dict[str, Any]) -> str:
    if isinstance(obj, dict) and "checkpoints" in obj:
        return "trajectory"
    if isinstance(obj, dict) and "messages" in obj and "response" in obj and "budget_t" in obj:
        return "sft"
    if isinstance(obj, dict) and "raw_text" in obj and "t" in obj and "uid" in obj:
        return "router_rows"
    return "unknown"


def _as_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _print_examples(title: str, examples: List[Tuple[str, str]], limit: int = 5) -> None:
    if not examples:
        return
    print(f"\n{title} (showing up to {limit})")
    for uid, msg in examples[:limit]:
        print(f"- {uid}: {msg}")


def check_trajectory_rows(
    rows: List[Dict[str, Any]],
    *,
    expected_ts: Optional[Sequence[int]] = None,
    check_correct_flag: bool = False,
) -> Tuple[Stats, Dict[str, Any]]:
    st = Stats(n_rows=len(rows))
    required_top = {"uid", "problem", "gold", "checkpoints"}

    uid_seen: Set[str] = set()
    dup_uids = 0
    missing_ts_uids: List[Tuple[str, str]] = []
    nonparseable_uids: List[Tuple[str, str]] = []
    mismatch_uids: List[Tuple[str, str]] = []

    all_ts: Counter[int] = Counter()
    mode_ctr: Counter[str] = Counter()
    per_t_parseable: Counter[int] = Counter()
    per_t_total: Counter[int] = Counter()
    per_t_missing_conf: Counter[int] = Counter()
    per_t_conf_oob: Counter[int] = Counter()
    per_t_correct_mismatch: Counter[int] = Counter()

    for obj in rows:
        uid = str(obj.get("uid", "<missing_uid>"))
        if uid in uid_seen:
            dup_uids += 1
        uid_seen.add(uid)

        if not required_top.issubset(obj.keys()):
            st.n_missing_required += 1
            continue

        cps = obj.get("checkpoints")
        if not isinstance(cps, list) or not cps:
            st.n_missing_required += 1
            continue

        ts_here: Set[int] = set()
        for cp in cps:
            if not isinstance(cp, dict):
                st.n_missing_required += 1
                continue
            t = cp.get("t")
            if not isinstance(t, int):
                try:
                    t = int(t)
                except Exception:
                    st.n_missing_required += 1
                    continue
            ts_here.add(t)
            all_ts[t] += 1
            mode_ctr[str(cp.get("mode", ""))] += 1
            per_t_total[t] += 1

            raw = cp.get("raw") or ""
            ans = cp.get("answer")
            conf = _as_float(cp.get("conf"))
            if ans is None or conf is None:
                a2, c2 = parse_answer_and_conf(raw)
                ans = ans if ans is not None else a2
                conf = conf if conf is not None else c2

            if ans is None:
                st.n_missing_answer += 1
            if conf is None:
                st.n_missing_conf += 1
                per_t_missing_conf[t] += 1
            else:
                if not (0.0 <= conf <= 1.0):
                    st.n_conf_oob += 1
                    per_t_conf_oob[t] += 1

            parseable = (ans is not None)
            if parseable:
                per_t_parseable[t] += 1
            else:
                st.n_nonparseable += 1
                nonparseable_uids.append((uid, f"t={t} non-parseable answer"))

            if check_correct_flag:
                gold = str(obj.get("gold", ""))
                stored = cp.get("correct")
                stored_int = None
                if stored is not None:
                    stored_int = int(bool(stored)) if isinstance(stored, bool) else int(stored)
                recomputed = int(is_correct(ans, gold))
                if stored_int is not None and stored_int != recomputed:
                    st.n_correct_mismatch += 1
                    per_t_correct_mismatch[t] += 1
                    mismatch_uids.append((uid, f"t={t} stored_correct={stored_int} recomputed={recomputed}"))

        if expected_ts is not None:
            exp = set(expected_ts)
            missing = sorted(exp - ts_here)
            if missing:
                st.n_missing_expected_ts += 1
                missing_ts_uids.append((uid, f"missing t={missing}"))

    st.n_duplicate_uids = dup_uids

    inferred_ts = sorted(all_ts.keys())
    report = {
        "schema": "trajectory",
        "n_rows": st.n_rows,
        "n_unique_uids": len(uid_seen),
        "inferred_ts": inferred_ts,
        "modes": dict(mode_ctr),
        "per_t": {
            str(t): {
                "count": per_t_total[t],
                "parseable_pct": (per_t_parseable[t] / per_t_total[t]) if per_t_total[t] else 0.0,
                "missing_conf": per_t_missing_conf[t],
                "conf_oob": per_t_conf_oob[t],
                "correct_mismatch": per_t_correct_mismatch[t],
            }
            for t in inferred_ts
        },
    }

    _print_examples("Examples with missing expected budgets", missing_ts_uids)
    _print_examples("Examples with non-parseable answers", nonparseable_uids)
    if check_correct_flag:
        _print_examples("Examples with correct-flag mismatches", mismatch_uids)

    return st, report


def check_sft_rows(
    rows: List[Dict[str, Any]],
    *,
    expected_ts: Optional[Sequence[int]] = None,
    check_correct_flag: bool = False,
) -> Tuple[Stats, Dict[str, Any]]:
    st = Stats(n_rows=len(rows))
    required_top = {"uid", "budget_t", "messages", "response", "gold"}

    uid_to_ts: Dict[str, Set[int]] = defaultdict(set)
    per_t_total: Counter[int] = Counter()
    per_t_parseable: Counter[int] = Counter()
    per_t_missing_conf: Counter[int] = Counter()
    per_t_conf_oob: Counter[int] = Counter()
    per_t_correct: Counter[int] = Counter()

    issues: List[Tuple[str, str]] = []

    for obj in rows:
        uid = str(obj.get("uid", "<missing_uid>"))
        if not required_top.issubset(obj.keys()):
            st.n_missing_required += 1
            issues.append((uid, "missing required top-level fields"))
            continue

        try:
            t = int(obj.get("budget_t"))
        except Exception:
            st.n_missing_required += 1
            issues.append((uid, "budget_t is not an int"))
            continue

        uid_to_ts[uid].add(t)
        per_t_total[t] += 1

        resp = obj.get("response") or ""
        ans, conf = parse_answer_and_conf(resp)
        if ans is None:
            st.n_nonparseable += 1
            issues.append((uid, f"t={t} non-parseable response"))
        else:
            per_t_parseable[t] += 1

        if conf is None:
            st.n_missing_conf += 1
            per_t_missing_conf[t] += 1
        else:
            if not (0.0 <= conf <= 1.0):
                st.n_conf_oob += 1
                per_t_conf_oob[t] += 1

        if check_correct_flag:
            gold = str(obj.get("gold", ""))
            ok = int(is_correct(ans, gold))
            per_t_correct[t] += ok

        msgs = obj.get("messages")
        if not isinstance(msgs, list) or not msgs:
            st.n_missing_required += 1
            issues.append((uid, f"t={t} messages is empty/invalid"))
        else:
            for m in msgs:
                if not isinstance(m, dict) or "role" not in m or "content" not in m:
                    st.n_missing_required += 1
                    issues.append((uid, f"t={t} malformed message"))
                    break

    if expected_ts is not None:
        exp = set(expected_ts)
        for uid, ts in uid_to_ts.items():
            missing = sorted(exp - ts)
            if missing:
                st.n_missing_expected_ts += 1

    inferred_ts = sorted({t for ts in uid_to_ts.values() for t in ts})
    report = {
        "schema": "sft",
        "n_rows": st.n_rows,
        "n_unique_uids": len(uid_to_ts),
        "inferred_ts": inferred_ts,
        "per_t": {
            str(t): {
                "count": per_t_total[t],
                "parseable_pct": (per_t_parseable[t] / per_t_total[t]) if per_t_total[t] else 0.0,
                "missing_conf": per_t_missing_conf[t],
                "conf_oob": per_t_conf_oob[t],
                "acc": (per_t_correct[t] / per_t_total[t]) if (check_correct_flag and per_t_total[t]) else None,
            }
            for t in inferred_ts
        },
    }

    _print_examples("SFT issues", issues)
    return st, report


def check_router_rows(
    rows: List[Dict[str, Any]],
    *,
    expected_ts: Optional[Sequence[int]] = None,
    check_correct_flag: bool = False,
) -> Tuple[Stats, Dict[str, Any]]:
    st = Stats(n_rows=len(rows))
    required_top = {"uid", "t", "raw_text", "gold"}

    seen_uid_t: Set[Tuple[str, int]] = set()
    uid_to_ts: Dict[str, Set[int]] = defaultdict(set)
    per_t_total: Counter[int] = Counter()
    per_t_parseable: Counter[int] = Counter()
    per_t_missing_conf: Counter[int] = Counter()
    per_t_conf_oob: Counter[int] = Counter()
    per_t_correct_mismatch: Counter[int] = Counter()
    issues: List[Tuple[str, str]] = []

    for obj in rows:
        uid = str(obj.get("uid", "<missing_uid>"))
        if not required_top.issubset(obj.keys()):
            st.n_missing_required += 1
            issues.append((uid, "missing required top-level fields"))
            continue

        try:
            t = int(obj.get("t"))
        except Exception:
            st.n_missing_required += 1
            issues.append((uid, "t is not an int"))
            continue

        key = (uid, t)
        if key in seen_uid_t:
            st.n_duplicate_uid_t += 1
        seen_uid_t.add(key)

        uid_to_ts[uid].add(t)
        per_t_total[t] += 1

        raw = obj.get("raw_text") or ""
        ans = obj.get("answer")
        conf = _as_float(obj.get("conf"))

        if ans is None or conf is None:
            a2, c2 = parse_answer_and_conf(raw)
            ans = ans if ans is not None else a2
            conf = conf if conf is not None else c2

        if ans is None:
            st.n_nonparseable += 1
            issues.append((uid, f"t={t} non-parseable answer"))
        else:
            per_t_parseable[t] += 1

        if conf is None:
            st.n_missing_conf += 1
            per_t_missing_conf[t] += 1
        else:
            if not (0.0 <= conf <= 1.0):
                st.n_conf_oob += 1
                per_t_conf_oob[t] += 1

        if check_correct_flag:
            gold = str(obj.get("gold", ""))
            recomputed = int(is_correct(ans, gold))
            stored = obj.get("correct")
            stored_int = None
            if stored is not None:
                stored_int = int(bool(stored)) if isinstance(stored, bool) else int(stored)
            if stored_int is not None and stored_int != recomputed:
                st.n_correct_mismatch += 1
                per_t_correct_mismatch[t] += 1
                issues.append((uid, f"t={t} stored_correct={stored_int} recomputed={recomputed}"))

    if expected_ts is not None:
        exp = set(expected_ts)
        for uid, ts in uid_to_ts.items():
            missing = sorted(exp - ts)
            if missing:
                st.n_missing_expected_ts += 1

    inferred_ts = sorted({t for ts in uid_to_ts.values() for t in ts})
    report = {
        "schema": "router_rows",
        "n_rows": st.n_rows,
        "n_unique_uids": len(uid_to_ts),
        "inferred_ts": inferred_ts,
        "per_t": {
            str(t): {
                "count": per_t_total[t],
                "parseable_pct": (per_t_parseable[t] / per_t_total[t]) if per_t_total[t] else 0.0,
                "missing_conf": per_t_missing_conf[t],
                "conf_oob": per_t_conf_oob[t],
                "correct_mismatch": per_t_correct_mismatch[t],
            }
            for t in inferred_ts
        },
        "duplicates_uid_t": st.n_duplicate_uid_t,
    }

    _print_examples("Router-row issues", issues)
    return st, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to a JSONL file")
    ap.add_argument("--max_lines", type=int, default=None, help="Read at most N non-empty lines")
    ap.add_argument(
        "--expected_ts",
        type=str,
        default=None,
        help="Comma-separated expected budgets (e.g., '1,2,3,4'). If omitted, only inferred budgets are reported.",
    )
    ap.add_argument(
        "--check_correct",
        action="store_true",
        help="Recompute correctness from (answer,gold) and compare to stored correct (when present).",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any issues are detected.",
    )
    ap.add_argument(
        "--save_report",
        type=str,
        default=None,
        help="Optional path to save a JSON summary report.",
    )
    args = ap.parse_args()

    expected_ts = None
    if args.expected_ts:
        expected_ts = [int(x.strip()) for x in args.expected_ts.split(",") if x.strip()]
        if not expected_ts:
            expected_ts = None

    rows: List[Dict[str, Any]] = []
    json_errors: List[Tuple[int, str]] = []
    for ln, line in iter_jsonl(args.path, max_lines=args.max_lines):
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                json_errors.append((ln, "top-level JSON is not an object"))
        except Exception as e:
            json_errors.append((ln, str(e)))

    if not rows and json_errors:
        print(f"ERROR: could not parse any rows from {args.path}")
        for ln, err in json_errors[:10]:
            print(f"  line {ln}: {err}")
        sys.exit(1)

    schema = detect_schema(rows[0])
    if schema == "unknown":
        print("ERROR: could not auto-detect schema from first row keys:")
        print(sorted(rows[0].keys()))
        sys.exit(1)

    if schema == "trajectory":
        st, report = check_trajectory_rows(rows, expected_ts=expected_ts, check_correct_flag=args.check_correct)
    elif schema == "sft":
        st, report = check_sft_rows(rows, expected_ts=expected_ts, check_correct_flag=args.check_correct)
    elif schema == "router_rows":
        st, report = check_router_rows(rows, expected_ts=expected_ts, check_correct_flag=args.check_correct)
    else:
        raise RuntimeError("unreachable")

    st.n_json_errors = len(json_errors)
    if json_errors:
        print(f"\nJSON parse errors: {len(json_errors)} (showing up to 10)")
        for ln, err in json_errors[:10]:
            print(f"  line {ln}: {err}")

    print("\n=== Sanity Check Summary ===")
    print(f"Path: {args.path}")
    print(f"Schema: {schema}")
    print(f"Rows read: {st.n_rows}")
    print(f"Unique uids: {report.get('n_unique_uids')}")
    print(f"Inferred budgets (t): {report.get('inferred_ts')}")
    if schema == "trajectory" and report.get("modes"):
        print(f"Modes: {report.get('modes')}")

    print("\nIssues")
    print(f"- JSON errors: {st.n_json_errors}")
    print(f"- Missing required fields: {st.n_missing_required}")
    print(f"- Non-parseable answers: {st.n_nonparseable}")
    print(f"- Missing answer: {st.n_missing_answer}")
    print(f"- Missing conf: {st.n_missing_conf}")
    print(f"- Conf out-of-range: {st.n_conf_oob}")
    if args.check_correct:
        print(f"- Correct mismatches: {st.n_correct_mismatch}")
    print(f"- Missing expected budgets (per uid): {st.n_missing_expected_ts}")
    print(f"- Duplicate uids: {st.n_duplicate_uids}")
    print(f"- Duplicate (uid,t): {st.n_duplicate_uid_t}")

    if args.save_report:
        os.makedirs(os.path.dirname(args.save_report) or ".", exist_ok=True)
        payload = {"stats": st.__dict__, "report": report}
        with open(args.save_report, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved report to: {args.save_report}")

    total_issues = (
        st.n_json_errors
        + st.n_missing_required
        + st.n_nonparseable
        + st.n_missing_answer
        + st.n_missing_conf
        + st.n_conf_oob
        + st.n_correct_mismatch
        + st.n_missing_expected_ts
        + st.n_duplicate_uids
        + st.n_duplicate_uid_t
    )
    if args.strict and total_issues > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
