#!/usr/bin/env python3
"""Generate IFEval + namus responses with thinking open, t=0, no grammar."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_KOK = Path(__file__).resolve().parents[1]
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

from egitim.fsocietybox_olc import IFEVAL, NAMUS, PROTO, oku_ifeval

CHAT = (
    "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
)


def uret_bir(model: Path, prompt: str, n_predict: int = 512) -> tuple[str, str]:
    cmd = [
        "llama-cli",
        "-m",
        str(model),
        "-n",
        str(n_predict),
        "--temp",
        "0",
        "-no-cnv",
        "-p",
        CHAT.format(prompt=prompt),
        "--log-disable",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    raw = proc.stdout
    finish = "stop" if proc.returncode == 0 else "error"
    if "n_predict" in (proc.stderr or "") and "limit" in (proc.stderr or ""):
        finish = "length"
    # strip prompt echo if present
    marker = "<|im_start|>assistant"
    if marker in raw:
        raw = raw.split(marker, 1)[-1]
    if raw.startswith("\n<think>\n"):
        raw = raw[len("\n<think>\n") :]
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[-1]
    text = raw.replace("<|im_end|>", "").strip()
    if "<tool_call>" in text:
        finish = finish
    return text, finish


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--ifeval-out", type=Path, required=True)
    p.add_argument("--namus-out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    inputs = oku_ifeval(IFEVAL)
    if args.limit:
        inputs = inputs[: args.limit]
    mapping = {}
    rows = []
    for inp in inputs:
        text, finish = uret_bir(args.model, inp.prompt)
        mapping[inp.prompt] = text
        rows.append({"prompt": inp.prompt, "response": text, "finish": finish, "proto": PROTO})
    args.ifeval_out.parent.mkdir(parents=True, exist_ok=True)
    args.ifeval_out.write_text(json.dumps(mapping, ensure_ascii=False) + "\n", encoding="utf-8")
    with args.ifeval_out.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    namus_rows = []
    with NAMUS.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            user = rec["messages"][0]["content"]
            text, finish = uret_bir(args.model, user, n_predict=64)
            rec["response"] = text
            rec["finish"] = finish
            namus_rows.append(rec)
    with args.namus_out.open("w", encoding="utf-8") as f:
        for r in namus_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("ifeval", len(mapping), "namus", len(namus_rows), file=sys.stderr)


if __name__ == "__main__":
    main()
