#!/usr/bin/env python3
"""fsocietyBox production. 127.0.0.1:8081 OpenAI /v1. LoRA yok. Eğitim yok."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_KOK = Path(__file__).resolve().parents[1]
_STD = Path("/home/void0x14/Documents/echo-a10g-training-standard")
for p in (_KOK, _STD):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

TABAN = os.environ.get("FSOCIETYBOX_MODEL", "Qwen/Qwen3.5-0.8B-Base")
REV = os.environ.get("FSOCIETYBOX_REV", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
PORT = int(os.environ.get("FSOCIETYBOX_PORT", "8081"))
HOST = os.environ.get("FSOCIETYBOX_HOST", "127.0.0.1")
VIDA = os.environ.get("FSOCIETYBOX_VIDA", "V0")

MODEL = None
TOK = None
LOCK = threading.Lock()
READY = False


def _fla() -> None:
    try:
        import fla.ops.gated_delta_rule  # noqa: F401
    except Exception:
        pass


def _katman_durum(model) -> dict:
    from training.fsocietybox import (
        ALTI,
        FsocietyBox,
        ICRA_KATMAN,
        _katmanlar,
        gdn_forward_sarili_mi,
    )

    layers = _katmanlar(model)
    tasir = sum(
        1
        for i in ICRA_KATMAN
        if isinstance(getattr(layers[i], "fsocietybox", None), FsocietyBox) and layers[i].fsocietybox.tasir
    )
    tasimaz = sum(
        1
        for i in ALTI
        if isinstance(getattr(layers[i], "fsocietybox", None), FsocietyBox) and not layers[i].fsocietybox.tasir
    )
    return {
        "fsocietybox": True,
        "tasir": tasir,
        "tasimaz": tasimaz,
        "gdn_sarili": gdn_forward_sarili_mi(model),
        "vida": VIDA,
        "model": TABAN,
    }


def _user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return str(m.get("content") or "")
    return ""


def _span_kur(model, tokenizer, input_ids, user_text: str) -> None:
    from egitim.fsocietybox import atama, talimat_listesi
    from training.fsocietybox import kur_atama

    import torch

    T = int(input_ids.shape[1])
    talimatlar = talimat_listesi(user_text)
    plan = atama(talimatlar)
    spans: list[torch.Tensor | None] = [None] * 18
    govde = torch.zeros(1, T, dtype=torch.bool, device=input_ids.device)
    decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    for slot in plan.kutu:
        if slot is None:
            continue
        j, tal = slot
        piece = tal.metin
        pos = decoded.find(piece)
        if pos < 0:
            continue
        acc = 0
        start = end = None
        for ti, tid in enumerate(input_ids[0].tolist()):
            tok = tokenizer.decode([tid], skip_special_tokens=False)
            nxt = acc + len(tok)
            if start is None and nxt > pos:
                start = ti
            if start is not None and acc >= pos + len(piece):
                end = ti
                break
            acc = nxt
        if start is None:
            continue
        if end is None:
            end = min(start + 1, T)
        m = torch.zeros(1, T, dtype=torch.bool, device=input_ids.device)
        m[:, start:end] = True
        spans[j - 1] = m
        govde = govde | m
    kur_atama(model, spans, govde if govde.any() else None)


def yukle() -> None:
    global MODEL, TOK, READY
    _fla()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from training.fsocietybox import yerlestir

    tok = AutoTokenizer.from_pretrained(TABAN, revision=REV)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        TABAN,
        revision=REV,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    yerlestir(model, VIDA)
    MODEL = model
    TOK = tok
    READY = True
    print("fsocietybox_ready", json.dumps(_katman_durum(model)), flush=True)


def _generate(messages: list[dict], max_tokens: int, temperature: float) -> tuple[str, str]:
    import torch

    assert MODEL is not None and TOK is not None
    text = TOK.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    enc = TOK(text, return_tensors="pt", add_special_tokens=False)
    _span_kur(MODEL, TOK, enc["input_ids"], _user_text(messages))
    gen_kw = {
        "max_new_tokens": max(1, int(max_tokens)),
        "do_sample": temperature > 0,
        "pad_token_id": TOK.pad_token_id,
        "eos_token_id": TOK.eos_token_id,
        "use_cache": False,
    }
    if temperature > 0:
        gen_kw["temperature"] = float(temperature)
    with torch.inference_mode():
        out = MODEL.generate(**enc, **gen_kw)
    new = out[0, enc["input_ids"].shape[1] :]
    raw = TOK.decode(new, skip_special_tokens=True)
    finish = "stop"
    if len(new) >= int(max_tokens):
        finish = "length"
    content = raw
    if "</think>" in raw:
        content = raw.split("</think>", 1)[-1].strip()
    return content, finish


def _kill_port(port: int) -> None:
    try:
        out = subprocess.check_output(["ss", "-tlnp"], text=True)
    except Exception:
        return
    for line in out.splitlines():
        if f":{port} " not in line and not line.endswith(f":{port}"):
            continue
        if "pid=" not in line:
            continue
        for part in line.split("pid=")[1:]:
            pid = int(part.split(",")[0].split(")")[0])
            if pid == os.getpid():
                continue
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.4)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("fsocietybox %s\n" % (fmt % args))

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] in ("/health", "/v1/health"):
            if not READY:
                self._json(503, {"fsocietybox": False, "ready": False})
                return
            d = _katman_durum(MODEL)
            d["ready"] = True
            self._json(200, d)
            return
        if self.path.startswith("/v1/models"):
            self._json(
                200,
                {
                    "object": "list",
                    "data": [{"id": "echo", "object": "model", "owned_by": "fsocietybox"}],
                },
            )
            return
        self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if not READY:
            self._json(503, {"error": "loading"})
            return
        path = self.path.split("?", 1)[0]
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            req = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._json(400, {"error": "bad_json"})
            return
        if path not in ("/v1/chat/completions", "/chat/completions", "/v1/completions"):
            self._json(404, {"error": "not_found"})
            return
        messages = req.get("messages")
        if not messages:
            prompt = req.get("prompt") or ""
            messages = [{"role": "user", "content": prompt}]
        max_tokens = int(req.get("max_tokens") or req.get("max_completion_tokens") or 256)
        temperature = float(req.get("temperature") or 0)
        stream = bool(req.get("stream"))
        try:
            with LOCK:
                content, finish = _generate(messages, max_tokens, temperature)
        except Exception as exc:
            self._json(500, {"error": type(exc).__name__, "detail": str(exc)[:800]})
            return
        cid = "fsocietybox-" + uuid.uuid4().hex[:12]
        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            chunk = {
                "id": cid,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}],
            }
            self.wfile.write(b"data: " + json.dumps(chunk, ensure_ascii=False).encode() + b"\n\n")
            end = {
                "id": cid,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            }
            self.wfile.write(b"data: " + json.dumps(end).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self._json(
            200,
            {
                "id": cid,
                "object": "chat.completion",
                "model": "echo",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": finish,
                    }
                ],
            },
        )


def main() -> None:
    yukle()
    _kill_port(PORT)
    httpd = ThreadingHTTPServer((HOST, PORT), H)
    print(f"fsocietybox_listen {HOST}:{PORT}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
