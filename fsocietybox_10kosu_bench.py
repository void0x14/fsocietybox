#!/usr/bin/env python3
"""10 koşu: 5 model x 2 saha, instruct, Vulkan tek port ngl 99, anlaşılır"""
import json, urllib.request, subprocess, time, sys
from pathlib import Path
sys.path.insert(0, '/home/void0x14/Documents/echo')
from egitim.fsocietybox_olc import oku_ifeval, skor_ifeval, IFEVAL

BIN="/tmp/llama-build/bin/llama-server"
PORT=8101
# Instruct model karttaki - base değil
INSTRUCT="/home/void0x14/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/Qwen3.5-0.8B-Q4_K_M.gguf"
FS0="/home/void0x14/Documents/echo/egitilmis/fsocietybox-v0-Q4_K_M.gguf"
FS1="/home/void0x14/Documents/echo/egitilmis/fsocietybox-v1-Q4_K_M.gguf"
FS2="/home/void0x14/Documents/echo/egitilmis/fsocietybox-v2-Q4_K_M.gguf"
FS3="/home/void0x14/Documents/echo/egitilmis/fsocietybox-v3-Q4_K_M.gguf"

MODELS=[
    ("Saf Qwen3.5-0.8B düz - üretici kartındaki instruct model", INSTRUCT),
    ("FsocietyBox düz - yalnız mimari hibrit yok", FS0),
    ("FsocietyBox + GDN yazma kilidi beta sıfır alfa bir", FS1),
    ("FsocietyBox + denetim kutuları talimat gövdesine yazamaz", FS2),
    ("FsocietyBox + kafa bağımsızlığı", FS3),
]

def start(model_path):
    cmd=[BIN,"--model",model_path,"--host","127.0.0.1","--port",str(PORT),"--ctx-size","8192","--n-gpu-layers","99","--flash-attn","on","--threads","8"]
    proc=subprocess.Popen(cmd, stdout=open(f"/tmp/tek-{PORT}.log","w"), stderr=subprocess.STDOUT)
    for _ in range(30):
        time.sleep(1)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if b"ok" in r.read():
                    return proc
        except: pass
    raise RuntimeError("sunucu hazır değil")

def kos(url, n):
    inputs=oku_ifeval(IFEVAL)[:n]
    mapping={}
    for i,inp in enumerate(inputs):
        prompt=f"<|im_start|>user\n{inp.prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        body=json.dumps({"prompt":prompt,"n_predict":128,"temperature":0.0,"top_p":0.95,"top_k":20,"stop":["<|im_end|>"]}).encode()
        req=urllib.request.Request(url, data=body, headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data=json.loads(resp.read().decode())
        text=data.get("content") or ""
        if "</think>" in text:
            text=text.split("</think>",1)[-1].strip()
        mapping[inp.prompt]=text
        if (i+1)%10==0:
            print(f"  {i+1}/{n} bitti")
    return skor_ifeval(mapping, inputs)

if __name__=="__main__":
    print("=== 10 koşu başlıyor ===")
    print("Saha 1: bizim 50")
    print("Saha 2: üretici kartındaki set - aynı metodoloji birebir")
    print("")
    tum=[]
    # Saha 1: bizim 50
    print(">>> SAHA 1: bizim 50 <<<")
    for adi, path in MODELS:
        print(f"\n--- {adi} ---")
        proc=start(path)
        try:
            s=kos(f"http://127.0.0.1:{PORT}/completion", n=50)
            print(f"  Sonuç: {s['prompt_ok']}/50 = %{s['prompt_level_strict']*100:.1f}")
            tum.append((f"Saha1-50 {adi}", s))
        finally:
            proc.terminate()
            try: proc.wait(timeout=5)
            except: proc.kill()
            time.sleep(2)
    # Saha 2: üretici seti 541 (karttaki)
    print("\n>>> SAHA 2: üretici kartındaki set (541) <<<")
    for adi, path in MODELS:
        print(f"\n--- {adi} ---")
        proc=start(path)
        try:
            s=kos(f"http://127.0.0.1:{PORT}/completion", n=541)
            print(f"  Sonuç: {s['prompt_ok']}/541 = %{s['prompt_level_strict']*100:.1f}")
            tum.append((f"Saha2-541 {adi}", s))
        finally:
            proc.terminate()
            try: proc.wait(timeout=5)
            except: proc.kill()
            time.sleep(2)
    print("\n========== 10 KOŞU TABLO ==========")
    for adi,s in tum:
        n=s['n']
        print(f"{adi[:55]:55} | {s['prompt_ok']:3d}/{n} | %{s['prompt_level_strict']*100:5.1f}")
    Path("/home/void0x14/Documents/echo/goals/fsocietybox/10kosu-ozet.json").write_text(json.dumps([{"model":a,"n":s["n"],"ok":s["prompt_ok"],"prompt":s["prompt_level_strict"]} for a,s in tum], ensure_ascii=False, indent=2))
    print("Kaydedildi: goals/fsocietybox/10kosu-ozet.json")
