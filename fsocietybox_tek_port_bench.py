#!/usr/bin/env python3
"""Tek port, sırayla 5 model, Vulkan ngl 99 yalnız GPU, anlaşılır çıktı."""
import json, urllib.request, subprocess, time, sys, signal, os
from pathlib import Path
sys.path.insert(0, '/home/void0x14/Documents/echo')
from egitim.fsocietybox_olc import oku_ifeval, skor_ifeval, IFEVAL

BIN="/tmp/llama-build/bin/llama-server"
PORT=8101
MODELS=[
    ("Saf Qwen3.5-0.8B düz - üretici kartındaki model", "/home/void0x14/Documents/echo/base/gguf/Qwen3.5-0.8B-Base-Q4_K_M.gguf"),
    ("FsocietyBox düz - yalnız mimari hibrit yok", "/home/void0x14/Documents/echo/egitilmis/fsocietybox-v0-Q4_K_M.gguf"),
    ("FsocietyBox + GDN yazma kilidi beta sıfır alfa bir", "/home/void0x14/Documents/echo/egitilmis/fsocietybox-v1-Q4_K_M.gguf"),
    ("FsocietyBox + denetim kutuları talimat gövdesine yazamaz", "/home/void0x14/Documents/echo/egitilmis/fsocietybox-v2-Q4_K_M.gguf"),
    ("FsocietyBox + kafa bağımsızlığı", "/home/void0x14/Documents/echo/egitilmis/fsocietybox-v3-Q4_K_M.gguf"),
]

def start(model_path):
    cmd=[BIN,"--model",model_path,"--host","127.0.0.1","--port",str(PORT),"--ctx-size","8192","--n-gpu-layers","99","--flash-attn","on","--threads","8"]
    proc=subprocess.Popen(cmd, stdout=open(f"/tmp/tek-{PORT}.log","w"), stderr=subprocess.STDOUT)
    # bekle sağlık
    for _ in range(30):
        time.sleep(1)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if b"ok" in r.read():
                    print(f"  Sunucu hazır {model_path.split('/')[-1]}")
                    return proc
        except: pass
    raise RuntimeError("sunucu hazır değil")

def kos(url, n=50):
    inputs=oku_ifeval(IFEVAL)[:n]
    mapping={}
    for i,inp in enumerate(inputs):
        prompt=f"<|im_start|>user\n{inp.prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        body=json.dumps({"prompt":prompt,"n_predict":128,"temperature":0.6,"top_p":0.95,"top_k":20,"stop":["<|im_end|>"]}).encode()
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
    n=50
    if len(sys.argv)>1:
        n=int(sys.argv[1])
    print(f"\n=== 5 model tek port {PORT} sırayla, Vulkan ngl 99 yalnız GPU ===")
    sonuclar=[]
    for adi, path in MODELS:
        print(f"\n--- {adi} ---")
        proc=start(path)
        try:
            s=kos(f"http://127.0.0.1:{PORT}/completion", n=n)
            print(f"  Sonuç: {s['prompt_ok']}/{n} tamamen doğru = %{s['prompt_level_strict']*100:.1f}")
            sonuclar.append((adi,s))
            Path(f"/home/void0x14/Documents/echo/goals/fsocietybox/tek-{adi[:15]}.json").write_text(json.dumps(s, ensure_ascii=False, indent=2))
        finally:
            proc.terminate()
            try: proc.wait(timeout=5)
            except: proc.kill()
            time.sleep(2)
    print("\n========== YAN YANA TABLO ==========")
    print(f"{'Model':45} | Doğru/{n} | Başarı")
    print("-"*75)
    for adi,s in sonuclar:
        print(f"{adi[:45]:45} | {s['prompt_ok']:2d}/{n}    | %{s['prompt_level_strict']*100:4.1f}")
    print("-"*75)
    Path("/home/void0x14/Documents/echo/goals/fsocietybox/tek-5li-ozet.json").write_text(json.dumps([{"model":a,"ok":s["prompt_ok"],"prompt":s["prompt_level_strict"]} for a,s in sonuclar], ensure_ascii=False, indent=2))
    print("Kaydedildi: goals/fsocietybox/tek-5li-ozet.json")
