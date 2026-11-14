#!/usr/bin/env python3
"""Vulkan 5'li bench - tek komut, deterministik, hızlı. Model kartı metodolojisi birebir."""
import json, urllib.request, time, sys
from pathlib import Path
sys.path.insert(0, '/home/void0x14/Documents/echo')
from egitim.fsocietybox_olc import oku_ifeval, skor_ifeval, IFEVAL

# Kart: Qwen3.5-0.8B instruct - düşünme kapalı 52.1 / açık 44.0
# Biz düşünme açık koşuyoruz (fsocietyBox), kart sampling: thinking temp 0.6 top_p 0.95 top_k 20 presence 1.5
# Vulkan bench'in gösterdiği gibi -ngl 99 -fa on -b 512 ile 80-85 token ortalama, n_predict 128 yeterli (tg128 28.6 t/s)
# Deterministik: seed sabit, aynı 50 prompt

MODELS = [
    ("Saf Qwen3.5-0.8B düz - üretici kartındaki model", "http://127.0.0.1:8093/completion", "/home/void0x14/Documents/echo/base/gguf/Qwen3.5-0.8B-Base-Q4_K_M.gguf"),
    ("FsocietyBox düz - yalnız mimari hibrit yok", "http://127.0.0.1:8091/completion", "/home/void0x14/Documents/echo/egitilmis/fsocietybox-v0-Q4_K_M.gguf"),
    ("FsocietyBox + GDN yazma kilidi beta sıfır alfa bir", "http://127.0.0.1:8092/completion", "/home/void0x14/Documents/echo/egitilmis/fsocietybox-v1-Q4_K_M.gguf"),
    ("FsocietyBox + denetim kutuları talimat gövdesine yazamaz", "http://127.0.0.1:8094/completion", "/home/void0x14/Documents/echo/egitilmis/fsocietybox-v2-Q4_K_M.gguf"),
    ("FsocietyBox + kafa bağımsızlığı", "http://127.0.0.1:8095/completion", "/home/void0x14/Documents/echo/egitilmis/fsocietybox-v3-Q4_K_M.gguf"),
]

def kos(model_adi, url, n=50, n_predict=128):
    inputs = oku_ifeval(IFEVAL)[:n]
    mapping = {}
    t0=time.time()
    for i,inp in enumerate(inputs):
        prompt=f"<|im_start|>user\n{inp.prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        body=json.dumps({
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "repeat_penalty": 1.0,
            "presence_penalty": 1.5,
            "stop": ["<|im_end|>"]
        }).encode()
        req=urllib.request.Request(url, data=body, headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data=json.loads(resp.read().decode())
        text=data.get("content") or ""
        if "</think>" in text:
            text=text.split("</think>",1)[-1].strip()
        mapping[inp.prompt]=text
        # deterministik ilerleme dosyası
        Path(f"/tmp/bench-{n}-{model_adi[:10]}.log").write_text(f"{i+1}/50 {time.time()-t0:.1f}s\n")
        print(f"{model_adi} {i+1}/{n} {time.time()-t0:.0f}s", flush=True)
    s=skor_ifeval(mapping, inputs)
    s["model"]=model_adi
    s["url"]=url
    s["n_predict"]=n_predict
    s["sampling"]="kart düşünme açık temp0.6 top_p0.95"
    return s

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    args=p.parse_args()
    sonuclar=[]
    for adi,url,gguf in MODELS:
        print(f"\n=== {adi} ===")
        # sağlık kontrol
        try:
            urllib.request.urlopen(urllib.request.Request(url.replace("/completion","/health")), timeout=5)
        except Exception as e:
            print(f"UYARI {adi} sağlık yok {e} - atlanıyor")
            continue
        s=kos(adi, url, n=args.n)
        print(f"{adi} prompt {s['prompt_level_strict']:.3f} {s['prompt_ok']}/{s['n']} inst {s['instruction_level_strict']:.3f}")
        sonuclar.append({"model":adi, "prompt":s['prompt_level_strict'], "prompt_ok":s['prompt_ok'], "inst":s['instruction_level_strict']})
        Path(f"/home/void0x14/Documents/echo/goals/fsocietybox/bench-{adi[:20]}.json").write_text(json.dumps(s, ensure_ascii=False, indent=2))
    print("\n--- 5'li tablo ---")
    for r in sonuclar:
        print(f"{r['model'][:40]:40} prompt {r['prompt']:.3f} ({r['prompt_ok']}/50) inst {r['inst']:.3f}")
    Path("/home/void0x14/Documents/echo/goals/fsocietybox/bench-5li-ozet.json").write_text(json.dumps(sonuclar, ensure_ascii=False, indent=2))
    print("bitti: goals/fsocietybox/bench-5li-ozet.json")
