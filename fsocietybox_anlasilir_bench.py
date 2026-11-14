#!/usr/bin/env python3
"""Anlaşılır bench - 10 yaşındaki gibi, teknik log yok, sadece sonuç."""
import json, urllib.request, time, sys
from pathlib import Path
sys.path.insert(0, '/home/void0x14/Documents/echo')
from egitim.fsocietybox_olc import oku_ifeval, skor_ifeval, IFEVAL

MODELS = [
    ("Saf Qwen3.5-0.8B düz - üretici kartındaki model", "http://127.0.0.1:8101/completion"),
    ("FsocietyBox düz - yalnız mimari hibrit yok", "http://127.0.0.1:8102/completion"),
    ("FsocietyBox + GDN yazma kilidi beta sıfır alfa bir", "http://127.0.0.1:8103/completion"),
    ("FsocietyBox + denetim kutuları talimat gövdesine yazamaz", "http://127.0.0.1:8104/completion"),
    ("FsocietyBox + kafa bağımsızlığı", "http://127.0.0.1:8105/completion"),
]

def kos_anlasilir(n=50):
    inputs = oku_ifeval(IFEVAL)[:n]
    print(f"\n=== 5 model yan yana, {n} soru, Vulkan ngl 99 yalnız GPU ===")
    print(f"Kart gibi: 80-85 token, temp 0.6, aynı sorular")
    print("")
    sonuclar=[]
    for adi, url in MODELS:
        print(f"--- {adi} ---")
        print(f"Başlıyor... 0/{n}")
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
            # anlaşılır ilerleme: her 10'da bir
            if (i+1)%10==0:
                print(f"  {i+1}/{n} bitti")
        s=skor_ifeval(mapping, inputs)
        print(f"  Sonuç: {s['prompt_ok']}/{n} soru tamamen doğru")
        print(f"  Yani %{s['prompt_level_strict']*100:.1f} başarı")
        print(f"  Talimat başarısı: %{s['instruction_level_strict']*100:.1f}\n")
        sonuclar.append((adi, s))
    # anlaşılır tablo
    print("\n========== YAN YANA TABLO ==========")
    print(f"{'Model':45} | Doğru/50 | Başarı")
    print("-"*75)
    for adi, s in sonuclar:
        print(f"{adi[:45]:45} | {s['prompt_ok']:2d}/50    | %{s['prompt_level_strict']*100:4.1f}")
    print("-"*75)
    print("\nÜretici kartı: Saf Qwen3.5-0.8B düşünme kapalı %52.1 açık %44.0")
    print("Senin mimarin kartın üstüne eklenince ne kadar sıçradığı burada belli olur")
    # kaydet
    out=Path("/home/void0x14/Documents/echo/goals/fsocietybox/anlasilir-5li.json")
    out.write_text(json.dumps({adi:{"prompt":s["prompt_level_strict"],"ok":s["prompt_ok"]} for adi,s in sonuclar}, ensure_ascii=False, indent=2))
    print(f"\nKaydedildi: {out}")

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    args=p.parse_args()
    kos_anlasilir(n=args.n)
