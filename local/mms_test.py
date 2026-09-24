"""
Quick standard-Bangla test: facebook/mms-tts-ben (open, NO login, NO reference).
VITS — CPU-te fast. Sudhu text -> speech.

    python mms_test.py --text-file script.txt --out out_mms_ben.wav
"""
import argparse
import re
import sys


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument("--text-file", required=True)
    p.add_argument("--out", default="out_mms_ben.wav")
    a = p.parse_args()

    text = open(a.text_file, encoding="utf-8").read().strip()
    # sentence e bhag kore (VITS lomba text e drift kore)
    sents = [s.strip() for s in re.split(r"(?<=[।!?])\s+|\n+", text) if s.strip()]

    import numpy as np
    import soundfile as sf
    import torch
    from transformers import VitsModel, AutoTokenizer

    print("[i] Loading facebook/mms-tts-ben (open, no auth)...")
    model = VitsModel.from_pretrained("facebook/mms-tts-ben")
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-ben")
    sr = model.config.sampling_rate

    chunks = []
    gap = np.zeros(int(sr * 0.25), dtype=np.float32)
    for i, s in enumerate(sents):
        print(f"[i] ({i+1}/{len(sents)}) {s[:50]}...")
        inp = tok(s, return_tensors="pt")
        with torch.no_grad():
            w = model(**inp).waveform[0].cpu().numpy().astype(np.float32)
        if i:
            chunks.append(gap)
        chunks.append(w)
    out = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
    sf.write(a.out, out, sr)
    print(f"[OK] Saved: {a.out}  ({len(out)/sr:.1f}s @ {sr}Hz)")


if __name__ == "__main__":
    main()
