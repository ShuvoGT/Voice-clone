"""
Local CPU Bangla TTS shootout — Colab GPU quota lage na.

IndicF5 (base) vs ehzawad/indicf5-bangla-tts (Bangladeshi) — CPU-te chalay.
⚠️ SLOW (CPU): protyek clip ~3-15 min lagte pare. RAM ~5-8GB free rakho
   (onno app bondho koro; model load-e memory lage).

--- Setup (Python 3.11 ba 3.13; 3.14 e torch nao thakte pare) ---
    py -3.13 -m venv .venv
    .venv\\Scripts\\activate
    pip install -r requirements_shootout.txt
    pip install git+https://github.com/ai4bharat/IndicF5.git

--- Run ---
    python shootout_local.py --ref ref.wav --ref-text "reference clip e ja bola" ^
        --text "সুপ্রভাত। আজকের আবহাওয়া সম্পর্কে কিছু তথ্য জানানো হচ্ছে।"

Output: out_A_indicf5_base.wav, out_B_indicf5_bangladeshi.wav  (24kHz)
Duita shune compare koro — kon-ta best Bangladeshi promito.

Note: Windows + CPU e IndicF5 (f5-tts) install kokhono jhamela kore. Error asle
      seta amake dao — Colab (GPU quota reset holে) alternative.
"""
import argparse
import os


def to_f32(a):
    import numpy as np
    a = np.asarray(a)
    if a.dtype == np.int16:
        a = a.astype(np.float32) / 32768.0
    return a.astype(np.float32).squeeze()


def run_model(repo_id, text, ref, ref_text, out_path):
    import gc
    import soundfile as sf
    import torch
    from transformers import AutoModel

    print(f"\n[i] Loading {repo_id} (CPU, koyek min)...")
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)  # CPU by default
    print("[i] Generating (SLOW on CPU, dhoirjo)...")
    wav = model(text, ref_audio_path=ref, ref_text=ref_text)
    sf.write(out_path, to_f32(wav), 24000)
    print(f"[✓] Saved: {out_path}")
    del model
    gc.collect()


def main():
    p = argparse.ArgumentParser(description="Local CPU Bangla TTS shootout")
    p.add_argument("--ref", required=True, help="Bangladeshi promito reference wav (clean, 5-15s)")
    p.add_argument("--ref-text", required=True, help="Reference clip er exact transcript")
    p.add_argument("--text", required=True, help="Target promito Bangla text to speak")
    p.add_argument("--out-dir", default=".", help="Output folder")
    args = p.parse_args()

    if not os.path.exists(args.ref):
        raise SystemExit(f"Reference pawa jay nai: {args.ref}")
    os.makedirs(args.out_dir, exist_ok=True)

    # ekbar-e ekta model (RAM bachate)
    run_model("ai4bharat/IndicF5", args.text, args.ref, args.ref_text,
              os.path.join(args.out_dir, "out_A_indicf5_base.wav"))
    run_model("ehzawad/indicf5-bangla-tts", args.text, args.ref, args.ref_text,
              os.path.join(args.out_dir, "out_B_indicf5_bangladeshi.wav"))

    print("\n[✓] Done. out_A_* (base) ar out_B_* (Bangladeshi) shune compare koro.")


if __name__ == "__main__":
    main()
