"""
Local voice-clone test using XTTS-v2 (Coqui TTS).
English + 16 languages zero-shot clone. Bangla NAI (Bangla er jonno Colab notebook use koro).

Usage:
    python clone_xtts.py --ref ../voices/sample.wav --text "Hello, this is my cloned voice." --lang en --out ../output/en.wav

CPU te slow (ek line ~1-2 min). GPU thakle onek fast.
Prothombar chalale model download hobe (~1.8 GB).
"""
import argparse
import os


def main():
    p = argparse.ArgumentParser(description="XTTS-v2 voice clone (local)")
    p.add_argument("--ref", required=True, help="Reference voice wav (10-30s clean)")
    p.add_argument("--text", required=True, help="Text to speak (cloned voice)")
    p.add_argument("--lang", default="en",
                   help="Language code: en, hi, ar, es, fr, de, it, pt, pl, tr, ru, nl, cs, zh-cn, ja, ko, hu")
    p.add_argument("--out", default="../output/output.wav", help="Output wav path")
    args = p.parse_args()

    if not os.path.exists(args.ref):
        raise SystemExit(f"Reference file pawa jay nai: {args.ref}\n"
                         f"voices/ folder e sample.wav rakho (dekho RECORDING_GUIDE.md).")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Import ekhane rakhlam jate --help fast thake
    import torch
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[i] Device: {device}  (CPU hole slow hobe, dhoirjo rakho)")
    print("[i] Model load hocche (prothombar download hobe ~1.8GB)...")

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    print(f"[i] Generate hocche -> {args.out}")
    tts.tts_to_file(
        text=args.text,
        speaker_wav=args.ref,
        language=args.lang,
        file_path=args.out,
    )
    print(f"[✓] Done: {args.out}")


if __name__ == "__main__":
    main()
