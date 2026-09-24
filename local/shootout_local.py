"""
Local CPU Bangla TTS shootout — Colab GPU quota lage na.

IndicF5 (base) vs ehzawad/indicf5-bangla-tts (Bangladeshi) — CPU-te chalay.
Reference: YouTube link (--ref-yt) ba local file (--ref)। Transcript auto (whisper) ba --ref-text.

⚠️ SLOW (CPU): protyek clip ~3-15 min. RAM ~5-8GB free rakho (onno app bondho).

--- Setup (Python 3.13 venv) ---
    pip install -r requirements_shootout.txt
    pip install git+https://github.com/ai4bharat/IndicF5.git
    pip install yt-dlp faster-whisper        # YouTube ref + auto transcript er jonno

--- Run (YouTube reference) ---
    python shootout_local.py --ref-yt "https://youtu.be/XXXX" --ref-start 5 --ref-dur 12 ^
        --text "সুপ্রভাত। আজকের আবহাওয়া সম্পর্কে কিছু তথ্য জানানো হচ্ছে।"

--- Run (local file reference) ---
    python shootout_local.py --ref ref.wav --ref-text "clip e ja bola" --text "..."

Output: out_A_indicf5_base.wav, out_B_indicf5_bangladeshi.wav  (24kHz)
"""
import argparse
import glob
import os
import subprocess


def to_f32(a):
    import numpy as np
    a = np.asarray(a)
    if a.dtype == np.int16:
        a = a.astype(np.float32) / 32768.0
    return a.astype(np.float32).squeeze()


def yt_to_wav(url, start, dur, dst):
    print(f"[i] YouTube theke reference download ({start}-{start+dur}s)...")
    base = os.path.splitext(dst)[0] + "_src"
    sec = f"*{start}-{start + dur}"
    subprocess.run(["yt-dlp", "-f", "bestaudio/best", "--no-playlist", "--quiet",
                    "--no-warnings", "--download-sections", sec,
                    "-o", base + ".%(ext)s", url], check=False)
    srcs = glob.glob(base + ".*")
    if not srcs:
        subprocess.run(["yt-dlp", "-f", "bestaudio/best", "--no-playlist", "--quiet",
                        "--no-warnings", "-o", base + ".%(ext)s", url], check=False)
        srcs = glob.glob(base + ".*")
    if not srcs:
        raise SystemExit("YouTube audio download fail. Link check koro / yt-dlp install ache?")
    subprocess.run(["ffmpeg", "-y", "-ss", str(start), "-t", str(dur),
                    "-i", srcs[0], "-ac", "1", "-ar", "24000", dst, "-loglevel", "error"], check=True)
    print(f"[✓] Reference: {dst}")


def transcribe(path):
    print("[i] Reference transcript banacchi (faster-whisper medium, bn)...")
    from faster_whisper import WhisperModel
    wm = WhisperModel("medium", device="cpu", compute_type="int8")
    segs, _ = wm.transcribe(path, language="bn", beam_size=5)
    txt = "".join(s.text for s in segs).strip()
    del wm
    print("[i] REF_TEXT =", txt)
    return txt


def run_model(repo_id, text, ref, ref_text, out_path):
    import gc
    import soundfile as sf
    from transformers import AutoModel

    print(f"\n[i] Loading {repo_id} (CPU, koyek min)...")
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
    print("[i] Generating (SLOW on CPU, dhoirjo)...")
    wav = model(text, ref_audio_path=ref, ref_text=ref_text)
    sf.write(out_path, to_f32(wav), 24000)
    print(f"[✓] Saved: {out_path}")
    del model
    gc.collect()


def main():
    p = argparse.ArgumentParser(description="Local CPU Bangla TTS shootout")
    p.add_argument("--ref", help="Local reference wav/mp3 (5-15s)")
    p.add_argument("--ref-yt", help="YouTube URL (reference hisebe)")
    p.add_argument("--ref-start", type=float, default=5)
    p.add_argument("--ref-dur", type=float, default=12)
    p.add_argument("--ref-text", help="Reference transcript (na dile auto-transcribe)")
    p.add_argument("--text", required=True, help="Target promito Bangla text")
    p.add_argument("--out-dir", default=".")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ref = os.path.join(args.out_dir, "ref.wav")

    if args.ref_yt:
        yt_to_wav(args.ref_yt, args.ref_start, args.ref_dur, ref)
    elif args.ref:
        if not os.path.exists(args.ref):
            raise SystemExit(f"Reference pawa jay nai: {args.ref}")
        subprocess.run(["ffmpeg", "-y", "-i", args.ref, "-ac", "1", "-ar", "24000",
                        "-t", "15", ref, "-loglevel", "error"], check=True)
    else:
        raise SystemExit("--ref-yt ba --ref ekta dao.")

    ref_text = args.ref_text or transcribe(ref)
    if not ref_text:
        raise SystemExit("Reference transcript khali — --ref-text hate dao.")

    run_model("ai4bharat/IndicF5", args.text, ref, ref_text,
              os.path.join(args.out_dir, "out_A_indicf5_base.wav"))
    run_model("ehzawad/indicf5-bangla-tts", args.text, ref, ref_text,
              os.path.join(args.out_dir, "out_B_indicf5_bangladeshi.wav"))
    print("\n[✓] Done. out_A_* (base) ar out_B_* (Bangladeshi) shune compare koro.")


if __name__ == "__main__":
    main()
