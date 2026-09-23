#!/usr/bin/env python3
"""
train_voice.py — headless Chatterbox full-T3 voice finetune (Bangla + English).

Productionized from notebooks/chatterbox_bangla_finetune_colab.ipynb. Same proven
pipeline (silero-vad slicing -> faster-whisper transcription -> LJSpeech dataset ->
Banglabox training/train.py full-T3 finetune), but headless: audio path is passed
in (no google.colab / files.upload), progress is streamed to a JSON status file, and
BOTH languages go through the SAME ChatterboxTTS (is_turbo=False) full-finetune path.

------------------------------------------------------------------------------
BACKEND USAGE (FastAPI runs this as a subprocess)
------------------------------------------------------------------------------
    python webapp/backend/train_voice.py \
        --audio      /path/to/user_voice.wav \
        --language   bn \
        --voice-id   shuvo01 \
        --out-dir    /content/drive/MyDrive/voices/shuvo01 \
        --status-file /tmp/train_shuvo01.json \
        --epochs     12 \
        --keep-bangla            # (bn only; use --no-keep-bangla to disable)

The parent polls --status-file (rewritten atomically at every stage). The process
exits 0 on success and nonzero on error (the error is also written to the status
file with done=false). Heavy imports live inside functions so `--help` is instant.

MEMORY NOTE (free-tier Colab T4, ~13 GB RAM): the ASR model is fully freed before
the merge/train stage, ASR defaults to faster-whisper "medium" + int8, and the merge
frees the base weights early. If the process is still SIGKILLed by the OS it exits
with returncode -9 (uncatchable by Python, so the status file keeps its last stage) —
the parent should map returncode == -9 to: "Out of memory (exit -9) — try a shorter
clip or ASR_MODEL=small". In-process OOMs (MemoryError / CUDA OOM) are caught and
written to the status file with that guidance automatically.

Language handling:
  * bn: new_vocab_size = checkpoint/NEW_VOCAB_SIZE.txt (2530). With --keep-bangla,
        the released Bangla LoRA adapter is merged into a resized T3 first, so the
        finetuned model KEEPS Bangla and only adapts to the speaker.
  * en: base Chatterbox is English-native. Full-T3 finetune from base weights with
        the model's ORIGINAL text_tokens_dict_size (read from t3_cfg.safetensors,
        = 704). No resize to 2530, no Bangla merge.

------------------------------------------------------------------------------
WHAT IS WRITTEN TO --out-dir
------------------------------------------------------------------------------
    t3_finetuned.safetensors   trained FULL T3 weights (NOT a LoRA adapter)
    tokenizer.json             the tokenizer these weights pair with
    voice.json                 metadata (schema below)

The large base weights (ve/s3gen/t3_cfg/conds) are NOT copied per-voice — they come
from the SHARED Banglabox snapshot pretrained_models that engines.py already
downloads. voice.json["pairs_with"] records that pairing.

voice.json schema:
    {
      "voice_id": "shuvo01",
      "name": null,
      "language": "bn",                       # "bn" | "en"
      "engine": "chatterbox-finetune",
      "vocab_size": 2530,                     # bn=2530, en=704
      "base_vocab_size": 704,                 # base T3 text_emb rows
      "keep_bangla": true,                    # bn only; false for en
      "sr": 24000,
      "created": "2026-09-23T12:00:00Z",
      "files": {"t3_weights": "t3_finetuned.safetensors",
                "tokenizer": "tokenizer.json"},
      "pairs_with": {
        "snapshot_repo": "Banglabox/chatterbox-bangla-tts",
        "pretrained_models": ["ve.safetensors", "s3gen.safetensors",
                              "t3_cfg.safetensors", "conds.pt", "tokenizer.json"]
      },
      "inference": {"class": "ChatterboxTTS.from_local + resize_and_load_t3_weights",
                    "load": "see module docstring / HOW INFERENCE LOADS below"}
    }

------------------------------------------------------------------------------
HOW INFERENCE LOADS A TRAINED VOICE  (wire this into engines.py)
------------------------------------------------------------------------------
Mirror inference/infer.py's BanglaTTS load path, but load the full finetuned T3
instead of PeftModel(adapter). Given VOICE_DIR (the --out-dir) and the shared
Banglabox snapshot:

    import json, torch
    from safetensors.torch import load_file
    from chatterbox_.tts import ChatterboxTTS          # vendored (engines._ensure_banglabox)
    from chatterbox_.models.t3.t3 import T3
    from model import resize_and_load_t3_weights        # inference/src/model.py

    meta   = json.load(open(f"{VOICE_DIR}/voice.json"))
    mdir   = f"{SNAPSHOT}/pretrained_models"             # shared base weights
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base = ChatterboxTTS.from_local(mdir, device="cpu")
    cfg  = base.t3.hp
    cfg.text_tokens_dict_size = meta["vocab_size"]       # bn=2530, en=704
    if hasattr(cfg, "use_cache"): cfg.use_cache = False
    nt = T3(hp=cfg)
    nt = resize_and_load_t3_weights(nt, base.t3.state_dict())
    sd = load_file(f"{VOICE_DIR}/t3_finetuned.safetensors")
    nt.load_state_dict(sd, strict=False)                 # overlay finetuned weights
    base.t3 = nt
    base.t3.to(device).eval(); base.s3gen.to(device).eval(); base.ve.to(device).eval()
    base.device = device

    GEN = dict(temperature=0.5, repetition_penalty=1.5, min_p=0.05,
               cfg_weight=0.5, exaggeration=0.5)
    text = user_text
    if meta["language"] == "bn":
        from bn_norm import for_tts as _n; text = _n(text)      # Stage-2 Bangla norm
        text = text if text.strip().startswith("[bn]") else "[bn]" + text.strip()
    wav = base.generate(text=text, audio_prompt_path=user_ref_wav, **GEN)  # (float32, base.sr)

Notes:
  * The reference clip (audio_prompt_path) is still required at inference (zero-shot
    prompt); use one of the user's own clips for best identity match.
  * For bn you MAY also re-apply infer.py's grapheme remap ({"ঢ":"ধ","খ":"ক"}) as a
    safety net, but it is optional here (the merged base already baked it in).
"""

import argparse
import os
import sys


# --------------------------------------------------------------------------- #
# status file (atomic JSON, polled by FastAPI)                                 #
# --------------------------------------------------------------------------- #
_STATUS = {
    "voice_id": None,
    "language": None,
    "stage": "init",
    "progress_pct": 0,
    "message": "",
    "done": False,
    "error": None,
}
_STATUS_FILE = None


def set_status(**fields):
    """Merge fields into the global status dict and atomically rewrite the file."""
    import json
    import tempfile
    _STATUS.update(fields)
    if not _STATUS_FILE:
        return
    try:
        d = os.path.dirname(os.path.abspath(_STATUS_FILE)) or "."
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(_STATUS, f, ensure_ascii=False)
        os.replace(tmp, _STATUS_FILE)
    except Exception as e:  # status write must never crash training
        print(f"[status] write failed: {e}", flush=True)


# --------------------------------------------------------------------------- #
# small helpers                                                                #
# --------------------------------------------------------------------------- #
def _link_or_copy(src, dst):
    """Symlink src->dst (cheap, for big dirs); fall back to copy on failure."""
    import shutil
    if os.path.lexists(dst):
        return
    try:
        os.symlink(src, dst)
    except OSError:
        (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, dst)


def _find(root, filename):
    for dirpath, _dirs, files in os.walk(root):
        if filename in files:
            return dirpath
    return None


def _safetensors_vocab(path, key="text_emb.weight"):
    """Read only the safetensors JSON header to get a tensor's row count (vocab)."""
    import json
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
    if key in hdr:
        return int(hdr[key]["shape"][0])
    for k in hdr:
        if k.endswith(key):
            return int(hdr[k]["shape"][0])
    raise KeyError(f"{key} not found in {path}")


# --------------------------------------------------------------------------- #
# 1. snapshot + writable workdir                                              #
# --------------------------------------------------------------------------- #
def setup_workspace(workdir):
    """Download the Banglabox snapshot and assemble a writable training workdir.

    Layout produced (cwd for train.py):
        workdir/train.py, workdir/src/ ...     (training code, copied)
        workdir/pretrained_models              (symlink -> snapshot base weights + Bangla tokenizer)
        workdir/checkpoint                     (symlink -> adapter + NEW_VOCAB_SIZE.txt)
    """
    import shutil
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from huggingface_hub import snapshot_download

    snap = snapshot_download(repo_id="Banglabox/chatterbox-bangla-tts")

    train_dir = _find(snap, "train.py")            # .../training
    pm_dir = _find(snap, "ve.safetensors")         # .../pretrained_models
    ckpt_dir = _find(snap, "NEW_VOCAB_SIZE.txt")   # .../checkpoint (root copy)
    # prefer the checkpoint/ dir (has adapter/ sibling), not inference/
    adapter_parent = _find(snap, "adapter_config.json")  # .../checkpoint/adapter
    if adapter_parent:
        ckpt_dir = os.path.dirname(adapter_parent)
    if not (train_dir and pm_dir and ckpt_dir):
        raise RuntimeError(f"Unexpected snapshot layout: train={train_dir} "
                           f"pm={pm_dir} ckpt={ckpt_dir}")

    os.makedirs(workdir, exist_ok=True)
    # copy the (small) training code so config/train are importable & editable
    for item in os.listdir(train_dir):
        s = os.path.join(train_dir, item)
        d = os.path.join(workdir, item)
        if os.path.exists(d):
            continue
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
    # symlink the big base weights + checkpoint (no multi-GB copy per run)
    _link_or_copy(pm_dir, os.path.join(workdir, "pretrained_models"))
    _link_or_copy(ckpt_dir, os.path.join(workdir, "checkpoint"))

    return snap, workdir


# --------------------------------------------------------------------------- #
# 2. slice (silero-vad) + transcribe (faster-whisper) -> LJSpeech dataset      #
# --------------------------------------------------------------------------- #
def build_dataset(audio_path, data_dir, language, asr_model="medium",
                  target_sr=24000, min_clip_s=4.0, max_clip_s=12.0, merge_gap_s=0.35):
    import csv
    import gc
    import shutil

    import librosa
    import numpy as np
    import soundfile as sf
    import torch

    wavs = os.path.join(data_dir, "wavs")
    shutil.rmtree(data_dir, ignore_errors=True)
    os.makedirs(wavs, exist_ok=True)

    # ---- slice with silero-vad ----
    set_status(stage="slicing", progress_pct=8, message="Loading VAD and slicing audio")
    from silero_vad import get_speech_timestamps, load_silero_vad
    vad = load_silero_vad()

    y, sr = librosa.load(audio_path, sr=None, mono=True)  # handles wav/mp3 via audioread
    y = y.astype("float32")
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
    ts = get_speech_timestamps(torch.from_numpy(y16), vad,
                               sampling_rate=16000, min_silence_duration_ms=250, threshold=0.5)
    if not ts:
        raise RuntimeError("No speech detected in --audio (VAD returned nothing).")

    # merge tiny gaps
    merged = [dict(ts[0])]
    for seg in ts[1:]:
        if (seg["start"] - merged[-1]["end"]) / 16000.0 <= merge_gap_s:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    # greedily pack into min..max second windows at natural pauses
    chunks, cs, ce = [], None, None
    for seg in merged:
        s, e = seg["start"] / 16000.0, seg["end"] / 16000.0
        if cs is None:
            cs, ce = s, e
        elif (e - cs) <= max_clip_s:
            ce = e
        else:
            if (ce - cs) >= min_clip_s:
                chunks.append((cs, ce))
            cs, ce = s, e
        if ce is not None and (ce - cs) >= max_clip_s:
            chunks.append((cs, ce))
            cs, ce = None, None
    if cs is not None and (ce - cs) >= min_clip_s:
        chunks.append((cs, ce))

    # resample full audio once, then cut clips
    y_out = librosa.resample(y, orig_sr=sr, target_sr=target_sr) if sr != target_sr else y
    base = os.path.splitext(os.path.basename(audio_path))[0].replace(" ", "_")
    clip_names = []
    for i, (s, e) in enumerate(chunks):
        a, b = int(s * target_sr), int(e * target_sr)
        clip = y_out[a:b]
        if len(clip) < int(min_clip_s * target_sr):
            continue
        name = f"{base}_{i:04d}.wav"
        sf.write(os.path.join(wavs, name), clip, target_sr)
        clip_names.append(name)
    if not clip_names:
        raise RuntimeError("Slicing produced 0 usable clips (audio too short/quiet?).")
    print(f"[slice] {len(clip_names)} clips", flush=True)

    # ---- transcribe with faster-whisper ----
    # Default to the lighter "medium" model + int8 to keep peak RAM low on free-tier
    # Colab (~13 GB). The ASR model MUST be fully freed before the merge/train stage.
    set_status(stage="asr", progress_pct=20,
               message=f"Transcribing {len(clip_names)} clips ({language}, {asr_model})")
    from faster_whisper import WhisperModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "int8_float16" if dev == "cuda" else "int8"
    wm = WhisperModel(asr_model, device=dev, compute_type=compute_type)

    rows = []
    total = len(clip_names)
    for idx, name in enumerate(clip_names):
        segs, _info = wm.transcribe(os.path.join(wavs, name), language=language,
                                    beam_size=5, vad_filter=False)
        text = " ".join(seg.text.strip() for seg in segs).strip()  # consume generator now
        if text:
            rows.append((name, text))
        del segs, _info
        if idx % 10 == 0:
            set_status(stage="asr",
                       progress_pct=20 + int(15 * (idx + 1) / total),
                       message=f"Transcribed {idx + 1}/{total}")

    # AGGRESSIVELY free ASR before anything else — no reference may survive into
    # the merge/train stage (this is what caused the SIGKILL / exit -9 OOM).
    del wm
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    if not rows:
        raise RuntimeError("Transcription produced no non-empty text.")

    meta_csv = os.path.join(data_dir, "metadata.csv")
    with open(meta_csv, "w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        for name, text in rows:
            w.writerow([name, text])
    print(f"[asr] {len(rows)} usable rows -> {meta_csv}", flush=True)
    return len(rows)


# --------------------------------------------------------------------------- #
# 3b. Bangla-preserving base (merge released LoRA into resized T3)             #
# --------------------------------------------------------------------------- #
def build_bangla_base(workdir, vocab_size, out_pt):
    """Replicates infer.py's Bangla load path + merge_and_unload -> a full T3
    (vocab=2530) state dict that already knows Bangla. Used as training init."""
    import gc

    import torch
    os.chdir(workdir)
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    from peft import PeftModel

    from src.chatterbox_.models.t3.t3 import T3
    from src.chatterbox_.tts import ChatterboxTTS
    from src.model import resize_and_load_t3_weights

    def _free():
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    adapter = os.path.join(workdir, "checkpoint", "adapter")

    # Load base on CPU, keep ONLY the T3 config + T3 weights, then free the rest
    # (s3gen/ve are not needed for the merge and just waste ~1.3 GB of RAM).
    base = ChatterboxTTS.from_local("./pretrained_models", device="cpu")
    cfg = base.t3.hp
    cfg.text_tokens_dict_size = vocab_size
    if hasattr(cfg, "use_cache"):
        cfg.use_cache = False
    ps = base.t3.state_dict()   # tensors keep t3 storage alive; s3gen/ve get GC'd
    del base
    _free()

    # Resize into a fresh T3, then drop the old base weights before the LoRA merge.
    nt = T3(hp=cfg)
    nt = resize_and_load_t3_weights(nt, ps)
    del ps
    _free()

    # Apply + merge the released Bangla LoRA, then free the PEFT wrapper immediately.
    peft_m = PeftModel.from_pretrained(nt, adapter, is_trainable=False)
    merged = peft_m.merge_and_unload()
    del peft_m, nt
    _free()

    sd = merged.state_dict()
    if "text_emb.weight" not in sd or sd["text_emb.weight"].shape[0] != vocab_size:
        raise RuntimeError(f"Merged T3 vocab mismatch: {sd.get('text_emb.weight', None)}")
    torch.save(sd, out_pt)
    print(f"[merge] Bangla base saved -> {out_pt} ({len(sd)} keys)", flush=True)
    del merged, sd
    _free()


# --------------------------------------------------------------------------- #
# 4. run training (train.py in-process, T4-patched, with progress callback)    #
# --------------------------------------------------------------------------- #
def run_training(workdir, data_dir, out_local, vocab_size,
                 keep_bangla, bangla_pt, epochs):
    import torch
    os.chdir(workdir)
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    import train  # noqa: applies its torch.load patch + binds names in train namespace
    from transformers import TrainerCallback
    from transformers import TrainingArguments as _TA

    from src.config import TrainConfig

    # -- (a) TrainConfig instance override (train_modal.py pattern) --
    overrides = dict(
        model_dir="./pretrained_models",
        csv_path=os.path.join(data_dir, "metadata.csv"),
        wav_dir=os.path.join(data_dir, "wavs"),
        preprocessed_dir=os.path.join(data_dir, "preprocess"),
        output_dir=out_local,
        ljspeech=True, json_format=False, preprocess=True,
        is_turbo=False, is_inference=False,
        new_vocab_size=vocab_size,
        batch_size=1, grad_accum=16, num_epochs=epochs,
        learning_rate=5e-6, save_steps=200, save_total_limit=1,
        dataloader_num_workers=2, max_text_len=256, max_speech_len=850,
        prompt_duration=3.0,
    )
    _orig_init = TrainConfig.__init__

    def _patched_init(self):
        _orig_init(self)
        for k, v in overrides.items():
            setattr(self, k, v)
    TrainConfig.__init__ = _patched_init

    # -- (b) T4: bf16 -> fp16 --
    def _TA_t4(**kw):
        kw["bf16"] = False
        kw["fp16"] = True
        return _TA(**kw)
    train.TrainingArguments = _TA_t4

    # -- (c) preprocess stage status + free VE/S3Gen VRAM before training --
    import gc as _gc
    _orig_pp = train.preprocess_dataset_ljspeech

    def _pp(cfg, eng):
        set_status(stage="preprocess", progress_pct=40,
                   message="Extracting speech/speaker tokens")
        out = _orig_pp(cfg, eng)
        # VE/S3Gen were moved to GPU for preprocessing but are NOT used during the
        # T3 training loop (only needed again at inference). Move them back to CPU
        # to lower peak VRAM while the T3 trains.
        try:
            eng.ve.to("cpu")
            eng.s3gen.to("cpu")
        except Exception:
            pass
        _gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return out
    train.preprocess_dataset_ljspeech = _pp

    # -- (d) Bangla weight overlay (keep Bangla) --
    if keep_bangla and bangla_pt and os.path.exists(bangla_pt):
        _orig_resize = train.resize_and_load_t3_weights

        def _resize_overlay(new_model, pretrained_sd):
            m = _orig_resize(new_model, pretrained_sd)
            sd = torch.load(bangla_pt, map_location="cpu")
            miss, unexp = m.load_state_dict(sd, strict=False)
            print(f"[overlay] Bangla init loaded missing={len(miss)} unexpected={len(unexp)}",
                  flush=True)
            return m
        train.resize_and_load_t3_weights = _resize_overlay

    # -- (e) progress callback --
    class _Prog(TrainerCallback):
        def on_train_begin(self, args, state, control, **k):
            set_status(stage="training", progress_pct=45, message="Training started")

        def on_step_end(self, args, state, control, **k):
            if state.max_steps:
                p = 45 + int(50 * state.global_step / max(1, state.max_steps))
                set_status(stage="training", progress_pct=min(94, p),
                           message=f"step {state.global_step}/{state.max_steps}")

        def on_train_end(self, args, state, control, **k):
            set_status(stage="training", progress_pct=95, message="Training loop done")

    _orig_trainer = train.Trainer

    def _trainer_factory(*a, **k):
        cbs = list(k.pop("callbacks", None) or [])
        cbs.append(_Prog())
        return _orig_trainer(*a, callbacks=cbs, **k)
    train.Trainer = _trainer_factory

    train.main()


# --------------------------------------------------------------------------- #
# 5. save artifacts to --out-dir                                              #
# --------------------------------------------------------------------------- #
def save_outputs(workdir, out_local, out_dir, voice_id, name, language,
                 vocab_size, base_vocab_size, keep_bangla, sr):
    import datetime
    import json
    import shutil

    os.makedirs(out_dir, exist_ok=True)
    src_weights = os.path.join(out_local, "t3_finetuned.safetensors")
    if not os.path.exists(src_weights):
        raise RuntimeError(f"Training did not produce {src_weights}")
    shutil.copy2(src_weights, os.path.join(out_dir, "t3_finetuned.safetensors"))
    shutil.copy2(os.path.join(workdir, "pretrained_models", "tokenizer.json"),
                 os.path.join(out_dir, "tokenizer.json"))

    meta = {
        "voice_id": voice_id,
        "name": name,
        "language": language,
        "engine": "chatterbox-finetune",
        "vocab_size": vocab_size,
        "base_vocab_size": base_vocab_size,
        "keep_bangla": bool(keep_bangla) if language == "bn" else False,
        "sr": sr,
        "created": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": {"t3_weights": "t3_finetuned.safetensors", "tokenizer": "tokenizer.json"},
        "pairs_with": {
            "snapshot_repo": "Banglabox/chatterbox-bangla-tts",
            "pretrained_models": ["ve.safetensors", "s3gen.safetensors",
                                  "t3_cfg.safetensors", "conds.pt", "tokenizer.json"],
        },
        "inference": {
            "class": "ChatterboxTTS.from_local + resize_and_load_t3_weights",
            "load": ("base=ChatterboxTTS.from_local(snapshot/pretrained_models); "
                     "cfg.text_tokens_dict_size=vocab_size; nt=T3(cfg); "
                     "nt=resize_and_load_t3_weights(nt, base.t3.state_dict()); "
                     "nt.load_state_dict(load_file(t3_finetuned.safetensors), strict=False); "
                     "base.t3=nt; generate(text, audio_prompt_path=ref, **GEN). "
                     "bn: prepend [bn] + bn_norm.for_tts."),
        },
    }
    with open(os.path.join(out_dir, "voice.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[save] wrote {out_dir}/{{t3_finetuned.safetensors,tokenizer.json,voice.json}}",
          flush=True)
    return meta


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description="Headless Chatterbox full-T3 voice finetune (Bangla + English).")
    p.add_argument("--audio", required=True, help="User voice recording (wav/mp3), 5-10 min.")
    p.add_argument("--language", required=True, choices=["bn", "en"], help="Target language.")
    p.add_argument("--voice-id", required=True, help="Slug identifying this voice.")
    p.add_argument("--out-dir", required=True,
                   help="Output dir for trained model + voice.json (can be a Drive path).")
    p.add_argument("--status-file", required=True,
                   help="JSON file continuously rewritten with progress.")
    p.add_argument("--epochs", type=int, default=12, help="Training epochs (default 12).")
    p.add_argument("--asr-model", default=os.environ.get("ASR_MODEL", "medium"),
                   help="faster-whisper model for transcription (default 'medium', or "
                        "env ASR_MODEL). Use 'small'/'base' for even less RAM, "
                        "'large-v3' for best accuracy if you have more RAM.")
    p.add_argument("--name", default=None, help="Optional human-readable voice name.")
    p.add_argument("--workdir", default=None,
                   help="Writable scratch dir (default: <tempdir>/tv_<voice-id>).")
    keep = getattr(argparse, "BooleanOptionalAction", None)
    if keep is not None:
        p.add_argument("--keep-bangla", action=keep, default=True,
                       help="bn only: init from released Bangla LoRA (keeps Bangla). "
                            "Use --no-keep-bangla to disable.")
    else:  # very old argparse fallback
        p.add_argument("--keep-bangla", dest="keep_bangla", action="store_true", default=True)
        p.add_argument("--no-keep-bangla", dest="keep_bangla", action="store_false")
    return p


def main(argv=None):
    global _STATUS_FILE
    args = build_parser().parse_args(argv)

    _STATUS_FILE = args.status_file
    keep_bangla = bool(getattr(args, "keep_bangla", True)) and args.language == "bn"
    _STATUS.update(voice_id=args.voice_id, language=args.language)
    set_status(stage="setup", progress_pct=1, message="Starting", done=False, error=None)

    import tempfile
    workdir = args.workdir or os.path.join(tempfile.gettempdir(), f"tv_{args.voice_id}")
    data_dir = os.path.join(workdir, "MyTTSDataset")
    out_local = os.path.join(workdir, "chatterbox_output")
    bangla_pt = os.path.join(workdir, "bangla_merged_t3.pt")
    sr = 24000  # Chatterbox S3Gen output sample rate

    try:
        if not os.path.exists(args.audio):
            raise FileNotFoundError(f"--audio not found: {args.audio}")

        # 1. snapshot + workspace
        set_status(stage="setup", progress_pct=3, message="Downloading model snapshot")
        snap, workdir = setup_workspace(workdir)

        # vocab size per language
        t3_cfg_path = os.path.join(workdir, "pretrained_models", "t3_cfg.safetensors")
        base_vocab = _safetensors_vocab(t3_cfg_path)  # 704 for base Chatterbox
        if args.language == "bn":
            vocab_size = int(open(os.path.join(
                workdir, "checkpoint", "NEW_VOCAB_SIZE.txt")).read().strip())  # 2530
        else:
            vocab_size = base_vocab  # en: keep original, no resize
        set_status(stage="setup", progress_pct=5,
                   message=f"vocab={vocab_size} base_vocab={base_vocab} keep_bangla={keep_bangla}")

        # 2. dataset (slice + transcribe)
        n_rows = build_dataset(args.audio, data_dir, args.language, asr_model=args.asr_model)
        set_status(stage="asr", progress_pct=36, message=f"Dataset ready ({n_rows} clips)")

        # 3b. Bangla merge (init that keeps Bangla)
        if keep_bangla:
            set_status(stage="merge", progress_pct=38,
                       message="Merging released Bangla adapter into base")
            build_bangla_base(workdir, vocab_size, bangla_pt)

        # 4. train
        run_training(workdir, data_dir, out_local, vocab_size,
                     keep_bangla, bangla_pt, args.epochs)

        # 5. save
        set_status(stage="saving", progress_pct=96, message="Saving model + metadata")
        save_outputs(workdir, out_local, args.out_dir, args.voice_id, args.name,
                     args.language, vocab_size, base_vocab, keep_bangla, sr)

        set_status(stage="done", progress_pct=100, message="Finetune complete",
                   done=True, error=None)
        print("DONE", flush=True)
        return 0

    except Exception as e:
        import traceback
        traceback.print_exc()
        msg = f"{type(e).__name__}: {e}"
        low = f"{type(e).__name__} {e}".lower()
        if isinstance(e, MemoryError) or "out of memory" in low or "cuda" in low and "memory" in low:
            msg = ("Out of memory — try a shorter clip or ASR_MODEL=small "
                   "(--asr-model small). Details: " + msg)
        set_status(stage="error", message=msg, done=False, error=msg)
        return 1


if __name__ == "__main__":
    sys.exit(main())
