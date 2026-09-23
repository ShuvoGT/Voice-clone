"""
FastAPI backend for the Voice Cloning web app.

Endpoints
---------
GET  /api/health           -> {status, device, engines}
POST /api/clone            -> multipart: reference(file) + text + language ; returns wav

Run (local, dev):
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

GPU nai bole local e slow / Bangla nao cholte pare.
Colab GPU te chalate: ../colab_backend.ipynb use koro (public URL dey).
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

import engines

HERE = os.path.dirname(os.path.abspath(__file__))

# Trained voices persist here. Google Drive (mounted in Colab) hole oikhane,
# nahole local. colab_backend.ipynb Drive mount kore /content/drive/MyDrive dey.
_DRIVE = "/content/drive/MyDrive"
VOICES_DIR = os.environ.get("VOICES_DIR") or (
    os.path.join(_DRIVE, "voiceclone_voices") if os.path.isdir(_DRIVE)
    else os.path.join(tempfile.gettempdir(), "voiceclone_voices")
)
os.makedirs(VOICES_DIR, exist_ok=True)

TRAIN_JOBS = {}   # job_id -> {status_file, voice_id, out_dir, proc}


def _slug(name):
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return (s or "voice")[:32]

app = FastAPI(title="Voice Cloning API", version="1.0.0")

# CORS — frontend (Vite dev :5173, ba deployed) theke call korar jonno
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # production e nijer domain e restrict koro
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WORK_DIR = os.path.join(tempfile.gettempdir(), "voiceclone")
os.makedirs(WORK_DIR, exist_ok=True)

MAX_TEXT = 1000               # ek call e max characters
ALLOWED_AUDIO = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4", ".aac"}
TARGET_SR = 24000             # engine gulor jonno normalized mono wav


def _to_wav(src: str, dst: str, sr: int = TARGET_SR) -> str:
    """Je kono audio (webm/mp3/m4a/...) ke mono <sr>Hz WAV e convert kore.

    Recorded .webm engine load korte pare na — tai eta must.
    ffmpeg thakle ffmpeg, na thakle librosa fallback.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        proc = subprocess.run(
            [ffmpeg, "-y", "-i", src, "-ac", "1", "-ar", str(sr), dst],
            capture_output=True, text=True,
        )
        if proc.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0:
            return dst
        # ffmpeg fail hole error message rakho, niche fallback try hobe
        ff_err = proc.stderr[-400:] if proc.stderr else "ffmpeg failed"
    else:
        ff_err = "ffmpeg nai"

    try:
        import librosa
        import soundfile as sf
        y, _ = librosa.load(src, sr=sr, mono=True)
        if y is None or len(y) == 0:
            raise ValueError("audio khali / decode fail")
        sf.write(dst, y, sr)
        return dst
    except Exception as e:
        raise HTTPException(
            400,
            f"Reference audio WAV e convert kora gelo na ({ff_err}; librosa: {e}). "
            f"WAV/MP3 hisebe upload kore dekho.",
        )


def _safe_remove(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


def _youtube_to_wav(url: str, start: float, duration: float, dst: str) -> str:
    """YouTube link theke audio niye [start, start+duration] segment mono WAV e banay."""
    if not shutil.which("yt-dlp"):
        raise HTTPException(500, "yt-dlp install nai (backend requirements re-install koro).")
    if not shutil.which("ffmpeg"):
        raise HTTPException(500, "ffmpeg nai (backend host e).")

    base = dst + ".src"
    dl = ["yt-dlp", "-f", "bestaudio/best", "--no-playlist", "--quiet",
          "--no-warnings", "-o", base + ".%(ext)s"]

    # speed: sudhu dorkari section download korar chesta
    section = f"*{start}-{start + duration}"
    p1 = subprocess.run(dl + ["--download-sections", section, url],
                        capture_output=True, text=True)
    srcs = glob.glob(base + ".*")
    trim = False
    if not srcs:
        # fallback: full audio download, pore ffmpeg e cut
        p2 = subprocess.run(dl + [url], capture_output=True, text=True)
        srcs = glob.glob(base + ".*")
        if not srcs:
            err = (p1.stderr or "")[-250:] + " | " + (p2.stderr or "")[-250:]
            raise HTTPException(400, f"YouTube audio download fail: {err}")
        trim = True

    src = srcs[0]
    cmd = ["ffmpeg", "-y"]
    if trim:
        cmd += ["-ss", str(start), "-t", str(duration)]
    cmd += ["-i", src, "-ac", "1", "-ar", str(TARGET_SR), dst]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    _safe_remove(src)
    if proc.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
        raise HTTPException(400, f"Audio process fail: {(proc.stderr or '')[-250:]}")
    return dst


@app.get("/api/health")
def health():
    import torch
    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "engines": {"english_multilingual": "chatterbox-multilingual", "bangla": "chatterbox-bangla"},
        "voices_dir": VOICES_DIR,
        "drive": os.path.isdir(_DRIVE),
    }


@app.post("/api/clone")
async def clone(
    reference: UploadFile = File(None, description="Reference voice clip (10-30s). Optional if voice_id given."),
    text: str = Form(..., description="Text to speak in the cloned voice"),
    language: str = Form("en", description="'bn' for Bangla, else en/hi/ar/es/fr/..."),
    voice_id: str = Form("", description="Trained voice id (from /api/voices). Empty = zero-shot."),
    exaggeration: float = Form(0.5, description="Emotion intensity 0-1 (higher = more expressive)"),
    cfg_weight: float = Form(0.5, description="Pace/adherence 0-1 (lower = slower, more emotional)"),
    temperature: float = Form(0.8, description="Variation 0.1-1.5"),
    pause_ms: int = Form(0, description="Line gap: silence (ms) between sentences; 0 = off"),
):
    text = (text or "").strip()
    if not text:
        raise HTTPException(400, "Text khali.")
    if len(text) > MAX_TEXT:
        raise HTTPException(400, f"Text boro (max {MAX_TEXT} chars). Chhoto kore chalao.")

    # clamp params to safe ranges
    exaggeration = max(0.0, min(float(exaggeration), 1.5))
    cfg_weight = max(0.0, min(float(cfg_weight), 1.0))
    temperature = max(0.1, min(float(temperature), 1.5))
    pause_ms = max(0, min(int(pause_ms), 2000))

    # trained voice?
    voice_dir = None
    voice_id = (voice_id or "").strip()
    if voice_id:
        voice_dir = os.path.join(VOICES_DIR, voice_id)
        if not os.path.exists(os.path.join(voice_dir, "voice.json")):
            raise HTTPException(404, f"Trained voice pawa jay nai: {voice_id}")

    job = uuid.uuid4().hex[:12]
    ref_wav = os.path.join(WORK_DIR, f"{job}_ref.wav")
    out_path = os.path.join(WORK_DIR, f"{job}_out.wav")
    raw_path = None

    # reference: uploaded, ba trained voice er saved reference.wav
    if reference is not None and reference.filename:
        ext = os.path.splitext(reference.filename)[1].lower()
        if ext not in ALLOWED_AUDIO:
            raise HTTPException(400, f"Audio format support kore na: {ext}. WAV/MP3 daw.")
        raw_path = os.path.join(WORK_DIR, f"{job}_raw{ext}")
        data = await reference.read()
        if not data:
            raise HTTPException(400, "Reference audio khali.")
        with open(raw_path, "wb") as f:
            f.write(data)
        try:
            _to_wav(raw_path, ref_wav)
        except HTTPException:
            _safe_remove(raw_path)
            raise
    elif voice_dir and os.path.exists(os.path.join(voice_dir, "reference.wav")):
        shutil.copy(os.path.join(voice_dir, "reference.wav"), ref_wav)
    else:
        raise HTTPException(400, "Reference audio daw (ba trained voice select koro).")

    # synthesize
    try:
        engines.synthesize(
            text=text, ref_wav=ref_wav, out_path=out_path, language=language,
            voice_dir=voice_dir,
            exaggeration=exaggeration, cfg_weight=cfg_weight,
            temperature=temperature, pause_ms=pause_ms,
        )
    except HTTPException:
        _safe_remove(raw_path, ref_wav, out_path)
        raise
    except Exception as e:
        _safe_remove(raw_path, ref_wav, out_path)
        raise HTTPException(500, f"Generate fail: {type(e).__name__}: {e}")

    _safe_remove(raw_path, ref_wav)

    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise HTTPException(500, "Output toiri hoy nai.")

    # response pathanor por out file cleanup (disk bhorbe na)
    return FileResponse(
        out_path,
        media_type="audio/wav",
        filename=f"cloned_{language}.wav",
        background=BackgroundTask(_safe_remove, out_path),
    )


@app.post("/api/youtube-audio")
async def youtube_audio(
    url: str = Form(..., description="YouTube video URL"),
    start: float = Form(0, description="Segment shuru (second)"),
    duration: float = Form(25, description="Segment length (second, 3-60)"),
):
    """YouTube link theke reference audio clip ber kore (WAV) — clone er jonno."""
    if not re.match(r"^https?://", url.strip()):
        raise HTTPException(400, "Thik YouTube URL daw (https:// diye shuru).")
    start = max(0.0, float(start))
    duration = max(3.0, min(float(duration), 600.0))  # up to 10 min

    job = uuid.uuid4().hex[:12]
    out_wav = os.path.join(WORK_DIR, f"{job}_yt.wav")

    try:
        _youtube_to_wav(url.strip(), start, duration, out_wav)
    except HTTPException:
        _safe_remove(out_wav)
        raise
    except Exception as e:
        _safe_remove(out_wav)
        raise HTTPException(500, f"YouTube audio fail: {type(e).__name__}: {e}")

    return FileResponse(
        out_wav,
        media_type="audio/wav",
        filename="youtube_reference.wav",
        background=BackgroundTask(_safe_remove, out_wav),
    )


@app.post("/api/train")
async def train(
    voice_name: str = Form(..., description="Human name for this voice"),
    language: str = Form(..., description="'bn' or 'en'"),
    reference: UploadFile = File(None, description="5-10 min voice (wav/mp3) — or use youtube_url"),
    youtube_url: str = Form("", description="YouTube URL (alternative to file)"),
    start: float = Form(0),
    duration: float = Form(300),
    epochs: int = Form(12),
):
    """Tomar voice e fine-tune (Chatterbox full-T3) — background job. Returns job_id + voice_id."""
    language = language if language in ("bn", "en") else "bn"
    voice_id = f"{_slug(voice_name)}-{uuid.uuid4().hex[:6]}"
    out_dir = os.path.join(VOICES_DIR, voice_id)
    os.makedirs(out_dir, exist_ok=True)
    train_audio = os.path.join(WORK_DIR, f"{voice_id}_train.wav")
    status_file = os.path.join(WORK_DIR, f"{voice_id}_status.json")

    # get audio -> normalized wav
    if reference is not None and reference.filename:
        ext = os.path.splitext(reference.filename)[1].lower()
        if ext not in ALLOWED_AUDIO:
            raise HTTPException(400, f"Audio format support kore na: {ext}.")
        raw = os.path.join(WORK_DIR, f"{voice_id}_raw{ext}")
        data = await reference.read()
        if not data:
            raise HTTPException(400, "Audio khali.")
        with open(raw, "wb") as f:
            f.write(data)
        try:
            _to_wav(raw, train_audio)
        finally:
            _safe_remove(raw)
    elif youtube_url.strip():
        _youtube_to_wav(youtube_url.strip(), max(0.0, float(start)),
                        max(30.0, min(float(duration), 900.0)), train_audio)
    else:
        raise HTTPException(400, "Voice file ba YouTube URL daw.")

    # save a short reference clip for later generation
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-i", train_audio, "-t", "25",
                        "-ac", "1", "-ar", str(TARGET_SR), os.path.join(out_dir, "reference.wav")],
                       capture_output=True)

    # launch training subprocess (headless train_voice.py)
    cmd = [sys.executable, os.path.join(HERE, "train_voice.py"),
           "--audio", train_audio, "--language", language, "--voice-id", voice_id,
           "--out-dir", out_dir, "--status-file", status_file,
           "--epochs", str(int(epochs)), "--name", voice_name]
    proc = subprocess.Popen(cmd, cwd=HERE)

    job_id = uuid.uuid4().hex[:12]
    TRAIN_JOBS[job_id] = {"status_file": status_file, "voice_id": voice_id,
                          "out_dir": out_dir, "proc": proc}
    return {"job_id": job_id, "voice_id": voice_id, "language": language}


@app.get("/api/train/status/{job_id}")
def train_status(job_id: str):
    j = TRAIN_JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "Job pawa jay nai.")
    st = {"stage": "starting", "progress_pct": 0, "message": "", "done": False, "error": None,
          "voice_id": j["voice_id"]}
    try:
        with open(j["status_file"], encoding="utf-8") as f:
            st.update(json.load(f))
    except Exception:
        pass
    alive = j["proc"].poll() is None
    st["alive"] = alive
    if not alive and not st["done"] and not st["error"]:
        rc = j["proc"].returncode
        if rc == -9:
            st["error"] = ("Out of memory (exit -9) — voice clip choto koro (5-6 min), "
                           "ba ASR_MODEL=small diye chesta koro.")
        else:
            st["error"] = f"Training process theme geche (exit {rc})."
    return st


@app.get("/api/voices")
def voices():
    """Trained voice list (VOICES_DIR e voice.json ache emon)."""
    out = []
    try:
        for d in sorted(os.listdir(VOICES_DIR)):
            vj = os.path.join(VOICES_DIR, d, "voice.json")
            if os.path.exists(vj):
                try:
                    meta = json.load(open(vj, encoding="utf-8"))
                except Exception:
                    meta = {}
                out.append({
                    "voice_id": d,
                    "name": meta.get("name") or d,
                    "language": meta.get("language", "?"),
                    "engine": meta.get("engine", "chatterbox-finetune"),
                    "has_reference": os.path.exists(os.path.join(VOICES_DIR, d, "reference.wav")),
                })
    except OSError:
        pass
    return out


@app.get("/")
def root():
    return JSONResponse({"name": "Voice Cloning API", "docs": "/docs", "health": "/api/health"})
