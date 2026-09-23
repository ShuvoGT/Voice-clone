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
import os
import shutil
import subprocess
import tempfile
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

import engines

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


@app.get("/api/health")
def health():
    import torch
    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "engines": {"english_multilingual": "xtts_v2", "bangla": "chatterbox-bangla"},
    }


@app.post("/api/clone")
async def clone(
    reference: UploadFile = File(..., description="Reference voice clip (10-30s clean)"),
    text: str = Form(..., description="Text to speak in the cloned voice"),
    language: str = Form("en", description="'bn' for Bangla, else en/hi/ar/es/fr/..."),
):
    text = (text or "").strip()
    if not text:
        raise HTTPException(400, "Text khali.")
    if len(text) > MAX_TEXT:
        raise HTTPException(400, f"Text boro (max {MAX_TEXT} chars). Chhoto kore chalao.")

    ext = os.path.splitext(reference.filename or "")[1].lower()
    if ext not in ALLOWED_AUDIO:
        raise HTTPException(400, f"Audio format support kore na: {ext}. WAV/MP3 daw.")

    job = uuid.uuid4().hex[:12]
    raw_path = os.path.join(WORK_DIR, f"{job}_raw{ext}")
    ref_wav = os.path.join(WORK_DIR, f"{job}_ref.wav")
    out_path = os.path.join(WORK_DIR, f"{job}_out.wav")

    # reference save
    data = await reference.read()
    if not data:
        raise HTTPException(400, "Reference audio khali.")
    with open(raw_path, "wb") as f:
        f.write(data)

    # normalize -> mono wav (webm/mp3/m4a shob handle hobe)
    try:
        _to_wav(raw_path, ref_wav)
    except HTTPException:
        _safe_remove(raw_path)
        raise

    # synthesize
    try:
        engines.synthesize(text=text, ref_wav=ref_wav, out_path=out_path, language=language)
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


@app.get("/")
def root():
    return JSONResponse({"name": "Voice Cloning API", "docs": "/docs", "health": "/api/health"})
