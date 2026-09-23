"""
TTS engines — ALL from Banglabox's vendored `chatterbox_` code (one proven env).

- MultilingualEngine : English + 20+ languages (chatterbox_.mtl_tts.ChatterboxMultilingualTTS)
- BanglaEngine       : Bangla fine-tuned (inference/infer.py -> BanglaTTS)

No pip `chatterbox-tts` (it dragged in coqui-tts -> transformers/peft conflicts).
Both engines share the same downloaded snapshot + same dependency pins.
Lazy-loaded; GPU (CUDA) if available.
"""
import functools
import os
import sys

import numpy as np
import soundfile as sf


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _find(root, filename):
    for dirpath, _dirs, files in os.walk(root):
        if filename in files:
            return dirpath
    return None


_SNAP = None


def _ensure_banglabox():
    """Banglabox snapshot download kore, `infer` ও `chatterbox_` import-able kore."""
    global _SNAP
    if _SNAP is not None:
        return _SNAP
    from huggingface_hub import snapshot_download
    snap = snapshot_download(repo_id="Banglabox/chatterbox-bangla-tts")

    infer_dir = _find(snap, "infer.py")            # .../inference
    mtl_dir = _find(snap, "mtl_tts.py")            # .../inference/src/chatterbox_

    paths = []
    if infer_dir:
        paths.append(infer_dir)                    # `import infer`
    if mtl_dir:
        paths.append(os.path.dirname(mtl_dir))     # src -> `import chatterbox_`
    for p in paths:
        if p and p not in sys.path:
            sys.path.insert(0, p)

    if not infer_dir or not mtl_dir:
        raise RuntimeError(f"Banglabox structure unexpected in {snap} "
                           f"(infer={infer_dir}, mtl={mtl_dir})")
    _SNAP = snap
    return snap


def _save(wav, out_path, sr):
    """torch tensor / numpy waveform -> WAV file."""
    import torch
    if isinstance(wav, torch.Tensor):
        w = wav.detach().cpu().float()
        if w.dim() == 1:
            w = w.unsqueeze(0)
        import torchaudio
        torchaudio.save(out_path, w, sr)
    else:
        arr = np.asarray(wav, dtype=np.float32).squeeze()
        sf.write(out_path, arr, sr)
    return out_path


# --------------------------------------------------------------------------- #
# English + multilingual  (vendored chatterbox_ multilingual)                   #
# --------------------------------------------------------------------------- #
class MultilingualEngine:
    LANG_MAP = {
        "en": "en", "hi": "hi", "ar": "ar", "es": "es", "fr": "fr", "de": "de",
        "it": "it", "pt": "pt", "ru": "ru", "zh-cn": "zh", "ja": "ja", "ko": "ko",
    }

    def __init__(self):
        self._m = None

    def _load(self):
        if self._m is None:
            _ensure_banglabox()
            from chatterbox_.mtl_tts import ChatterboxMultilingualTTS
            self._m = ChatterboxMultilingualTTS.from_pretrained(device=_device())
        return self._m

    def synthesize(self, text, ref_wav, out_path, language="en"):
        m = self._load()
        lang = self.LANG_MAP.get(language, "en")
        wav = m.generate(text, language_id=lang, audio_prompt_path=ref_wav)
        return _save(wav, out_path, m.sr)


# --------------------------------------------------------------------------- #
# Bangla  (Banglabox fine-tuned)                                               #
# --------------------------------------------------------------------------- #
class BanglaEngine:
    def __init__(self):
        self._tts = None

    def _load(self):
        if self._tts is None:
            _ensure_banglabox()
            from infer import BanglaTTS  # noqa: class from Banglabox repo
            self._tts = BanglaTTS(device=_device())
        return self._tts

    def synthesize(self, text, ref_wav, out_path, language="bn"):
        tts = self._load()
        wav, sr = tts.tts(text, ref_wav)          # (float32 waveform, sample_rate)
        return _save(wav, out_path, sr)


# --------------------------------------------------------------------------- #
# Router                                                                       #
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _multi():
    return MultilingualEngine()


@functools.lru_cache(maxsize=1)
def _bangla():
    return BanglaEngine()


def synthesize(text: str, ref_wav: str, out_path: str, language: str = "en"):
    """Pick engine by language. 'bn' -> Bangla (fine-tuned), else -> multilingual."""
    if language == "bn":
        return _bangla().synthesize(text, ref_wav, out_path, language)
    return _multi().synthesize(text, ref_wav, out_path, language)
