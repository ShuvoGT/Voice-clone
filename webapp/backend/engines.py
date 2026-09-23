"""
TTS engines for the voice-cloning web app — SINGLE stack (Chatterbox).

- MultilingualEngine : English + 20+ languages (ChatterboxMultilingualTTS), zero-shot clone.
- BanglaEngine       : Bangla (Banglabox/chatterbox-bangla-tts), zero-shot clone.

Ek-i dependency stack (chatterbox) — tai coqui-tts er moto conflict hoy na.
Both LAZY-loaded: model weights load on first use. GPU (CUDA) thakle fast.
"""
import functools
import os
import sys

import numpy as np
import soundfile as sf


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _save(wav, out_path, sr):
    """Chatterbox tensor / numpy waveform -> WAV file."""
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
# English + multilingual  (base Chatterbox multilingual)                        #
# --------------------------------------------------------------------------- #
class MultilingualEngine:
    # frontend code -> Chatterbox multilingual language_id
    LANG_MAP = {
        "en": "en", "hi": "hi", "ar": "ar", "es": "es", "fr": "fr", "de": "de",
        "it": "it", "pt": "pt", "ru": "ru", "zh-cn": "zh", "ja": "ja", "ko": "ko",
    }

    def __init__(self):
        self._m = None

    def _load(self):
        if self._m is None:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            self._m = ChatterboxMultilingualTTS.from_pretrained(device=_device())
        return self._m

    def synthesize(self, text, ref_wav, out_path, language="en"):
        m = self._load()
        lang = self.LANG_MAP.get(language, "en")
        wav = m.generate(text, language_id=lang, audio_prompt_path=ref_wav)
        return _save(wav, out_path, m.sr)


# --------------------------------------------------------------------------- #
# Bangla  (Banglabox/chatterbox-bangla-tts)                                    #
# --------------------------------------------------------------------------- #
class BanglaEngine:
    def __init__(self):
        self._tts = None

    @staticmethod
    def _find(snapshot, filename):
        for dirpath, _dirs, files in os.walk(snapshot):
            if filename in files:
                return dirpath
        return None

    def _load(self):
        if self._tts is None:
            from huggingface_hub import snapshot_download
            snapshot = snapshot_download(repo_id="Banglabox/chatterbox-bangla-tts")

            inf_dir = self._find(snapshot, "infer.py")
            if inf_dir is None:
                raise RuntimeError(f"infer.py pawa jay nai snapshot e: {snapshot}")
            if inf_dir not in sys.path:
                sys.path.insert(0, inf_dir)

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
