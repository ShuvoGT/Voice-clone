"""
TTS engines for the voice-cloning web app.

- XTTSEngine     : English + 16 languages (Coqui XTTS-v2), zero-shot clone.
- BanglaEngine   : Bangla (Banglabox/chatterbox-bangla-tts), zero-shot clone.

Both are LAZY-loaded: model weights load on first use, not at import.
GPU (CUDA) thakle fast; CPU te cholе kintu slow.
"""
import os
import sys
import functools

import numpy as np
import soundfile as sf


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# XTTS-v2  (English + multilingual)                                            #
# --------------------------------------------------------------------------- #
class XTTSEngine:
    # XTTS-v2 supported languages
    LANGS = {"en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
             "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"}

    def __init__(self):
        self._tts = None

    def _load(self):
        if self._tts is None:
            os.environ.setdefault("COQUI_TOS_AGREED", "1")
            from TTS.api import TTS
            self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(_device())
        return self._tts

    def synthesize(self, text: str, ref_wav: str, out_path: str, language: str = "en"):
        if language not in self.LANGS:
            language = "en"
        tts = self._load()
        tts.tts_to_file(text=text, speaker_wav=ref_wav, language=language, file_path=out_path)
        return out_path


# --------------------------------------------------------------------------- #
# Bangla  (Banglabox/chatterbox-bangla-tts)                                    #
# --------------------------------------------------------------------------- #
class BanglaEngine:
    def __init__(self):
        self._tts = None
        self._root = None

    @staticmethod
    def _find(snapshot, filename):
        """snapshot er moddhe je folder e `filename` ache seta khuje ber kore."""
        for dirpath, _dirs, files in os.walk(snapshot):
            if filename in files:
                return dirpath
        return None

    def _load(self):
        if self._tts is None:
            from huggingface_hub import snapshot_download
            snapshot = snapshot_download(repo_id="Banglabox/chatterbox-bangla-tts")
            self._root = snapshot

            # infer.py kothay ache khuji (inference/ ba root — version onujayi)
            inf_dir = self._find(snapshot, "infer.py")
            if inf_dir is None:
                raise RuntimeError(
                    f"infer.py pawa jay nai snapshot e: {snapshot}. "
                    "Banglabox repo structure bodleche hote pare."
                )
            if inf_dir not in sys.path:
                sys.path.insert(0, inf_dir)

            from infer import BanglaTTS  # noqa: class from Banglabox repo
            # root override kori NA — BanglaTTS er default (HERE = infer.py er folder)
            # snapshot er relative structure onujayi thik adapter/tokenizer khuje ney.
            self._tts = BanglaTTS(device=_device())
        return self._tts

    def synthesize(self, text: str, ref_wav: str, out_path: str, language: str = "bn"):
        tts = self._load()
        wav, sr = tts.tts(text, ref_wav)          # returns (float32 waveform, sample_rate)
        wav = np.asarray(wav, dtype=np.float32).squeeze()
        sf.write(out_path, wav, sr)
        return out_path


# --------------------------------------------------------------------------- #
# Router                                                                       #
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _xtts():
    return XTTSEngine()


@functools.lru_cache(maxsize=1)
def _bangla():
    return BanglaEngine()


def synthesize(text: str, ref_wav: str, out_path: str, language: str = "en"):
    """Pick engine by language.  'bn' -> Bangla, everything else -> XTTS."""
    if language == "bn":
        return _bangla().synthesize(text, ref_wav, out_path, language)
    return _xtts().synthesize(text, ref_wav, out_path, language)
