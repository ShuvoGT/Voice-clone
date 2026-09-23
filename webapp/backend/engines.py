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
import re
import sys

import numpy as np
import soundfile as sf


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _split_sentences(text):
    """Bangla (।) + English (.!?) + newline diye sentence e bhag kore."""
    parts = re.split(r"(?<=[।!?\.])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def _to_np(wav):
    import torch
    if isinstance(wav, torch.Tensor):
        return wav.detach().cpu().float().squeeze().numpy()
    return np.asarray(wav, dtype=np.float32).squeeze()


def _concat(chunks, sr, pause_ms):
    """Numpy wav chunk gula majhe pause_ms silence diye jora dey."""
    gap = np.zeros(int(sr * pause_ms / 1000.0), dtype=np.float32)
    out = []
    for i, c in enumerate(chunks):
        if i:
            out.append(gap)
        out.append(np.asarray(c, dtype=np.float32).squeeze())
    return np.concatenate(out) if out else np.zeros(1, dtype=np.float32)


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

    def synthesize(self, text, ref_wav, out_path, language="en",
                   exaggeration=0.5, cfg_weight=0.5, temperature=0.8, pause_ms=0):
        m = self._load()
        lang = self.LANG_MAP.get(language, "en")
        gen = dict(exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature)

        def one(t):
            return _to_np(m.generate(t, language_id=lang, audio_prompt_path=ref_wav, **gen))

        # Lomba/multi-sentence text always chunk kori — nahole drift/skip/barti hoy.
        sentences = _split_sentences(text)
        if len(sentences) > 1:
            chunks = [one(s) for s in sentences]
            return _save(_concat(chunks, m.sr, pause_ms), out_path, m.sr)
        return _save(one(text), out_path, m.sr)


# --------------------------------------------------------------------------- #
# Bangla  (Banglabox fine-tuned)                                               #
# --------------------------------------------------------------------------- #
def _build_bangla_kit(snapshot):
    """infer.py root/{pretrained_models,adapter,NEW_VOCAB_SIZE.txt} ek jayga e chay,
    kintu repo te egula chorano. Symlink diye ekta 'kit' assemble kori."""
    import tempfile
    kit = os.path.join(tempfile.gettempdir(), "banglabox_kit")
    os.makedirs(kit, exist_ok=True)

    pm = _find(snapshot, "ve.safetensors")            # .../pretrained_models
    adapter = _find(snapshot, "adapter_config.json")  # .../checkpoint/adapter
    vocab = _find(snapshot, "NEW_VOCAB_SIZE.txt")     # .../checkpoint
    if not (pm and adapter and vocab):
        raise RuntimeError(f"Kit parts missing: pm={pm}, adapter={adapter}, vocab={vocab}")

    def link(src, dst):
        if os.path.lexists(dst):
            return
        try:
            os.symlink(src, dst)
        except OSError:
            import shutil
            (shutil.copytree if os.path.isdir(src) else shutil.copy)(src, dst)

    link(pm, os.path.join(kit, "pretrained_models"))
    link(adapter, os.path.join(kit, "adapter"))
    link(os.path.join(vocab, "NEW_VOCAB_SIZE.txt"), os.path.join(kit, "NEW_VOCAB_SIZE.txt"))
    return kit


class BanglaEngine:
    def __init__(self):
        self._tts = None

    def _load(self):
        if self._tts is None:
            snapshot = _ensure_banglabox()
            kit = _build_bangla_kit(snapshot)
            from infer import BanglaTTS  # noqa: class from Banglabox repo
            self._tts = BanglaTTS(root=kit, device=_device())
        return self._tts

    def synthesize(self, text, ref_wav, out_path, language="bn",
                   exaggeration=0.5, cfg_weight=0.5, temperature=0.8, pause_ms=0):
        tts = self._load()
        # infer.GEN (module-level locked params) override kori emotion/pace er jonno
        import infer
        infer.GEN.update(exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature)

        # Lomba/multi-sentence text always chunk kori — nahole drift/skip/barti hoy.
        sentences = _split_sentences(text)
        if len(sentences) > 1:
            chunks, sr = [], None
            for s in sentences:
                w, sr = tts.tts(s, ref_wav)
                chunks.append(_to_np(w))
            return _save(_concat(chunks, sr, pause_ms), out_path, sr)

        wav, sr = tts.tts(text, ref_wav)
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


def synthesize(text: str, ref_wav: str, out_path: str, language: str = "en", **params):
    """Pick engine by language. 'bn' -> Bangla (fine-tuned), else -> multilingual.

    params: exaggeration, cfg_weight, temperature, pause_ms
    """
    engine = _bangla() if language == "bn" else _multi()
    return engine.synthesize(text, ref_wav, out_path, language, **params)
