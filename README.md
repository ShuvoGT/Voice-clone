# Voice Cloning (Bangla + English)

Nijer voice **record** korbe (~5 min), oi voice diye **je kono text** (Bangla ba English) **bolano** hobe — cloned voice te.

## TL;DR — kon tool kobe

| Chai | Tool | Kothay chalabe | Data lage |
|------|------|----------------|-----------|
| **English** (+ 16 lang), fast, easy | XTTS-v2 (Coqui) | Google Colab (free GPU) | 6-30 sec |
| **Bangla** zero-shot clone | Chatterbox-Bangla (Banglabox) | Google Colab | 6-30 sec |
| **Highest quality**, tomar 5-min recording diye fine-tune | GPT-SoVITS | Google Colab | 1-5 min |

> **Keno Colab?** HP ProBook 450 G10 te NVIDIA GPU nai (Intel Iris Xe). Voice models CUDA er upor cholе. Colab free te NVIDIA T4 GPU dey — training/inference oikhane fast. Result wav laptop e download korbe.
> Local CPU test korte chaile `local/` folder er script ache (slow, sudhu try korar jonno).

## Ki ki ache ei project e

```
Voice Cloning/
├── README.md                        <- ei file
├── RECORDING_GUIDE.md               <- kivabe voice record korbe (important!)
├── notebooks/
│   ├── voice_clone_colab.ipynb      <- Zero-shot clone: XTTS (En) + Bangla, just Run
│   └── gpt_sovits_train_colab.ipynb <- 5-min recording diye FULL train (best quality)
├── webapp/                          <- 🌐 Web app (React + FastAPI) — browser theke clone
│   ├── README.md                    <- webapp chalanor niyom
│   ├── frontend/                    <- React + Vite + Tailwind UI
│   ├── backend/                     <- FastAPI (XTTS + Bangla engine)
│   └── colab_backend.ipynb          <- backend Colab GPU te chalao (public URL)
├── local/
│   ├── requirements.txt
│   └── clone_xtts.py                <- laptop e English clone test (CPU, slow)
├── voices/                          <- ekhane tomar reference recording rakho (sample.wav)
└── output/                          <- generated audio ekhane jabe
```

## Kon ta use korbo?

- **Just try / quick clone** → `notebooks/voice_clone_colab.ipynb` (Colab, zero-shot)
- **Best quality (5-min train)** → `notebooks/gpt_sovits_train_colab.ipynb` (Colab, WebUI)
- **Real app, browser theke bar bar** → `webapp/` (React UI + FastAPI on Colab GPU)

## Workflow (step by step)

1. **Record** koro — `RECORDING_GUIDE.md` follow kore ekta clean `sample.wav` banao, `voices/` folder e rakho.
2. **Colab kholo** — `notebooks/voice_clone_colab.ipynb` GitHub/Drive theke Colab e open koro. (Runtime → Change runtime type → **T4 GPU**)
3. **Upload** — Colab e reference `sample.wav` upload koro.
4. **Text daw** — je kotha bolate chao (Bangla ba English) box e likho.
5. **Generate** — cell run koro, cloned voice er wav pabe, download koro.

## Repo references

- XTTS-v2 / Coqui TTS — https://github.com/coqui-ai/TTS
- Chatterbox (base, multilingual) — https://github.com/resemble-ai/chatterbox
- Chatterbox Bangla — https://huggingface.co/Banglabox/chatterbox-bangla-tts
- Comprehensive Bangla TTS — https://github.com/mobassir94/comprehensive-bangla-tts
- GPT-SoVITS — https://github.com/RVC-Boss/GPT-SoVITS
- F5-TTS (alternative) — https://github.com/SWivid/F5-TTS

## ⚠️ Ethics / Legal

Sudhu **nijer** voice, othoba jar **onumoti** ache tar voice clone koro. Onner voice permission chhara clone kora illegal/unethical (fraud, impersonation). Generated audio te consent thaka uchit.
