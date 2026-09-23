# Voice Cloning Web App

Nijer voice **record ba upload** koro → text likho (Bangla/English) → **cloned voice** e audio pao. Browser theke.

## Stack

| Layer | Tech |
|-------|------|
| Frontend | **React 18 + Vite + TailwindCSS** |
| Backend | **FastAPI + Uvicorn** |
| English (+16 lang) engine | **Coqui XTTS-v2** |
| Bangla engine | **Chatterbox-Bangla** (Banglabox/chatterbox-bangla-tts) |
| GPU host | **Google Colab** (tomar laptop e NVIDIA GPU nai) |

```
webapp/
├── backend/
│   ├── main.py            # FastAPI app (/api/health, /api/clone)
│   ├── engines.py         # XTTS + BanglaTTS (lazy-loaded)
│   └── requirements.txt
├── frontend/
│   ├── src/App.jsx        # pura UI (record/upload, text, generate, play, download)
│   └── ... (Vite + Tailwind config)
└── colab_backend.ipynb    # backend ke Colab GPU te chalao (public URL dey)
```

## Architecture (keno emon)

TTS model gulo **GPU** ছাড়া thik moto cholе na. Tomar ProBook e NVIDIA GPU nai. Tai:

```
[ Browser: React app (tomar laptop) ]
                │  HTTPS (POST /api/clone: reference audio + text)
                ▼
[ FastAPI backend (Colab T4 GPU) ]  ← cloudflared public URL
                │
     XTTS-v2 (en/multi)  |  Chatterbox-Bangla (bn)
```

Frontend tomar laptop e cholе, heavy inference Colab GPU te. Free.

---

## Run korar niyom

### Step A — Backend (Colab GPU)

1. `webapp/colab_backend.ipynb` Colab e open koro → Runtime → **T4 GPU**.
2. Cell run koro — code **GitHub theke auto clone** hobe (`ShuvoGT/Voice-clone`), deps install hobe.
3. Shesh cell ekta URL dibe:
   ```
   VITE_API_URL=https://xxxx-xxxx.trycloudflare.com
   ```
   Ei URL copy koro. **Cell ta chalu rakho.**

### Step B — Frontend (tomar laptop)

```bash
cd webapp/frontend
cp .env.example .env
# .env kholo, VITE_API_URL= er por Colab er URL boshao
npm install
npm run dev
```
Browser e খোলো: http://localhost:5173

### Step C — Use

1. **Record** (10-30 sec) ba **Upload** reference voice.
2. Language select koro (Bangla ba English).
3. Text likho → **Generate** → play + download.

> Same reference audio Bangla ও English dutai te dile eki voice ashbe.

---

## Fully local run (GPU thakle only)

NVIDIA GPU + CUDA thakle backend laptop e-o cholе:
```bash
cd webapp/backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
CPU te English (XTTS) khub slow, Bangla nao cholte pare — tai Colab recommended.

## Notes

- Prothom `/api/clone` call e model download+load hoy (kichu somoy)। Por er call fast.
- `colab_backend.ipynb` er tunnel URL protibar notun hoy — Colab restart hole `.env` update korte hobe।
- **Ethics:** sudhu nijer / onumoti-prapto voice. Onner voice clone kora illegal।
