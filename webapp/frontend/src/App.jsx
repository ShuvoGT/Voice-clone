import React, { useState, useRef, useEffect } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const LANGUAGES = [
  { code: "bn", label: "বাংলা (Bangla)" },
  { code: "en", label: "English" },
  { code: "hi", label: "हिन्दी (Hindi)" },
  { code: "ar", label: "العربية (Arabic)" },
  { code: "es", label: "Español" },
  { code: "fr", label: "Français" },
  { code: "de", label: "Deutsch" },
  { code: "it", label: "Italiano" },
  { code: "pt", label: "Português" },
  { code: "ru", label: "Русский" },
  { code: "zh-cn", label: "中文" },
  { code: "ja", label: "日本語" },
  { code: "ko", label: "한국어" },
];

const SAMPLE_TEXT = {
  bn: "নমস্কার, এটা আমার ক্লোন করা কণ্ঠস্বর। আপনি যে লেখা দেবেন আমি সেটাই পড়ে শোনাবো।",
  en: "Hello, this is my cloned voice. I will read whatever text you give me.",
};

export default function App() {
  const [tab, setTab] = useState("record"); // record | upload | youtube
  const [ytUrl, setYtUrl] = useState("");
  const [ytStart, setYtStart] = useState(0);
  const [ytDur, setYtDur] = useState(25);
  const [ytBusy, setYtBusy] = useState(false);
  const [refBlob, setRefBlob] = useState(null);
  const [refUrl, setRefUrl] = useState(null);
  const [refName, setRefName] = useState("");
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [language, setLanguage] = useState("bn");
  const [text, setText] = useState(SAMPLE_TEXT.bn);
  const [exaggeration, setExaggeration] = useState(0.5);
  const [cfgWeight, setCfgWeight] = useState(0.5);
  const [temperature, setTemperature] = useState(0.8);
  const [pauseMs, setPauseMs] = useState(0);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [resultUrl, setResultUrl] = useState(null);
  const [health, setHealth] = useState(null);

  // dashboard: view + trained voices
  const [view, setView] = useState("generate"); // generate | train
  const [voices, setVoices] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState(""); // voice_id for generation ("" = zero-shot)

  // train form
  const [trainName, setTrainName] = useState("");
  const [trainLang, setTrainLang] = useState("bn");
  const [trainSrc, setTrainSrc] = useState("upload"); // upload | youtube
  const [trainFile, setTrainFile] = useState(null);
  const [trainYt, setTrainYt] = useState("");
  const [trainStatus, setTrainStatus] = useState(null);
  const [training, setTraining] = useState(false);

  const mediaRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);
  const pollRef = useRef(null);

  useEffect(() => {
    fetch(`${API_URL}/api/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: "offline" }));
    refreshVoices();
  }, []);

  function refreshVoices() {
    fetch(`${API_URL}/api/voices`)
      .then((r) => r.json())
      .then((v) => setVoices(Array.isArray(v) ? v : []))
      .catch(() => {});
  }

  async function startTraining() {
    setError("");
    if (!trainName.trim()) return setError("Voice er ekta naam dao.");
    if (trainSrc === "upload" && !trainFile) return setError("Voice file (5-10 min) upload koro.");
    if (trainSrc === "youtube" && !/^https?:\/\//.test(trainYt.trim()))
      return setError("Thik YouTube URL dao.");
    setTraining(true);
    setTrainStatus({ stage: "starting", progress_pct: 0, message: "Shuru hocche..." });
    try {
      const fd = new FormData();
      fd.append("voice_name", trainName.trim());
      fd.append("language", trainLang);
      if (trainSrc === "upload") fd.append("reference", trainFile, trainFile.name);
      else {
        fd.append("youtube_url", trainYt.trim());
        fd.append("duration", "600");
      }
      const res = await fetch(`${API_URL}/api/train`, { method: "POST", body: fd });
      if (!res.ok) throw new Error((await res.text()) || `HTTP ${res.status}`);
      const { job_id } = await res.json();
      pollTraining(job_id);
    } catch (e) {
      setError("Training start fail: " + e.message);
      setTraining(false);
      setTrainStatus(null);
    }
  }

  function pollTraining(jobId) {
    clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const st = await fetch(`${API_URL}/api/train/status/${jobId}`).then((r) => r.json());
        setTrainStatus(st);
        if (st.done || st.error) {
          clearInterval(pollRef.current);
          setTraining(false);
          refreshVoices();
        }
      } catch {
        /* keep polling */
      }
    }, 5000);
  }

  function setReference(blob, name) {
    if (refUrl) URL.revokeObjectURL(refUrl);
    setRefBlob(blob);
    setRefUrl(URL.createObjectURL(blob));
    setRefName(name);
  }

  async function startRecording() {
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => e.data.size && chunksRef.current.push(e.data);
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        setReference(blob, "recording.webm");
        stream.getTracks().forEach((t) => t.stop());
      };
      mr.start();
      mediaRef.current = mr;
      setRecording(true);
      setSeconds(0);
      timerRef.current = setInterval(() => setSeconds((s) => s + 1), 1000);
    } catch (e) {
      setError("Mic access dorkar. Browser permission daw. (" + e.message + ")");
    }
  }

  function stopRecording() {
    mediaRef.current?.stop();
    setRecording(false);
    clearInterval(timerRef.current);
  }

  function onUpload(e) {
    const f = e.target.files?.[0];
    if (f) setReference(f, f.name);
  }

  async function fetchYoutube() {
    setError("");
    if (!/^https?:\/\//.test(ytUrl.trim())) {
      return setError("Thik YouTube URL daw (https:// diye).");
    }
    setYtBusy(true);
    try {
      const fd = new FormData();
      fd.append("url", ytUrl.trim());
      fd.append("start", String(ytStart || 0));
      fd.append("duration", String(ytDur || 25));
      const res = await fetch(`${API_URL}/api/youtube-audio`, { method: "POST", body: fd });
      if (!res.ok) throw new Error((await res.text()) || `HTTP ${res.status}`);
      const blob = await res.blob();
      setReference(blob, "youtube_reference.wav");
    } catch (e) {
      setError("YouTube audio fail: " + e.message);
    } finally {
      setYtBusy(false);
    }
  }

  async function generate() {
    setError("");
    if (resultUrl) URL.revokeObjectURL(resultUrl); // purano result cleanup
    setResultUrl(null);
    if (!refBlob && !selectedVoice)
      return setError("Reference voice dao, ba ekta Trained voice select koro.");
    if (!text.trim()) return setError("Text likho.");

    setBusy(true);
    try {
      const fd = new FormData();
      if (refBlob) {
        const fname = refName.endsWith(".webm") ? "reference.webm" : refName;
        fd.append("reference", refBlob, fname || "reference.wav");
      }
      if (selectedVoice) fd.append("voice_id", selectedVoice);
      fd.append("text", text);
      fd.append("language", language);
      fd.append("exaggeration", String(exaggeration));
      fd.append("cfg_weight", String(cfgWeight));
      fd.append("temperature", String(temperature));
      fd.append("pause_ms", String(pauseMs));

      const res = await fetch(`${API_URL}/api/clone`, { method: "POST", body: fd });
      if (!res.ok) {
        const msg = await res.text();
        throw new Error(msg || `HTTP ${res.status}`);
      }
      const blob = await res.blob();
      setResultUrl(URL.createObjectURL(blob));
    } catch (e) {
      setError("Generate fail: " + e.message);
    } finally {
      setBusy(false);
    }
  }

  const online = health?.status === "ok";
  const mm = String(Math.floor(seconds / 60)).padStart(2, "0");
  const ss = String(seconds % 60).padStart(2, "0");

  return (
    <div className="min-h-full bg-gradient-to-br from-slate-900 via-slate-800 to-indigo-950 text-slate-100">
      <div className="max-w-2xl mx-auto px-4 py-10">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">🎙️ Voice Cloning</h1>
            <p className="text-slate-400 text-sm mt-1">Bangla + English · nijer voice diye text bolao</p>
          </div>
          <span
            className={`text-xs px-3 py-1 rounded-full border ${
              online
                ? "bg-emerald-500/10 border-emerald-500/40 text-emerald-300"
                : "bg-rose-500/10 border-rose-500/40 text-rose-300"
            }`}
          >
            {online ? `● GPU: ${health.device}` : "● API offline"}
          </span>
        </div>

        {/* Nav */}
        <div className="flex gap-2 mb-6">
          {[["generate", "🎙️ Generate"], ["train", "🎓 Train Voice"]].map(([v, l]) => (
            <button
              key={v}
              onClick={() => setView(v)}
              className={`px-4 py-2 rounded-xl text-sm font-medium transition ${
                view === v ? "bg-indigo-500 text-white" : "bg-white/5 text-slate-300 hover:bg-white/10"
              }`}
            >
              {l}
            </button>
          ))}
        </div>

        {view === "generate" && (
        <>
        {/* Voice source: zero-shot or a trained voice */}
        <section className="bg-white/5 border border-white/10 rounded-2xl p-5 mb-5">
          <h2 className="font-semibold mb-3">Voice source</h2>
          <select
            value={selectedVoice}
            onChange={(e) => {
              const v = e.target.value;
              setSelectedVoice(v);
              const vc = voices.find((x) => x.voice_id === v);
              if (vc) {
                setLanguage(vc.language);
                if (SAMPLE_TEXT[vc.language]) setText(SAMPLE_TEXT[vc.language]);
              }
            }}
            className="w-full bg-slate-800 border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            <option value="">Zero-shot (reference clip diye)</option>
            {voices.map((v) => (
              <option key={v.voice_id} value={v.voice_id}>
                🎓 {v.name} ({v.language}) — trained
              </option>
            ))}
          </select>
          <p className="text-xs text-slate-500 mt-2">
            {selectedVoice
              ? "Trained voice — high similarity. Reference optional (na dile trained reference use hobe)."
              : "Zero-shot — niche reference voice dao. Ba Train tab e nijer voice train koro."}
          </p>
        </section>

        {/* Step 1: reference */}
        <section className="bg-white/5 border border-white/10 rounded-2xl p-5 mb-5">
          <h2 className="font-semibold mb-3">1. Reference voice {selectedVoice && <span className="text-xs text-slate-400">(optional)</span>}</h2>
          <div className="flex gap-2 mb-4">
            {[
              ["record", "🎤 Record"],
              ["upload", "📁 Upload"],
              ["youtube", "🎬 YouTube"],
            ].map(([t, lbl]) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-4 py-1.5 rounded-lg text-sm transition ${
                  tab === t ? "bg-indigo-500 text-white" : "bg-white/5 text-slate-300 hover:bg-white/10"
                }`}
              >
                {lbl}
              </button>
            ))}
          </div>

          {tab === "record" && (
            <div className="flex items-center gap-4">
              {!recording ? (
                <button
                  onClick={startRecording}
                  className="px-5 py-2.5 bg-rose-500 hover:bg-rose-600 rounded-xl font-medium transition"
                >
                  ● Start recording
                </button>
              ) : (
                <button
                  onClick={stopRecording}
                  className="px-5 py-2.5 bg-slate-200 text-slate-900 hover:bg-white rounded-xl font-medium transition animate-pulse"
                >
                  ■ Stop ({mm}:{ss})
                </button>
              )}
              <span className="text-xs text-slate-400">10–30 sec clean audio-i enough</span>
            </div>
          )}

          {tab === "upload" && (
            <label className="block">
              <input
                type="file"
                accept="audio/*"
                onChange={onUpload}
                className="block w-full text-sm text-slate-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-indigo-500 file:text-white hover:file:bg-indigo-600 cursor-pointer"
              />
            </label>
          )}

          {tab === "youtube" && (
            <div className="space-y-3">
              <input
                type="url"
                value={ytUrl}
                onChange={(e) => setYtUrl(e.target.value)}
                placeholder="https://www.youtube.com/watch?v=..."
                className="w-full bg-slate-800/70 border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
              <div className="flex gap-3 items-center text-sm">
                <label className="flex items-center gap-1 text-slate-400">
                  Start (s)
                  <input
                    type="number"
                    min={0}
                    value={ytStart}
                    onChange={(e) => setYtStart(Number(e.target.value))}
                    className="w-20 bg-slate-800 border border-white/10 rounded px-2 py-1"
                  />
                </label>
                <label className="flex items-center gap-1 text-slate-400">
                  Length (s)
                  <input
                    type="number"
                    min={3}
                    max={600}
                    value={ytDur}
                    onChange={(e) => setYtDur(Number(e.target.value))}
                    className="w-20 bg-slate-800 border border-white/10 rounded px-2 py-1"
                  />
                </label>
              </div>
              <button
                onClick={fetchYoutube}
                disabled={ytBusy || !online}
                className="px-5 py-2.5 bg-rose-500 hover:bg-rose-600 disabled:opacity-40 rounded-xl font-medium transition"
              >
                {ytBusy ? "⏳ Audio ana hocche..." : "🎬 Fetch audio"}
              </button>
              <p className="text-xs text-slate-500">
                Video theke {ytDur}s clip nibe (start {ytStart}s theke). Sudhu nijer / onumoti-prapto content.
              </p>
            </div>
          )}

          {refUrl && (
            <div className="mt-4">
              <p className="text-xs text-slate-400 mb-1">Reference: {refName}</p>
              <audio src={refUrl} controls className="w-full" />
            </div>
          )}
        </section>

        {/* Step 2: language + text */}
        <section className="bg-white/5 border border-white/10 rounded-2xl p-5 mb-5">
          <h2 className="font-semibold mb-3">2. Text & language</h2>
          <div className="flex gap-3 mb-3">
            <select
              value={language}
              onChange={(e) => {
                setLanguage(e.target.value);
                if (SAMPLE_TEXT[e.target.value]) setText(SAMPLE_TEXT[e.target.value]);
              }}
              className="bg-slate-800 border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            >
              {LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.label}
                </option>
              ))}
            </select>
            <span className="text-xs text-slate-500 self-center">
              {language === "bn" ? "→ Chatterbox-Bangla" : "→ Chatterbox multilingual"}
            </span>
          </div>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={4}
            maxLength={1000}
            placeholder="Ekhane text likho..."
            className="w-full bg-slate-800/70 border border-white/10 rounded-xl p-3 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
          <div className="text-right text-xs text-slate-500 mt-1">{text.length}/1000</div>
        </section>

        {/* Advanced controls: emotion, pace, pauses */}
        <section className="bg-white/5 border border-white/10 rounded-2xl p-5 mb-5">
          <button
            onClick={() => setShowAdvanced((v) => !v)}
            className="w-full flex items-center justify-between font-semibold"
          >
            <span>3. Advanced — emotion & pacing</span>
            <span className="text-slate-400 text-sm">{showAdvanced ? "▲" : "▼"}</span>
          </button>

          {showAdvanced && (
            <div className="mt-4 space-y-4">
              <Slider
                label="🎭 Emotion (exaggeration)"
                hint="Beshi = beshi expressive/emotional"
                min={0} max={1.2} step={0.05}
                value={exaggeration} onChange={setExaggeration}
              />
              <Slider
                label="🐢 Pace (cfg weight)"
                hint="Kom = slower + beshi natural emotion"
                min={0.2} max={1} step={0.05}
                value={cfgWeight} onChange={setCfgWeight}
              />
              <Slider
                label="🎲 Variation (temperature)"
                hint="Beshi = beshi life, kom stable"
                min={0.1} max={1.5} step={0.05}
                value={temperature} onChange={setTemperature}
              />
              <Slider
                label="⏸️ Line gap (pause)"
                hint="Sentence-er majhe silence (ms). 0 = off"
                min={0} max={1500} step={50} unit="ms"
                value={pauseMs} onChange={(v) => setPauseMs(Math.round(v))}
              />
              <div className="flex gap-2 flex-wrap pt-1">
                <Preset label="Natural" onClick={() => { setExaggeration(0.5); setCfgWeight(0.5); setTemperature(0.8); setPauseMs(0); }} />
                <Preset label="Emotional" onClick={() => { setExaggeration(0.9); setCfgWeight(0.3); setTemperature(0.9); setPauseMs(250); }} />
                <Preset label="Calm / news" onClick={() => { setExaggeration(0.35); setCfgWeight(0.6); setTemperature(0.6); setPauseMs(150); }} />
              </div>
            </div>
          )}
        </section>

        {/* Step 3: generate */}
        <button
          onClick={generate}
          disabled={busy || !online}
          className="w-full py-3.5 rounded-2xl font-semibold text-lg bg-indigo-500 hover:bg-indigo-600 disabled:opacity-40 disabled:cursor-not-allowed transition"
        >
          {busy ? "⏳ Generating... (kichukhon)" : "✨ Generate cloned voice"}
        </button>

        {error && (
          <div className="mt-4 bg-rose-500/10 border border-rose-500/40 text-rose-200 text-sm rounded-xl p-3">
            {error}
          </div>
        )}

        {resultUrl && (
          <section className="mt-6 bg-emerald-500/10 border border-emerald-500/30 rounded-2xl p-5">
            <h2 className="font-semibold mb-3 text-emerald-200">✅ Result</h2>
            <audio src={resultUrl} controls autoPlay className="w-full mb-3" />
            <a
              href={resultUrl}
              download={`cloned_${language}.wav`}
              className="inline-block px-4 py-2 bg-emerald-500 hover:bg-emerald-600 rounded-lg text-sm font-medium"
            >
              ⬇ Download WAV
            </a>
          </section>
        )}
        </>
        )}

        {view === "train" && (
        <>
          <section className="bg-white/5 border border-white/10 rounded-2xl p-5 mb-5">
            <h2 className="font-semibold mb-1">🎓 Train your voice</h2>
            <p className="text-xs text-slate-400 mb-4">
              5-10 min voice dao → fine-tune → high-similarity voice. Bangla + English.
              <br />⏱️ ~2-3 ghonta lagbe (T4)। Colab tab + ei page khola rakho।
            </p>

            <label className="block text-sm mb-1 text-slate-300">Voice er naam</label>
            <input
              value={trainName}
              onChange={(e) => setTrainName(e.target.value)}
              placeholder="jemon: Amar Bangla Voice"
              className="w-full bg-slate-800 border border-white/10 rounded-lg px-3 py-2 text-sm mb-4 focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />

            <label className="block text-sm mb-1 text-slate-300">Language</label>
            <select
              value={trainLang}
              onChange={(e) => setTrainLang(e.target.value)}
              className="w-full bg-slate-800 border border-white/10 rounded-lg px-3 py-2 text-sm mb-4"
            >
              <option value="bn">বাংলা (Bangla)</option>
              <option value="en">English</option>
            </select>

            <div className="flex gap-2 mb-3">
              {[["upload", "📁 Upload"], ["youtube", "🎬 YouTube"]].map(([t, l]) => (
                <button
                  key={t}
                  onClick={() => setTrainSrc(t)}
                  className={`px-4 py-1.5 rounded-lg text-sm transition ${
                    trainSrc === t ? "bg-indigo-500 text-white" : "bg-white/5 text-slate-300 hover:bg-white/10"
                  }`}
                >
                  {l}
                </button>
              ))}
            </div>

            {trainSrc === "upload" ? (
              <input
                type="file"
                accept="audio/*"
                onChange={(e) => setTrainFile(e.target.files?.[0] || null)}
                className="block w-full text-sm text-slate-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-indigo-500 file:text-white cursor-pointer mb-2"
              />
            ) : (
              <input
                type="url"
                value={trainYt}
                onChange={(e) => setTrainYt(e.target.value)}
                placeholder="https://www.youtube.com/watch?v=... (tomar/onumoti-prapto voice)"
                className="w-full bg-slate-800 border border-white/10 rounded-lg px-3 py-2 text-sm mb-2 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              />
            )}

            <button
              onClick={startTraining}
              disabled={training || !online}
              className="w-full mt-3 py-3 rounded-2xl font-semibold bg-indigo-500 hover:bg-indigo-600 disabled:opacity-40 transition"
            >
              {training ? "⏳ Training cholche..." : "🚀 Start training"}
            </button>

            {trainStatus && (
              <div className="mt-4">
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>{trainStatus.stage}</span>
                  <span>{trainStatus.progress_pct ?? 0}%</span>
                </div>
                <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${trainStatus.error ? "bg-rose-500" : "bg-indigo-500"}`}
                    style={{ width: `${trainStatus.progress_pct ?? 0}%` }}
                  />
                </div>
                <p className="text-xs text-slate-400 mt-1">{trainStatus.message}</p>
                {trainStatus.done && <p className="text-emerald-300 text-sm mt-1">✅ Ready! Generate tab e voice-ta pabe.</p>}
                {trainStatus.error && <p className="text-rose-300 text-sm mt-1">❌ {trainStatus.error}</p>}
              </div>
            )}
          </section>

          <section className="bg-white/5 border border-white/10 rounded-2xl p-5 mb-5">
            <div className="flex items-center justify-between mb-3">
              <h2 className="font-semibold">Trained voices</h2>
              <button onClick={refreshVoices} className="text-xs text-indigo-300 hover:text-indigo-200">↻ Refresh</button>
            </div>
            {voices.length === 0 ? (
              <p className="text-sm text-slate-500">Ekhono kono trained voice nai.</p>
            ) : (
              <ul className="space-y-2">
                {voices.map((v) => (
                  <li key={v.voice_id} className="flex items-center justify-between bg-white/5 rounded-lg px-3 py-2 text-sm">
                    <span>🎓 {v.name} <span className="text-slate-500">({v.language})</span></span>
                    <button
                      onClick={() => { setSelectedVoice(v.voice_id); setLanguage(v.language); setView("generate"); }}
                      className="text-xs px-3 py-1 bg-indigo-500 hover:bg-indigo-600 rounded-lg"
                    >
                      Use →
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>

          {error && (
            <div className="mt-2 bg-rose-500/10 border border-rose-500/40 text-rose-200 text-sm rounded-xl p-3">
              {error}
            </div>
          )}
        </>
        )}

        <p className="text-center text-xs text-slate-500 mt-8">
          Sudhu nijer / onumoti-prapto voice clone koro. API: <code>{API_URL}</code>
        </p>
      </div>
    </div>
  );
}

function Slider({ label, hint, min, max, step, value, onChange, unit = "" }) {
  return (
    <div>
      <div className="flex items-center justify-between text-sm mb-1">
        <span className="text-slate-200">{label}</span>
        <span className="text-indigo-300 tabular-nums">
          {value}
          {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-indigo-500"
      />
      <p className="text-xs text-slate-500 mt-0.5">{hint}</p>
    </div>
  );
}

function Preset({ label, onClick }) {
  return (
    <button
      onClick={onClick}
      className="px-3 py-1.5 rounded-lg text-xs bg-white/5 hover:bg-white/10 border border-white/10 text-slate-200 transition"
    >
      {label}
    </button>
  );
}
