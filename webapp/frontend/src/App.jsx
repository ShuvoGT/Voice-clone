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
  const [tab, setTab] = useState("record"); // record | upload
  const [refBlob, setRefBlob] = useState(null);
  const [refUrl, setRefUrl] = useState(null);
  const [refName, setRefName] = useState("");
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [language, setLanguage] = useState("bn");
  const [text, setText] = useState(SAMPLE_TEXT.bn);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [resultUrl, setResultUrl] = useState(null);
  const [health, setHealth] = useState(null);

  const mediaRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);

  useEffect(() => {
    fetch(`${API_URL}/api/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: "offline" }));
  }, []);

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

  async function generate() {
    setError("");
    if (resultUrl) URL.revokeObjectURL(resultUrl); // purano result cleanup
    setResultUrl(null);
    if (!refBlob) return setError("Age reference voice record ba upload koro.");
    if (!text.trim()) return setError("Text likho.");

    setBusy(true);
    try {
      const fd = new FormData();
      const fname = refName.endsWith(".webm") ? "reference.webm" : refName;
      fd.append("reference", refBlob, fname || "reference.wav");
      fd.append("text", text);
      fd.append("language", language);

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

        {/* Step 1: reference */}
        <section className="bg-white/5 border border-white/10 rounded-2xl p-5 mb-5">
          <h2 className="font-semibold mb-3">1. Reference voice</h2>
          <div className="flex gap-2 mb-4">
            {["record", "upload"].map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`px-4 py-1.5 rounded-lg text-sm capitalize transition ${
                  tab === t ? "bg-indigo-500 text-white" : "bg-white/5 text-slate-300 hover:bg-white/10"
                }`}
              >
                {t === "record" ? "🎤 Record" : "📁 Upload"}
              </button>
            ))}
          </div>

          {tab === "record" ? (
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
          ) : (
            <label className="block">
              <input
                type="file"
                accept="audio/*"
                onChange={onUpload}
                className="block w-full text-sm text-slate-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-indigo-500 file:text-white hover:file:bg-indigo-600 cursor-pointer"
              />
            </label>
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
              {language === "bn" ? "→ Chatterbox-Bangla" : "→ XTTS-v2"}
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

        <p className="text-center text-xs text-slate-500 mt-8">
          Sudhu nijer / onumoti-prapto voice clone koro. API: <code>{API_URL}</code>
        </p>
      </div>
    </div>
  );
}
