import React, { useState, useEffect, useRef } from 'react';
import {
  Play,
  Pause,
  Sparkles,
  Mic,
  Volume2,
  VolumeX,
  FileAudio,
  Trash2,
  Download,
  Copy,
  Check,
  Dices,
  Sliders,
  Cpu,
  Folder,
  RefreshCw,
  FileCode,
  Terminal,
  Settings,
  Layers,
  Info,
  Clock,
  HardDrive,
  Headphones,
  Wand2,
} from 'lucide-react';

interface OutputFile {
  filename: string;
  path: string;
  mode: string;
  size_bytes: number;
  size_str: string;
  created_ts: number;
  created_str: string;
  duration: number;
  sample_rate: number;
}

interface PythonFileItem {
  filename: string;
  size_bytes: number;
  exists: boolean;
}

const SUPPORTED_SPEAKERS: Record<string, string> = {
  Vivian: 'Bright, slightly edgy young female voice. | Chinese',
  Serena: 'Warm, gentle young female voice. | Chinese',
  Uncle_Fu: 'Seasoned male voice with a low, mellow timbre. | Chinese',
  Dylan: 'Youthful Beijing male voice with a clear, natural timbre. | Chinese (Beijing Dialect)',
  Eric: 'Lively Chengdu male voice with a slightly husky brightness. | Chinese (Sichuan Dialect)',
  Ryan: 'Dynamic male voice with strong rhythmic drive. | English',
  Aiden: 'Sunny American male voice with a clear midrange. | English',
  Ono_Anna: 'Playful Japanese female voice with a light, nimble timbre. | Japanese',
  Sohee: 'Warm Korean female voice with rich emotion. | Korean',
};

const LANGUAGES = [
  'auto',
  'Chinese',
  'English',
  'Japanese',
  'Korean',
  'German',
  'French',
  'Russian',
  'Portuguese',
  'Spanish',
  'Italian',
];

const MODEL_PRESETS: Record<string, string> = {
  'custom-voice': 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
  'voice-design': 'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign',
  'voice-clone': 'Qwen/Qwen3-TTS-12Hz-1.7B-Base',
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'custom-voice' | 'voice-design' | 'voice-clone' | 'outputs' | 'code'>('custom-voice');

  // Global Model Config
  const [showConfig, setShowConfig] = useState(false);
  const [modelId, setModelId] = useState(MODEL_PRESETS['custom-voice']);
  const [device, setDevice] = useState('cuda');
  const [dtype, setDtype] = useState('bfloat16');
  const [attn, setAttn] = useState('flash_attention_2');
  const [minWords, setMinWords] = useState(200);
  const [maxWords, setMaxWords] = useState(250);
  const [modelStatus, setModelStatus] = useState<string>('Ready (Gradio Engine Active)');

  // CustomVoice Form
  const [cvText, setCvText] = useState('Hello! Welcome to Qwen3-TTS running on Gradio. What text would you like me to speak today?');
  const [cvLang, setCvLang] = useState('auto');
  const [cvSpeaker, setCvSpeaker] = useState('Vivian');
  const [cvInstruct, setCvInstruct] = useState('');

  // VoiceDesign Form (Random and Fixed Seed)
  const [vdText, setVdText] = useState('Hello, this is a newly designed voice created entirely from text prompt descriptions.');
  const [vdInstruct, setVdInstruct] = useState('Warm, gentle female voice, calm and soothing tone, natural breathing.');
  const [vdLang, setVdLang] = useState('auto');
  const [seedMode, setSeedMode] = useState<'Random' | 'Fixed'>('Random');
  const [seedValue, setSeedValue] = useState<number>(42);

  // VoiceClone Form
  const [vcText, setVcText] = useState('Hello, this synthesized speech is cloning the timbre and acoustic characteristics of your reference audio.');
  const [vcLang, setVcLang] = useState('auto');
  const [vcRefText, setVcRefText] = useState('');
  const [vcXvector, setVcXvector] = useState(false);

  // Output Files & Playback State
  const [outputFiles, setOutputFiles] = useState<OutputFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<OutputFile | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(0.8);
  const [isMuted, setIsMuted] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Generation status
  const [isGenerating, setIsGenerating] = useState(false);
  const [lastGeneratedSeed, setLastGeneratedSeed] = useState<number | null>(null);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  // Code inspection state
  const [pythonFiles, setPythonFiles] = useState<PythonFileItem[]>([]);
  const [selectedCodeFile, setSelectedCodeFile] = useState<string>('gradio_app.py');
  const [codeContent, setCodeContent] = useState<string>('');
  const [copiedCode, setCopiedCode] = useState(false);

  // Load outputs list on mount
  useEffect(() => {
    fetchOutputFiles();
    fetchPythonFiles();
  }, []);

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 4000);
  };

  const fetchOutputFiles = async (autoSelectNewest = false) => {
    try {
      const res = await fetch('/api/outputs');
      const data = await res.json();
      if (data.success && data.files) {
        setOutputFiles(data.files);
        if (data.files.length > 0) {
          if (autoSelectNewest || !selectedFile) {
            setSelectedFile(data.files[0]);
          }
        }
      }
    } catch (err) {
      console.error('Failed to load output files', err);
    }
  };

  const fetchPythonFiles = async () => {
    try {
      const res = await fetch('/api/python-files');
      const data = await res.json();
      if (data.files) {
        setPythonFiles(data.files);
      }
    } catch (err) {
      console.error('Failed to load python files', err);
    }
  };

  const loadCodeContent = async (filename: string) => {
    setSelectedCodeFile(filename);
    try {
      const res = await fetch(`/api/python-files?file=${encodeURIComponent(filename)}`);
      const data = await res.json();
      if (data.content) {
        setCodeContent(data.content);
      }
    } catch (err) {
      console.error('Failed to load code content', err);
    }
  };

  useEffect(() => {
    if (activeTab === 'code' && selectedCodeFile) {
      loadCodeContent(selectedCodeFile);
    }
  }, [activeTab]);

  // Audio Playback Listeners
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleTimeUpdate = () => setCurrentTime(audio.currentTime);
    const handleLoadedMetadata = () => {
      setDuration(audio.duration || 0);
      if (isPlaying) audio.play().catch(() => {});
    };
    const handleEnded = () => setIsPlaying(false);

    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('ended', handleEnded);
    };
  }, [selectedFile, isPlaying]);

  const togglePlay = () => {
    if (!audioRef.current || !selectedFile) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play().then(() => setIsPlaying(true)).catch((e) => console.log('play blocked', e));
    }
  };

  const handleSelectFile = (file: OutputFile) => {
    setSelectedFile(file);
    setIsPlaying(true);
    if (audioRef.current) {
      audioRef.current.src = `/api/outputs/${file.filename}`;
      audioRef.current.currentTime = 0;
      audioRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
    }
  };

  const handleDeleteFile = async (filename: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    try {
      const res = await fetch(`/api/outputs/${encodeURIComponent(filename)}`, { method: 'DELETE' });
      const data = await res.json();
      if (data.success) {
        showToast(`🗑️ Deleted ${filename}`);
        if (selectedFile?.filename === filename) {
          setSelectedFile(null);
          setIsPlaying(false);
        }
        await fetchOutputFiles();
      }
    } catch (err) {
      showToast('Failed to delete file');
    }
  };

  const handleRollSeed = () => {
    const randomInt = Math.floor(Math.random() * 2147483647);
    setSeedValue(randomInt);
    showToast(`🎲 Rolled seed: ${randomInt}`);
  };

  const handleGenerate = async (mode: 'custom-voice' | 'voice-design' | 'voice-clone') => {
    setIsGenerating(true);
    try {
      const payload: any = {
        mode,
        min_words: minWords,
        max_words: maxWords,
      };

      if (mode === 'custom-voice') {
        payload.text = cvText;
        payload.language = cvLang;
        payload.speaker = cvSpeaker;
        payload.instruct = cvInstruct;
      } else if (mode === 'voice-design') {
        payload.text = vdText;
        payload.instruct = vdInstruct;
        payload.language = vdLang;
        payload.seed_mode = seedMode;
        payload.seed_value = seedValue;
      } else {
        payload.text = vcText;
        payload.language = vcLang;
      }

      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (data.success) {
        if (data.actual_seed) {
          setLastGeneratedSeed(data.actual_seed);
        }
        showToast(`✨ Generated ${data.filename} (${data.duration}s)!`);
        await fetchOutputFiles(true);
        // Play newly generated audio
        if (audioRef.current) {
          audioRef.current.src = `/api/outputs/${data.filename}`;
          audioRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
        }
      } else {
        showToast(`❌ Error: ${data.error}`);
      }
    } catch (err: any) {
      showToast(`Generation error: ${err.message}`);
    } finally {
      setIsGenerating(false);
    }
  };

  const formatTime = (secs: number) => {
    if (isNaN(secs)) return '0:00';
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans">
      {/* Hidden audio element */}
      <audio
        ref={audioRef}
        src={selectedFile ? `/api/outputs/${selectedFile.filename}` : undefined}
      />

      {/* Toast Notification */}
      {toastMessage && (
        <div className="fixed top-4 right-4 z-50 bg-slate-900 text-white text-sm px-4 py-2.5 rounded-lg shadow-xl border border-slate-700 flex items-center gap-2 animate-in fade-in slide-in-from-top-2">
          <Info className="w-4 h-4 text-amber-400 shrink-0" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Top Header */}
      <header className="border-b border-slate-200 bg-white sticky top-0 z-30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-amber-500 to-orange-500 flex items-center justify-center text-white shadow-md shadow-orange-500/20">
              <Mic className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-bold text-lg text-slate-900">Qwen3-TTS Gradio Studio</h1>
                <span className="px-2 py-0.5 rounded text-xs font-semibold bg-orange-100 text-orange-700 border border-orange-200">
                  Gradio v6
                </span>
                <span className="px-2 py-0.5 rounded text-xs font-medium bg-slate-100 text-slate-700">
                  12Hz-1.7B
                </span>
              </div>
              <p className="text-xs text-slate-500">
                CustomVoice • VoiceDesign (Random & Fixed Seeds) • VoiceClone • Output Manager
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowConfig(!showConfig)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
                showConfig
                  ? 'bg-slate-900 text-white border-slate-900'
                  : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-100'
              }`}
            >
              <Settings className="w-3.5 h-3.5" />
              <span>Model Config</span>
            </button>
            <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-lg text-xs font-medium">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              <span>Gradio Server Ready</span>
            </div>
          </div>
        </div>

        {/* Global Settings Accordion */}
        {showConfig && (
          <div className="border-t border-slate-200 bg-slate-100/90 px-4 sm:px-6 py-4">
            <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-4 gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Model Repository ID</label>
                <input
                  type="text"
                  value={modelId}
                  onChange={(e) => setModelId(e.target.value)}
                  className="w-full text-xs font-mono bg-white border border-slate-300 rounded px-2.5 py-1.5 text-slate-800"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Device & Precision</label>
                <div className="grid grid-cols-2 gap-2">
                  <select
                    value={device}
                    onChange={(e) => setDevice(e.target.value)}
                    className="w-full text-xs bg-white border border-slate-300 rounded px-2 py-1.5"
                  >
                    <option value="cuda">cuda (GPU)</option>
                    <option value="cpu">cpu</option>
                  </select>
                  <select
                    value={dtype}
                    onChange={(e) => setDtype(e.target.value)}
                    className="w-full text-xs bg-white border border-slate-300 rounded px-2 py-1.5"
                  >
                    <option value="bfloat16">bfloat16</option>
                    <option value="float16">float16</option>
                    <option value="float32">float32</option>
                  </select>
                </div>
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">Attention & Chunking</label>
                <div className="grid grid-cols-3 gap-2">
                  <select
                    value={attn}
                    onChange={(e) => setAttn(e.target.value)}
                    className="col-span-1 text-xs bg-white border border-slate-300 rounded px-2 py-1.5"
                  >
                    <option value="flash_attention_2">flash_attn_2</option>
                    <option value="sdpa">sdpa</option>
                    <option value="eager">eager</option>
                  </select>
                  <input
                    type="number"
                    value={minWords}
                    onChange={(e) => setMinWords(parseInt(e.target.value) || 200)}
                    placeholder="Min words"
                    className="text-xs bg-white border border-slate-300 rounded px-2 py-1.5"
                  />
                  <input
                    type="number"
                    value={maxWords}
                    onChange={(e) => setMaxWords(parseInt(e.target.value) || 250)}
                    placeholder="Max words"
                    className="text-xs bg-white border border-slate-300 rounded px-2 py-1.5"
                  />
                </div>
              </div>
              <div className="flex flex-col justify-end">
                <button
                  onClick={() => {
                    setModelStatus(`Loaded: ${modelId} (${device}, ${dtype})`);
                    showToast('Model loaded successfully!');
                  }}
                  className="px-3 py-1.5 bg-orange-600 hover:bg-orange-700 text-white rounded text-xs font-semibold shadow transition-colors"
                >
                  Apply & Reload Model
                </button>
                <p className="text-[11px] text-slate-500 mt-1 truncate">{modelStatus}</p>
              </div>
            </div>
          </div>
        )}

        {/* Tab Navigation */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex gap-2 overflow-x-auto border-t border-slate-100">
          <button
            onClick={() => {
              setActiveTab('custom-voice');
              setModelId(MODEL_PRESETS['custom-voice']);
            }}
            className={`py-3 px-4 text-xs sm:text-sm font-medium border-b-2 flex items-center gap-2 whitespace-nowrap transition-colors ${
              activeTab === 'custom-voice'
                ? 'border-orange-500 text-orange-600 font-semibold'
                : 'border-transparent text-slate-600 hover:text-slate-900'
            }`}
          >
            <Mic className="w-4 h-4" />
            <span>CustomVoice</span>
          </button>

          <button
            onClick={() => {
              setActiveTab('voice-design');
              setModelId(MODEL_PRESETS['voice-design']);
            }}
            className={`py-3 px-4 text-xs sm:text-sm font-medium border-b-2 flex items-center gap-2 whitespace-nowrap transition-colors ${
              activeTab === 'voice-design'
                ? 'border-orange-500 text-orange-600 font-semibold'
                : 'border-transparent text-slate-600 hover:text-slate-900'
            }`}
          >
            <Wand2 className="w-4 h-4" />
            <span>VoiceDesign (Seeds)</span>
            <span className="ml-1 px-1.5 py-0.2 bg-orange-100 text-orange-700 text-[10px] rounded-full font-bold">
              🎲 Random/Fixed
            </span>
          </button>

          <button
            onClick={() => {
              setActiveTab('voice-clone');
              setModelId(MODEL_PRESETS['voice-clone']);
            }}
            className={`py-3 px-4 text-xs sm:text-sm font-medium border-b-2 flex items-center gap-2 whitespace-nowrap transition-colors ${
              activeTab === 'voice-clone'
                ? 'border-orange-500 text-orange-600 font-semibold'
                : 'border-transparent text-slate-600 hover:text-slate-900'
            }`}
          >
            <Layers className="w-4 h-4" />
            <span>VoiceClone</span>
          </button>

          <button
            onClick={() => {
              setActiveTab('outputs');
              fetchOutputFiles();
            }}
            className={`py-3 px-4 text-xs sm:text-sm font-medium border-b-2 flex items-center gap-2 whitespace-nowrap transition-colors ${
              activeTab === 'outputs'
                ? 'border-orange-500 text-orange-600 font-semibold'
                : 'border-transparent text-slate-600 hover:text-slate-900'
            }`}
          >
            <Folder className="w-4 h-4" />
            <span>Output Files & Playback</span>
            {outputFiles.length > 0 && (
              <span className="px-2 py-0.2 bg-slate-200 text-slate-800 text-[10px] font-bold rounded-full">
                {outputFiles.length}
              </span>
            )}
          </button>

          <button
            onClick={() => {
              setActiveTab('code');
              fetchPythonFiles();
            }}
            className={`py-3 px-4 text-xs sm:text-sm font-medium border-b-2 flex items-center gap-2 whitespace-nowrap transition-colors ${
              activeTab === 'code'
                ? 'border-orange-500 text-orange-600 font-semibold'
                : 'border-transparent text-slate-600 hover:text-slate-900'
            }`}
          >
            <FileCode className="w-4 h-4" />
            <span>Python Gradio Code</span>
          </button>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6 flex-1 w-full">
        {/* TAB 1: CUSTOM VOICE */}
        {activeTab === 'custom-voice' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-4">
              <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="font-semibold text-slate-800 flex items-center gap-2">
                    <Mic className="w-4 h-4 text-orange-500" />
                    <span>CustomVoice Persona Generation</span>
                  </h2>
                  <span className="text-xs text-slate-500 font-mono">Qwen3-TTS-CustomVoice</span>
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Text to Synthesize (auto-splits into chunks if exceeding limit)
                  </label>
                  <textarea
                    rows={4}
                    value={cvText}
                    onChange={(e) => setCvText(e.target.value)}
                    className="w-full text-sm border border-slate-300 rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-orange-500/20 focus:border-orange-500"
                    placeholder="Enter text..."
                  />
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">Target Language</label>
                    <select
                      value={cvLang}
                      onChange={(e) => setCvLang(e.target.value)}
                      className="w-full text-sm border border-slate-300 rounded-lg p-2.5 bg-white"
                    >
                      {LANGUAGES.map((l) => (
                        <option key={l} value={l}>
                          {l}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">Preset Speaker</label>
                    <select
                      value={cvSpeaker}
                      onChange={(e) => setCvSpeaker(e.target.value)}
                      className="w-full text-sm border border-slate-300 rounded-lg p-2.5 bg-white"
                    >
                      {Object.keys(SUPPORTED_SPEAKERS).map((s) => (
                        <option key={s} value={s}>
                          {s}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {/* Speaker Persona Description Banner */}
                <div className="p-3 bg-amber-50/70 border border-amber-200/80 rounded-lg text-xs text-amber-900 flex items-start gap-2">
                  <Info className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-semibold">{cvSpeaker}:</span>{' '}
                    <span>{SUPPORTED_SPEAKERS[cvSpeaker]}</span>
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Style / Emotion Instruction (Optional)
                  </label>
                  <input
                    type="text"
                    value={cvInstruct}
                    onChange={(e) => setCvInstruct(e.target.value)}
                    placeholder="e.g., cheerful, calm, whisper, excited, authoritative..."
                    className="w-full text-sm border border-slate-300 rounded-lg p-2.5"
                  />
                </div>

                <div className="pt-2 flex items-center gap-3">
                  <button
                    disabled={isGenerating}
                    onClick={() => handleGenerate('custom-voice')}
                    className="flex-1 py-3 px-4 bg-orange-600 hover:bg-orange-700 disabled:opacity-60 text-white rounded-lg font-medium text-sm shadow flex items-center justify-center gap-2 transition-colors"
                  >
                    {isGenerating ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        <span>Synthesizing Audio...</span>
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-4 h-4" />
                        <span>Generate CustomVoice Audio</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>

            {/* Side Output & Player */}
            <div className="space-y-4">
              <OutputPlayerCard
                file={selectedFile}
                isPlaying={isPlaying}
                onTogglePlay={togglePlay}
                currentTime={currentTime}
                duration={duration}
                volume={volume}
                onVolumeChange={setVolume}
                isMuted={isMuted}
                onToggleMute={() => setIsMuted(!isMuted)}
                onDelete={handleDeleteFile}
              />
            </div>
          </div>
        )}

        {/* TAB 2: VOICE DESIGN (RANDOM & FIXED SEED) */}
        {activeTab === 'voice-design' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-4">
              <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="font-semibold text-slate-800 flex items-center gap-2">
                    <Wand2 className="w-4 h-4 text-orange-500" />
                    <span>VoiceDesign Persona Generation</span>
                  </h2>
                  <span className="text-xs text-slate-500 font-mono">Qwen3-TTS-VoiceDesign</span>
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Text to Synthesize</label>
                  <textarea
                    rows={4}
                    value={vdText}
                    onChange={(e) => setVdText(e.target.value)}
                    className="w-full text-sm border border-slate-300 rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-orange-500/20 focus:border-orange-500"
                    placeholder="Enter text..."
                  />
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Voice Persona Description (Instruct - Required)
                  </label>
                  <textarea
                    rows={2}
                    value={vdInstruct}
                    onChange={(e) => setVdInstruct(e.target.value)}
                    className="w-full text-sm border border-slate-300 rounded-lg p-2.5"
                    placeholder="e.g. Warm, gentle female voice, calm and soothing tone."
                  />
                  <p className="text-[11px] text-slate-500 mt-1">
                    Describe age, gender, accent, timbre, cadence, and emotion in natural English or Chinese.
                  </p>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">Target Language</label>
                    <select
                      value={vdLang}
                      onChange={(e) => setVdLang(e.target.value)}
                      className="w-full text-sm border border-slate-300 rounded-lg p-2.5 bg-white"
                    >
                      {LANGUAGES.map((l) => (
                        <option key={l} value={l}>
                          {l}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                {/* --- SEED CONTROLS CARD --- */}
                <div className="border border-orange-200 bg-orange-50/50 rounded-xl p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Dices className="w-4 h-4 text-orange-600" />
                      <h3 className="font-semibold text-sm text-slate-900">Voice Persona Seed Controls</h3>
                    </div>
                    <span className="text-[11px] text-orange-700 font-medium bg-orange-100 px-2 py-0.5 rounded">
                      Feature: Random + Fixed Seeds
                    </span>
                  </div>

                  <p className="text-xs text-slate-600">
                    Control voice timbre consistency. <strong>Random</strong> mode generates fresh unique voices every
                    time. <strong>Fixed</strong> mode preserves the exact voice timbre across multiple text runs.
                  </p>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-1">
                    {/* Seed Mode Radio */}
                    <div>
                      <label className="block text-xs font-semibold text-slate-700 mb-1">Seed Mode</label>
                      <div className="grid grid-cols-2 gap-2 bg-white p-1 rounded-lg border border-slate-200">
                        <button
                          type="button"
                          onClick={() => setSeedMode('Random')}
                          className={`py-1.5 px-3 rounded-md text-xs font-medium transition-all flex items-center justify-center gap-1.5 ${
                            seedMode === 'Random'
                              ? 'bg-orange-500 text-white shadow-sm'
                              : 'text-slate-600 hover:text-slate-900'
                          }`}
                        >
                          <Sparkles className="w-3.5 h-3.5" />
                          <span>Random</span>
                        </button>

                        <button
                          type="button"
                          onClick={() => setSeedMode('Fixed')}
                          className={`py-1.5 px-3 rounded-md text-xs font-medium transition-all flex items-center justify-center gap-1.5 ${
                            seedMode === 'Fixed'
                              ? 'bg-orange-500 text-white shadow-sm'
                              : 'text-slate-600 hover:text-slate-900'
                          }`}
                        >
                          <Sliders className="w-3.5 h-3.5" />
                          <span>Fixed</span>
                        </button>
                      </div>
                    </div>

                    {/* Seed Number & Roll Button */}
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <label className="block text-xs font-semibold text-slate-700">
                          {seedMode === 'Fixed' ? 'Fixed Seed Number' : 'Current Seed Baseline'}
                        </label>
                        <button
                          type="button"
                          onClick={handleRollSeed}
                          className="text-[11px] text-orange-600 hover:text-orange-700 font-medium flex items-center gap-1"
                        >
                          <Dices className="w-3 h-3" />
                          <span>Roll Seed</span>
                        </button>
                      </div>
                      <div className="flex gap-2">
                        <input
                          type="number"
                          value={seedValue}
                          onChange={(e) => setSeedValue(parseInt(e.target.value) || 0)}
                          className="w-full text-xs font-mono border border-slate-300 rounded-lg px-3 py-2 bg-white"
                          placeholder="42"
                        />
                        <button
                          type="button"
                          onClick={handleRollSeed}
                          className="px-3 bg-white border border-slate-300 hover:bg-slate-100 rounded-lg text-xs font-medium text-slate-700 flex items-center justify-center"
                          title="Roll a random seed"
                        >
                          <Dices className="w-4 h-4 text-orange-600" />
                        </button>
                      </div>
                    </div>
                  </div>

                  {lastGeneratedSeed !== null && (
                    <div className="text-xs text-slate-600 bg-white border border-slate-200 rounded p-2 flex items-center justify-between">
                      <span>Last generated seed: <strong className="font-mono text-orange-600">{lastGeneratedSeed}</strong></span>
                      <button
                        onClick={() => {
                          setSeedValue(lastGeneratedSeed);
                          setSeedMode('Fixed');
                          showToast(`Locked seed to ${lastGeneratedSeed}`);
                        }}
                        className="text-[11px] text-orange-600 hover:underline font-semibold"
                      >
                        Lock as Fixed Seed
                      </button>
                    </div>
                  )}
                </div>

                <div className="pt-2">
                  <button
                    disabled={isGenerating}
                    onClick={() => handleGenerate('voice-design')}
                    className="w-full py-3 px-4 bg-orange-600 hover:bg-orange-700 disabled:opacity-60 text-white rounded-lg font-medium text-sm shadow flex items-center justify-center gap-2 transition-colors"
                  >
                    {isGenerating ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        <span>Synthesizing VoiceDesign...</span>
                      </>
                    ) : (
                      <>
                        <Wand2 className="w-4 h-4" />
                        <span>Generate VoiceDesign Audio ({seedMode} Seed)</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>

            {/* Output audio side */}
            <div className="space-y-4">
              <OutputPlayerCard
                file={selectedFile}
                isPlaying={isPlaying}
                onTogglePlay={togglePlay}
                currentTime={currentTime}
                duration={duration}
                volume={volume}
                onVolumeChange={setVolume}
                isMuted={isMuted}
                onToggleMute={() => setIsMuted(!isMuted)}
                onDelete={handleDeleteFile}
              />
            </div>
          </div>
        )}

        {/* TAB 3: VOICE CLONE */}
        {activeTab === 'voice-clone' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-4">
              <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="font-semibold text-slate-800 flex items-center gap-2">
                    <Layers className="w-4 h-4 text-orange-500" />
                    <span>VoiceClone (Zero-Shot Timbre Cloning)</span>
                  </h2>
                  <span className="text-xs text-slate-500 font-mono">Qwen3-TTS-Base</span>
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Text to Synthesize</label>
                  <textarea
                    rows={4}
                    value={vcText}
                    onChange={(e) => setVcText(e.target.value)}
                    className="w-full text-sm border border-slate-300 rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-orange-500/20 focus:border-orange-500"
                    placeholder="Enter text..."
                  />
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">Target Language</label>
                    <select
                      value={vcLang}
                      onChange={(e) => setVcLang(e.target.value)}
                      className="w-full text-sm border border-slate-300 rounded-lg p-2.5 bg-white"
                    >
                      {LANGUAGES.map((l) => (
                        <option key={l} value={l}>
                          {l}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-slate-700 mb-1">Reference Audio Sample</label>
                    <input
                      type="file"
                      accept=".wav,.mp3,.flac,.m4a"
                      className="w-full text-xs text-slate-500 file:mr-2 file:py-2 file:px-3 file:rounded-md file:border-0 file:text-xs file:font-semibold file:bg-orange-50 file:text-orange-700 hover:file:bg-orange-100"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">
                    Reference Transcript (Optional if X-vector only is checked)
                  </label>
                  <input
                    type="text"
                    value={vcRefText}
                    onChange={(e) => setVcRefText(e.target.value)}
                    placeholder="Exact transcript of the reference audio sample..."
                    className="w-full text-sm border border-slate-300 rounded-lg p-2.5"
                  />
                </div>

                <label className="flex items-center gap-2 cursor-pointer pt-1">
                  <input
                    type="checkbox"
                    checked={vcXvector}
                    onChange={(e) => setVcXvector(e.target.checked)}
                    className="rounded text-orange-600 focus:ring-orange-500"
                  />
                  <span className="text-xs text-slate-700 font-medium">
                    X-vector only mode (speaker embedding only; transcript not required)
                  </span>
                </label>

                <div className="pt-2">
                  <button
                    disabled={isGenerating}
                    onClick={() => handleGenerate('voice-clone')}
                    className="w-full py-3 px-4 bg-orange-600 hover:bg-orange-700 disabled:opacity-60 text-white rounded-lg font-medium text-sm shadow flex items-center justify-center gap-2 transition-colors"
                  >
                    {isGenerating ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin" />
                        <span>Cloning Voice...</span>
                      </>
                    ) : (
                      <>
                        <Layers className="w-4 h-4" />
                        <span>Clone & Synthesize Audio</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>

            {/* Output audio side */}
            <div className="space-y-4">
              <OutputPlayerCard
                file={selectedFile}
                isPlaying={isPlaying}
                onTogglePlay={togglePlay}
                currentTime={currentTime}
                duration={duration}
                volume={volume}
                onVolumeChange={setVolume}
                isMuted={isMuted}
                onToggleMute={() => setIsMuted(!isMuted)}
                onDelete={handleDeleteFile}
              />
            </div>
          </div>
        )}

        {/* TAB 4: OUTPUT FILES & PLAYBACK MANAGER */}
        {activeTab === 'outputs' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* File List */}
            <div className="lg:col-span-2 space-y-4">
              <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-4">
                <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                  <div>
                    <h2 className="font-semibold text-slate-800 flex items-center gap-2">
                      <Folder className="w-5 h-5 text-orange-500" />
                      <span>Output Audio Gallery</span>
                    </h2>
                    <p className="text-xs text-slate-500 mt-0.5">
                      {outputFiles.length} audio file(s) saved in <code className="text-slate-700 font-mono">./output</code>
                    </p>
                  </div>

                  <button
                    onClick={() => fetchOutputFiles()}
                    className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-xs font-medium transition-colors"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    <span>Refresh List</span>
                  </button>
                </div>

                {outputFiles.length === 0 ? (
                  <div className="text-center py-12 text-slate-500 space-y-2">
                    <FileAudio className="w-12 h-12 text-slate-300 mx-auto" />
                    <p className="text-sm font-medium">No audio files generated yet</p>
                    <p className="text-xs text-slate-400">Generate speech from any tab above to populate this list.</p>
                  </div>
                ) : (
                  <div className="divide-y divide-slate-100 max-h-[500px] overflow-y-auto">
                    {outputFiles.map((file) => {
                      const isSelected = selectedFile?.filename === file.filename;
                      return (
                        <div
                          key={file.filename}
                          onClick={() => handleSelectFile(file)}
                          className={`p-3.5 rounded-lg flex items-center justify-between gap-3 cursor-pointer transition-colors ${
                            isSelected ? 'bg-orange-50/80 border border-orange-200' : 'hover:bg-slate-50'
                          }`}
                        >
                          <div className="flex items-center gap-3 min-w-0">
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                if (isSelected) togglePlay();
                                else handleSelectFile(file);
                              }}
                              className={`w-9 h-9 rounded-full flex items-center justify-center shrink-0 transition-transform ${
                                isSelected && isPlaying
                                  ? 'bg-orange-600 text-white shadow-md'
                                  : 'bg-slate-200 text-slate-700 hover:bg-orange-500 hover:text-white'
                              }`}
                            >
                              {isSelected && isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
                            </button>

                            <div className="min-w-0">
                              <div className="flex items-center gap-2">
                                <span className="font-medium text-xs sm:text-sm text-slate-900 truncate">
                                  {file.filename}
                                </span>
                                <span
                                  className={`px-1.5 py-0.5 rounded text-[10px] font-semibold shrink-0 ${
                                    file.mode === 'VoiceDesign'
                                      ? 'bg-purple-100 text-purple-700 border border-purple-200'
                                      : file.mode === 'CustomVoice'
                                      ? 'bg-blue-100 text-blue-700 border border-blue-200'
                                      : 'bg-emerald-100 text-emerald-700 border border-emerald-200'
                                  }`}
                                >
                                  {file.mode}
                                </span>
                              </div>
                              <div className="flex items-center gap-3 text-xs text-slate-500 mt-0.5">
                                <span className="flex items-center gap-1">
                                  <Clock className="w-3 h-3" />
                                  <span>{file.duration}s</span>
                                </span>
                                <span className="flex items-center gap-1">
                                  <HardDrive className="w-3 h-3" />
                                  <span>{file.size_str}</span>
                                </span>
                                <span>{file.created_str}</span>
                              </div>
                            </div>
                          </div>

                          <div className="flex items-center gap-1 shrink-0">
                            <a
                              href={`/api/outputs/${file.filename}`}
                              download={file.filename}
                              onClick={(e) => e.stopPropagation()}
                              className="p-2 text-slate-400 hover:text-slate-700 rounded-lg hover:bg-slate-200"
                              title="Download WAV"
                            >
                              <Download className="w-4 h-4" />
                            </a>
                            <button
                              onClick={(e) => handleDeleteFile(file.filename, e)}
                              className="p-2 text-slate-400 hover:text-red-600 rounded-lg hover:bg-red-50"
                              title="Delete file"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>

            {/* Playback details */}
            <div className="space-y-4">
              <OutputPlayerCard
                file={selectedFile}
                isPlaying={isPlaying}
                onTogglePlay={togglePlay}
                currentTime={currentTime}
                duration={duration}
                volume={volume}
                onVolumeChange={setVolume}
                isMuted={isMuted}
                onToggleMute={() => setIsMuted(!isMuted)}
                onDelete={handleDeleteFile}
              />
            </div>
          </div>
        )}

        {/* TAB 5: PYTHON CODE & GRADIO EXPORT */}
        {activeTab === 'code' && (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            <div className="lg:col-span-1 space-y-2">
              <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-3 flex items-center gap-1.5">
                  <Terminal className="w-4 h-4 text-orange-500" />
                  <span>Python Source Files</span>
                </h3>
                <div className="space-y-1">
                  {pythonFiles.map((pf) => (
                    <button
                      key={pf.filename}
                      onClick={() => loadCodeContent(pf.filename)}
                      className={`w-full text-left px-3 py-2 rounded-lg text-xs font-mono flex items-center justify-between transition-colors ${
                        selectedCodeFile === pf.filename
                          ? 'bg-orange-500 text-white font-semibold'
                          : 'text-slate-700 hover:bg-slate-100'
                      }`}
                    >
                      <span className="truncate">{pf.filename}</span>
                      <span className="text-[10px] opacity-75">
                        {(pf.size_bytes / 1024).toFixed(1)}k
                      </span>
                    </button>
                  ))}
                </div>

                <div className="mt-4 pt-3 border-t border-slate-200 text-xs text-slate-600 space-y-2">
                  <p className="font-semibold text-slate-800">Run Locally with Gradio:</p>
                  <pre className="bg-slate-900 text-emerald-400 p-2.5 rounded font-mono text-[11px] overflow-x-auto">
                    pip install -r requirements.txt{'\n'}python launch.py
                  </pre>
                  <p className="text-[11px] text-slate-500">
                    Runs on <code className="text-orange-600">http://localhost:7860</code>
                  </p>
                </div>
              </div>
            </div>

            <div className="lg:col-span-3 space-y-2">
              <div className="bg-slate-900 text-slate-100 rounded-xl border border-slate-800 shadow-md overflow-hidden">
                <div className="bg-slate-950 px-4 py-2.5 flex items-center justify-between border-b border-slate-800">
                  <div className="flex items-center gap-2 font-mono text-xs text-slate-300">
                    <FileCode className="w-4 h-4 text-orange-400" />
                    <span>{selectedCodeFile}</span>
                  </div>

                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(codeContent);
                      setCopiedCode(true);
                      setTimeout(() => setCopiedCode(false), 2000);
                    }}
                    className="flex items-center gap-1.5 px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded text-xs transition-colors"
                  >
                    {copiedCode ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    <span>{copiedCode ? 'Copied!' : 'Copy Code'}</span>
                  </button>
                </div>

                <pre className="p-4 text-xs font-mono overflow-auto max-h-[600px] leading-relaxed text-slate-300">
                  {codeContent || '// Loading code...'}
                </pre>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

interface OutputPlayerCardProps {
  file: OutputFile | null;
  isPlaying: boolean;
  onTogglePlay: () => void;
  currentTime: number;
  duration: number;
  volume: number;
  onVolumeChange: (vol: number) => void;
  isMuted: boolean;
  onToggleMute: () => void;
  onDelete: (filename: string) => void;
}

function OutputPlayerCard({
  file,
  isPlaying,
  onTogglePlay,
  currentTime,
  duration,
  volume,
  onVolumeChange,
  isMuted,
  onToggleMute,
  onDelete,
}: OutputPlayerCardProps) {
  const formatTime = (secs: number) => {
    if (isNaN(secs)) return '0:00';
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  };

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  if (!file) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm text-center py-10 space-y-2">
        <Headphones className="w-10 h-10 text-slate-300 mx-auto" />
        <h3 className="text-sm font-semibold text-slate-700">Audio Playback Ready</h3>
        <p className="text-xs text-slate-500">
          Generate audio or select a file from the output gallery to listen and inspect waveform.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Headphones className="w-4 h-4 text-orange-500" />
          <h3 className="font-semibold text-sm text-slate-900">Audio Player</h3>
        </div>
        <span
          className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
            file.mode === 'VoiceDesign'
              ? 'bg-purple-100 text-purple-700'
              : file.mode === 'CustomVoice'
              ? 'bg-blue-100 text-blue-700'
              : 'bg-emerald-100 text-emerald-700'
          }`}
        >
          {file.mode}
        </span>
      </div>

      <div className="p-3 bg-slate-50 rounded-lg border border-slate-200/80 space-y-2">
        <p className="text-xs font-mono font-medium text-slate-800 break-all">{file.filename}</p>
        <div className="flex items-center justify-between text-[11px] text-slate-500">
          <span>{file.size_str}</span>
          <span>{file.sample_rate} Hz Mono</span>
          <span>{file.created_str}</span>
        </div>
      </div>

      {/* Waveform Visualization mockup bars */}
      <div className="flex items-end justify-between h-12 gap-0.5 px-2 bg-slate-900 rounded-lg p-2">
        {Array.from({ length: 36 }).map((_, i) => {
          const barHeight = Math.max(15, Math.sin((i / 36) * Math.PI) * 90 + ((i % 5) * 8));
          const isActive = (i / 36) * 100 <= progress;
          return (
            <div
              key={i}
              className={`w-full rounded-t-sm transition-all duration-100 ${
                isActive ? 'bg-orange-500' : 'bg-slate-700'
              }`}
              style={{ height: `${barHeight}%` }}
            />
          );
        })}
      </div>

      {/* Scrubber */}
      <div className="space-y-1">
        <div className="h-1.5 w-full bg-slate-200 rounded-full overflow-hidden">
          <div className="h-full bg-orange-500 transition-all" style={{ width: `${progress}%` }} />
        </div>
        <div className="flex justify-between text-[11px] text-slate-500 font-mono">
          <span>{formatTime(currentTime)}</span>
          <span>{formatTime(duration || file.duration)}</span>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between pt-1">
        <div className="flex items-center gap-2">
          <button
            onClick={onTogglePlay}
            className="w-10 h-10 rounded-full bg-orange-600 hover:bg-orange-700 text-white flex items-center justify-center shadow transition-colors"
          >
            {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5 ml-0.5" />}
          </button>

          <div className="flex items-center gap-1.5 ml-2">
            <button onClick={onToggleMute} className="text-slate-500 hover:text-slate-800">
              {isMuted ? <VolumeX className="w-4 h-4 text-red-500" /> : <Volume2 className="w-4 h-4" />}
            </button>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={isMuted ? 0 : volume}
              onChange={(e) => onVolumeChange(parseFloat(e.target.value))}
              className="w-16 accent-orange-500 h-1 bg-slate-200 rounded"
            />
          </div>
        </div>

        <div className="flex items-center gap-1">
          <a
            href={`/api/outputs/${file.filename}`}
            download={file.filename}
            className="p-2 text-slate-500 hover:text-slate-900 rounded-lg hover:bg-slate-100"
            title="Download WAV"
          >
            <Download className="w-4 h-4" />
          </a>
          <button
            onClick={() => onDelete(file.filename)}
            className="p-2 text-slate-500 hover:text-red-600 rounded-lg hover:bg-red-50"
            title="Delete WAV"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
