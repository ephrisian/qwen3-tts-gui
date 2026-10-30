import express, { Request, Response } from 'express';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { createServer as createViteServer } from 'vite';
import { execFile } from 'child_process';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;
const OUTPUT_DIR = path.resolve(__dirname, 'output');

if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Helper: parse WAV header for duration & sample rate
function getWavInfo(filePath: string) {
  try {
    const stats = fs.statSync(filePath);
    if (stats.size < 44) return { duration: 0, sampleRate: 24000, channels: 1 };
    const buffer = Buffer.alloc(44);
    const fd = fs.openSync(filePath, 'r');
    fs.readSync(fd, buffer, 0, 44, 0);
    fs.closeSync(fd);

    const channels = buffer.readUInt16LE(22);
    const sampleRate = buffer.readUInt32LE(24);
    const bitsPerSample = buffer.readUInt16LE(34);
    const bytesPerSample = (bitsPerSample / 8) * channels;

    // Search for data chunk or use remaining file size
    const dataSize = stats.size - 44;
    const duration = bytesPerSample > 0 && sampleRate > 0
      ? Math.round((dataSize / (sampleRate * bytesPerSample)) * 100) / 100
      : 0;

    return { duration, sampleRate, channels };
  } catch (err) {
    return { duration: 0, sampleRate: 24000, channels: 1 };
  }
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

// API: Get output files
app.get('/api/outputs', (req: Request, res: Response) => {
  try {
    const files = fs.readdirSync(OUTPUT_DIR);
    const validExts = ['.wav', '.mp3', '.ogg', '.flac'];
    const items = files
      .filter(f => validExts.includes(path.extname(f).toLowerCase()))
      .map(filename => {
        const fullPath = path.join(OUTPUT_DIR, filename);
        const stats = fs.statSync(fullPath);
        const info = getWavInfo(fullPath);

        let mode = 'Other';
        if (filename.startsWith('CustomVoice')) mode = 'CustomVoice';
        else if (filename.startsWith('VoiceDesign')) mode = 'VoiceDesign';
        else if (filename.startsWith('VoiceClone')) mode = 'VoiceClone';

        return {
          filename,
          path: fullPath,
          mode,
          size_bytes: stats.size,
          size_str: formatBytes(stats.size),
          created_ts: stats.mtimeMs,
          created_str: new Date(stats.mtimeMs).toISOString().replace('T', ' ').slice(0, 19),
          duration: info.duration,
          sample_rate: info.sampleRate,
        };
      })
      .sort((a, b) => b.created_ts - a.created_ts);

    res.json({ success: true, count: items.length, output_dir: OUTPUT_DIR, files: items });
  } catch (err: any) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// API: Stream audio file
app.get('/api/outputs/:filename', (req: Request, res: Response) => {
  const filename = path.basename(req.params.filename);
  const filePath = path.join(OUTPUT_DIR, filename);

  if (!fs.existsSync(filePath)) {
    return res.status(404).json({ error: 'File not found' });
  }

  const stat = fs.statSync(filePath);
  const total = stat.size;

  if (req.headers.range) {
    const range = req.headers.range;
    const parts = range.replace(/bytes=/, '').split('-');
    const partialstart = parts[0];
    const partialend = parts[1];

    const start = parseInt(partialstart, 10);
    const end = partialend ? parseInt(partialend, 10) : total - 1;
    const chunksize = end - start + 1;

    res.writeHead(206, {
      'Content-Range': `bytes ${start}-${end}/${total}`,
      'Accept-Ranges': 'bytes',
      'Content-Length': chunksize,
      'Content-Type': 'audio/wav',
    });
    fs.createReadStream(filePath, { start, end }).pipe(res);
  } else {
    res.writeHead(200, {
      'Content-Length': total,
      'Content-Type': 'audio/wav',
      'Content-Disposition': `inline; filename="${filename}"`,
    });
    fs.createReadStream(filePath).pipe(res);
  }
});

// API: Delete output file
app.delete('/api/outputs/:filename', (req: Request, res: Response) => {
  const filename = path.basename(req.params.filename);
  const filePath = path.join(OUTPUT_DIR, filename);

  if (!fs.existsSync(filePath)) {
    return res.status(404).json({ success: false, error: 'File not found' });
  }

  try {
    fs.unlinkSync(filePath);
    res.json({ success: true, message: `Deleted ${filename}` });
  } catch (err: any) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// API: Generate audio using Python generator
app.post('/api/generate', async (req: Request, res: Response) => {
  const {
    mode = 'custom-voice',
    text = 'Hello, this is Qwen3-TTS.',
    instruct = '',
    language = 'auto',
    speaker = 'Vivian',
    seed_mode = 'Random',
    seed_value = 42,
    min_words = 200,
    max_words = 250,
  } = req.body;

  try {
    // Resolve seed if random
    const isRandom = String(seed_mode).toLowerCase() === 'random';
    const actualSeed = isRandom ? Math.floor(Math.random() * 2147483647) : parseInt(String(seed_value), 10) || 42;

    // Call Python generator directly to create audio
    const pyScript = `
import json, sys
from tts_generator import TTSGenerator

gen = TTSGenerator(output_dir=${JSON.stringify(OUTPUT_DIR)})
mode = ${JSON.stringify(mode)}
text = ${JSON.stringify(text)}
lang = ${JSON.stringify(language)}
instruct = ${JSON.stringify(instruct)}
spk = ${JSON.stringify(speaker)}
seed_mode = ${JSON.stringify(seed_mode)}
actual_seed = ${actualSeed}
min_w = ${parseInt(String(min_words), 10) || 200}
max_w = ${parseInt(String(max_words), 10) || 250}

if mode == "custom-voice":
    res = gen.generate_custom_voice(text, language=lang, speaker=spk, instruct=instruct or None, min_words=min_w, max_words=max_w)
elif mode == "voice-design":
    res = gen.generate_voice_design(text, instruct=instruct, language=lang, seed_mode=seed_mode, seed_value=actual_seed, min_words=min_w, max_words=max_w)
elif mode == "voice-clone":
    res = gen.generate_voice_clone(text, ref_audio="output/CustomVoice_sample.wav", language=lang, min_words=min_w, max_words=max_w)
else:
    raise ValueError(f"Unknown mode: {mode}")

res["actual_seed"] = actual_seed
print(json.dumps(res))
`;

    const { stdout } = await execFileAsync('python3', ['-c', pyScript], {
      cwd: __dirname,
      timeout: 30000,
    });

    const lines = stdout.trim().split('\n');
    const lastLine = lines[lines.length - 1];
    const pyResult = JSON.parse(lastLine);

    const fullPath = pyResult.output_path;
    const info = getWavInfo(fullPath);
    const stats = fs.statSync(fullPath);

    res.json({
      success: true,
      filename: pyResult.filename,
      output_path: pyResult.output_path,
      audio_url: `/api/outputs/${pyResult.filename}`,
      actual_seed: pyResult.actual_seed || actualSeed,
      seed_mode: isRandom ? 'Random' : 'Fixed',
      chunks: pyResult.chunks || 1,
      duration: info.duration,
      size_str: formatBytes(stats.size),
      created_str: new Date().toISOString().replace('T', ' ').slice(0, 19),
    });
  } catch (err: any) {
    console.error('Generation error:', err);
    res.status(500).json({ success: false, error: err.message || 'Generation failed' });
  }
});

// API: Get Python Source Files for user inspection / download
app.get('/api/python-files', (req: Request, res: Response) => {
  const allowed = [
    'gradio_app.py',
    'app.py',
    'tts_generator.py',
    'config.py',
    'audio_utils.py',
    'text_utils.py',
    'model_loader.py',
    'launch.py',
    'tests.py',
    'requirements.txt',
  ];

  const fileParam = req.query.file as string;
  if (fileParam) {
    const safeName = path.basename(fileParam);
    if (!allowed.includes(safeName)) {
      return res.status(403).json({ error: 'File not permitted' });
    }
    const target = path.join(__dirname, safeName);
    if (!fs.existsSync(target)) {
      return res.status(404).json({ error: 'File not found' });
    }
    const content = fs.readFileSync(target, 'utf-8');
    return res.json({ filename: safeName, content });
  }

  const list = allowed.map(filename => {
    const target = path.join(__dirname, filename);
    const exists = fs.existsSync(target);
    const size = exists ? fs.statSync(target).size : 0;
    return { filename, size_bytes: size, exists };
  });

  res.json({ files: list });
});

async function startServer() {
  const isDev = process.env.NODE_ENV !== 'production';

  if (isDev) {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.resolve(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (req: Request, res: Response) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch(err => {
  console.error('Server startup error:', err);
  process.exit(1);
});
