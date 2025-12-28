/**
 * Video transcription pipeline
 *
 * Usage:
 *   bun run transcription <video-file> [--output <file.json>]
 *
 * Steps:
 *   1. Extract audio from video as WAV
 *   2. Run OpenAI transcription (chunked if > 10 min)
 *   3. Output transcript JSON
 */

import OpenAI from "openai";
import { unlink } from "node:fs/promises";

const CHUNK_DURATION_SEC = 600; // 10 minutes

interface Silence {
  start: number;
  end: number;
}

interface ChunkInfo {
  path: string;
  startOffset: number;
  duration: number;
}

interface DiarizedSegment {
  speaker: string;
  text: string;
  start: number;
  end: number;
}

interface VerboseResponse {
  text: string;
  words?: { word: string; start: number; end: number }[];
  segments?: DiarizedSegment[];
}

interface TranscriptionWord {
  word: string;
  startOffset?: string;
  endOffset?: string;
  startTime?: string;
  endTime?: string;
  speakerLabel?: string;
}

// ============ Audio Processing Functions ============

async function extractAudioFromVideo(videoPath: string): Promise<string> {
  const audioPath = `/tmp/transcription_audio_${Date.now()}.wav`;

  console.error(`Extracting audio from video...`);
  const proc = Bun.spawn([
    "ffmpeg", "-y", "-i", videoPath,
    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
    audioPath
  ], { stderr: "pipe" });

  await proc.exited;
  console.error(`Audio extracted to ${audioPath}`);
  return audioPath;
}

async function getAudioDuration(filePath: string): Promise<number> {
  const proc = Bun.spawn([
    "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
    "-of", "csv=p=0", filePath
  ], { stderr: "pipe" });
  const text = await new Response(proc.stdout).text();
  return parseFloat(text.trim());
}

async function detectSilences(filePath: string): Promise<Silence[]> {
  const proc = Bun.spawn([
    "ffmpeg", "-i", filePath,
    "-af", "silencedetect=noise=-30dB:d=2",
    "-f", "null", "-"
  ], { stderr: "pipe" });

  const text = await new Response(proc.stderr).text();
  const silences: Silence[] = [];
  let currentStart: number | null = null;

  for (const line of text.split("\n")) {
    const startMatch = line.match(/silence_start:\s*([\d.]+)/);
    const endMatch = line.match(/silence_end:\s*([\d.]+)/);

    if (startMatch) {
      currentStart = parseFloat(startMatch[1]);
    } else if (endMatch && currentStart !== null) {
      silences.push({ start: currentStart, end: parseFloat(endMatch[1]) });
      currentStart = null;
    }
  }

  return silences;
}

function findSplitPoints(silences: Silence[], totalDuration: number, chunkSec = CHUNK_DURATION_SEC): number[] {
  const splits: number[] = [0];

  for (let target = chunkSec; target < totalDuration; target += chunkSec) {
    let bestSilence: Silence | null = null;
    let bestDistance = Infinity;

    for (let window = 30; window <= 180 && !bestSilence; window += 30) {
      for (const s of silences) {
        if (s.start >= target - window && s.start <= target + window) {
          const distance = Math.abs(s.start - target);
          if (distance < bestDistance) {
            bestDistance = distance;
            bestSilence = s;
          }
        }
      }
    }

    splits.push(bestSilence ? bestSilence.start : target);
  }

  return splits;
}

async function splitAudio(filePath: string, splitPoints: number[], totalDuration: number): Promise<ChunkInfo[]> {
  const chunks: ChunkInfo[] = [];

  for (let i = 0; i < splitPoints.length; i++) {
    const start = splitPoints[i];
    const end = splitPoints[i + 1] ?? totalDuration;
    const chunkPath = `/tmp/transcribe_chunk_${i}.wav`;

    await Bun.spawn([
      "ffmpeg", "-y", "-i", filePath,
      "-ss", String(start), "-to", String(end),
      "-acodec", "pcm_s16le", "-ar", "16000", chunkPath
    ]).exited;

    chunks.push({ path: chunkPath, startOffset: start, duration: end - start });
  }

  return chunks;
}

// ============ Conversion Functions ============

function toMMSS(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins.toString().padStart(2, "0")}:${secs.toFixed(3).padStart(6, "0")}`;
}

function extractWords(result: VerboseResponse, offsetSec = 0): TranscriptionWord[] {
  const words: TranscriptionWord[] = [];

  if (result.words) {
    for (const w of result.words) {
      const startSec = w.start + offsetSec;
      const endSec = w.end + offsetSec;
      words.push({
        word: w.word,
        startOffset: `${startSec.toFixed(3)}s`,
        endOffset: `${endSec.toFixed(3)}s`,
        startTime: toMMSS(startSec),
        endTime: toMMSS(endSec),
        speakerLabel: "0",
      });
    }
  } else if (result.segments) {
    for (const segment of result.segments) {
      const segmentWords = segment.text.trim().split(/\s+/);
      if (segmentWords.length === 0 || segmentWords[0] === "") continue;

      const duration = segment.end - segment.start;
      const wordDuration = duration / segmentWords.length;

      for (let i = 0; i < segmentWords.length; i++) {
        const wordStart = segment.start + i * wordDuration + offsetSec;
        const wordEnd = segment.start + (i + 1) * wordDuration + offsetSec;

        words.push({
          word: segmentWords[i],
          startOffset: `${wordStart.toFixed(3)}s`,
          endOffset: `${wordEnd.toFixed(3)}s`,
          startTime: toMMSS(wordStart),
          endTime: toMMSS(wordEnd),
          speakerLabel: segment.speaker.replace("speaker_", ""),
        });
      }
    }
  }

  return words;
}

async function transcribeFile(
  openai: OpenAI,
  filePath: string,
  speakerRefDataUrl: string
): Promise<VerboseResponse> {
  const file = Bun.file(filePath);
  const transcription = await openai.audio.transcriptions.create({
    file: file,
    model: "gpt-4o-transcribe-diarize",
    // @ts-ignore
    response_format: "diarized_json",
    chunking_strategy: "auto",
    extra_body: {
      known_speaker_names: ["qqq"],
      known_speaker_references: [speakerRefDataUrl],
    },
  });
  return transcription as VerboseResponse;
}

// ============ Main ============

async function main() {
  const args = Bun.argv.slice(2);

  if (args.length === 0 || args.includes("--help") || args.includes("-h")) {
    console.log(`Usage: bun run scripts/transcription.ts <video-file> [options]

Options:
  --output <file>         Write result to JSON file
  --help, -h              Show this help

Environment:
  OPENAI_API_KEY   OpenAI API key`);
    process.exit(0);
  }

  let videoPath = "";
  let outputPath: string | null = null;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--output" && args[i + 1]) {
      outputPath = args[++i];
    } else if (!arg.startsWith("-")) {
      videoPath = arg;
    }
  }

  if (!videoPath) {
    console.error("Error: Video file path required");
    process.exit(1);
  }

  const videoFile = Bun.file(videoPath);
  if (!(await videoFile.exists())) {
    console.error(`Error: File not found: ${videoPath}`);
    process.exit(1);
  }

  // Extract audio from video
  const audioPath = await extractAudioFromVideo(videoPath);

  const openai = new OpenAI();

  // Load speaker reference
  const speakerRefPath = new URL("../speaker-ref.mp3", import.meta.url).pathname;
  const speakerRefFile = Bun.file(speakerRefPath);
  const speakerRefBase64 = Buffer.from(await speakerRefFile.arrayBuffer()).toString("base64");
  const speakerRefDataUrl = `data:audio/mp3;base64,${speakerRefBase64}`;

  console.error(`Analyzing audio...`);
  const duration = await getAudioDuration(audioPath);

  console.error(`Duration: ${(duration / 60).toFixed(1)} min`);
  console.error(`Using speaker reference: ${speakerRefPath}`);

  let allWords: TranscriptionWord[] = [];

  if (duration <= CHUNK_DURATION_SEC) {
    console.error(`Transcribing (single chunk)...`);
    const result = await transcribeFile(openai, audioPath, speakerRefDataUrl);
    console.error(`Done. Found ${result.segments?.length ?? 0} segments.`);
    allWords = extractWords(result);
  } else {
    console.error(`Detecting silences...`);
    const silences = await detectSilences(audioPath);
    console.error(`Found ${silences.length} silence periods.`);

    const splitPoints = findSplitPoints(silences, duration, CHUNK_DURATION_SEC);
    console.error(`Split points: ${splitPoints.map(s => `${(s / 60).toFixed(1)}min`).join(", ")}`);

    console.error(`Splitting audio...`);
    const chunks = await splitAudio(audioPath, splitPoints, duration);
    console.error(`Created ${chunks.length} chunks.`);

    console.error(`Transcribing ${chunks.length} chunks in parallel...`);
    const transcriptionPromises = chunks.map(async (chunk, i) => {
      console.error(`  Starting chunk ${i + 1}/${chunks.length} (offset: ${chunk.startOffset.toFixed(1)}s)...`);
      try {
        const result = await transcribeFile(openai, chunk.path, speakerRefDataUrl);
        console.error(`  Chunk ${i + 1} done. Found ${result.segments?.length ?? 0} segments.`);
        return { index: i, words: extractWords(result, chunk.startOffset), chunk };
      } catch (err: any) {
        console.error(`  Chunk ${i + 1} error: ${err.message}`);
        return { index: i, words: [] as TranscriptionWord[], chunk };
      }
    });

    const results = await Promise.all(transcriptionPromises);

    results.sort((a, b) => a.index - b.index);
    for (const r of results) {
      allWords.push(...r.words);
      await unlink(r.chunk.path).catch(() => {});
    }
  }

  // Cleanup extracted audio
  await unlink(audioPath).catch(() => {});

  const output = {
    results: [
      {
        alternatives: [{ words: allWords }],
        languageCode: "ko-KR",
      },
    ],
  };

  const json = JSON.stringify(output, null, 2);

  if (outputPath) {
    await Bun.write(outputPath, json);
    console.error(`Result written to ${outputPath}`);
    console.error(`Total words: ${allWords.length}`);
  } else {
    console.log(json);
  }
}

main().catch((err) => {
  console.error(`Error: ${err.message}`);
  process.exit(1);
});
