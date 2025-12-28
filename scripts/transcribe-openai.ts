/**
 * OpenAI STT API client with gpt-4o-transcribe-diarize
 *
 * Usage:
 *   bun run transcribe:openai <audio-file> [--output <file.json>]
 *
 * Environment:
 *   OPENAI_API_KEY - OpenAI API key
 */

import OpenAI from "openai";
import { unlink } from "node:fs/promises";
import { postprocessTranscript } from "./postprocess-transcript";
import type { TranscriptionResult } from "../src/types";

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
  startTime?: string;  // MM:SS.SSS format
  endTime?: string;    // MM:SS.SSS format
  speakerLabel?: string;
}

// ============ Audio Processing Functions ============

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
    // Find silence closest to target, expanding window until found
    let bestSilence: Silence | null = null;
    let bestDistance = Infinity;

    // Start with ±30s, expand up to ±180s (3 min)
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

    // Use WAV to avoid MP3 encoder delay issues
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

function convertToTranscriptionFormat(result: VerboseResponse): object {
  const words: TranscriptionWord[] = [];

  if (result.words) {
    // verbose_json with word timestamps
    for (const w of result.words) {
      words.push({
        word: w.word,
        startOffset: `${w.start.toFixed(3)}s`,
        endOffset: `${w.end.toFixed(3)}s`,
        speakerLabel: "0",
      });
    }
  } else if (result.segments) {
    // diarized_json with speaker segments
    for (const segment of result.segments) {
      const segmentWords = segment.text.trim().split(/\s+/);
      const duration = segment.end - segment.start;
      const wordDuration = duration / segmentWords.length;

      for (let i = 0; i < segmentWords.length; i++) {
        const wordStart = segment.start + i * wordDuration;
        const wordEnd = segment.start + (i + 1) * wordDuration;

        words.push({
          word: segmentWords[i],
          startOffset: `${wordStart.toFixed(3)}s`,
          endOffset: `${wordEnd.toFixed(3)}s`,
          speakerLabel: segment.speaker.replace("speaker_", ""),
        });
      }
    }
  }

  return {
    results: [
      {
        alternatives: [{ words }],
        languageCode: "ko-KR",
      },
    ],
  };
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
    // extra_body: {
    //   known_speaker_names: ["qqq"],
    //   known_speaker_references: [speakerRefDataUrl],
    // },
  });
  return transcription as VerboseResponse;
}

async function main() {
  const args = Bun.argv.slice(2);

  if (args.length === 0 || args.includes("--help") || args.includes("-h")) {
    console.log(`Usage: bun run scripts/transcribe-openai.ts <audio-file> [options]

Options:
  --output <file>         Write result to JSON file
  --chunk-duration <min>  Chunk duration in minutes (default: 10)
  --raw                   Output raw OpenAI response (not converted)
  --help, -h              Show this help

Environment:
  OPENAI_API_KEY   OpenAI API key`);
    process.exit(0);
  }

  let audioPath = "";
  let outputPath: string | null = null;
  let raw = false;
  let chunkDurationMin = 10;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--output" && args[i + 1]) {
      outputPath = args[++i];
    } else if (arg === "--chunk-duration" && args[i + 1]) {
      chunkDurationMin = parseInt(args[++i], 10);
    } else if (arg === "--raw") {
      raw = true;
    } else if (!arg.startsWith("-")) {
      audioPath = arg;
    }
  }

  if (!audioPath) {
    console.error("Error: Audio file path required");
    process.exit(1);
  }

  const file = Bun.file(audioPath);
  if (!(await file.exists())) {
    console.error(`Error: File not found: ${audioPath}`);
    process.exit(1);
  }

  const openai = new OpenAI();

  // Load speaker reference
  const speakerRefPath = new URL("../speaker-ref.mp3", import.meta.url).pathname;
  const speakerRefFile = Bun.file(speakerRefPath);
  const speakerRefBase64 = Buffer.from(await speakerRefFile.arrayBuffer()).toString("base64");
  const speakerRefDataUrl = `data:audio/mp3;base64,${speakerRefBase64}`;

  console.error(`Analyzing ${audioPath}...`);
  const duration = await getAudioDuration(audioPath);
  const chunkDurationSec = chunkDurationMin * 60;

  console.error(`Duration: ${(duration / 60).toFixed(1)} min`);
  console.error(`Using speaker reference: ${speakerRefPath}`);

  let allWords: TranscriptionWord[] = [];

  if (duration <= chunkDurationSec) {
    // Single chunk - transcribe directly
    console.error(`Transcribing (single chunk)...`);
    const result = await transcribeFile(openai, audioPath, speakerRefDataUrl);
    console.error(`Done. Found ${result.segments?.length ?? 0} segments.`);

    if (raw) {
      const json = JSON.stringify(result, null, 2);
      if (outputPath) {
        await Bun.write(outputPath, json);
        console.error(`Result written to ${outputPath}`);
      } else {
        console.log(json);
      }
      return;
    }

    allWords = extractWords(result);
  } else {
    // Multiple chunks - detect silences, split, transcribe each
    console.error(`Detecting silences...`);
    const silences = await detectSilences(audioPath);
    console.error(`Found ${silences.length} silence periods.`);

    const splitPoints = findSplitPoints(silences, duration, chunkDurationSec);
    console.error(`Split points: ${splitPoints.map(s => `${(s / 60).toFixed(1)}min`).join(", ")}`);

    console.error(`Splitting audio...`);
    const chunks = await splitAudio(audioPath, splitPoints, duration);
    console.error(`Created ${chunks.length} chunks.`);

    // Transcribe all chunks in parallel
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

    // Sort by index and merge words
    results.sort((a, b) => a.index - b.index);
    for (const r of results) {
      allWords.push(...r.words);
      await unlink(r.chunk.path).catch(() => {});
    }
  }

  const rawOutput: TranscriptionResult = {
    results: [
      {
        alternatives: [{ words: allWords }],
      },
    ],
  };

  // Post-process: Korean text → "큐", otherwise keep original
  const output = postprocessTranscript(rawOutput);
  console.error(`Post-processed speaker labels (Korean → 큐)`);

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
