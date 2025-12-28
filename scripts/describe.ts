/**
 * Describe: Generate scene descriptions using Gemini Flash
 *
 * Usage:
 *   bun run describe --video <video.mp4> --final <final.json> --output <output.json>
 *
 * Process:
 *   1. Read final.json with shots (start_frame, end_frame, sov_cues)
 *   2. For each shot, extract frames at 15-frame intervals
 *   3. Send frames + sov_cues to Gemini Flash
 *   4. Get single-line Korean description
 *   5. Write described.json with description field added
 */

import { GoogleGenAI, createUserContent } from "@google/genai";
import { z } from "zod";

const GEMINI_MODEL = "gemini-3-flash-preview";

const FRAME_INTERVAL = 15;
const BATCH_SIZE = 5; // Process 5 shots in parallel

// Schema for Gemini structured output
const DescriptionSchema = z.object({
  description: z.string().describe("Single-line Korean scene description"),
});

interface SovCue {
  speaker: string;
  text: string;
}

interface Shot {
  start_frame: number;
  end_frame: number;
  start_time: number;
  end_time: number;
  duration: number;
  rounded_duration: number;
  sov_cues: SovCue[];
}

interface FinalJson {
  video: string;
  fps: number;
  total_frames: number;
  shots: Shot[];
  transcript?: {
    words_count: number;
    source: string;
  };
}

/**
 * Get frame numbers to sample from a shot
 * start → +15 → +15 → ... → end
 */
function getFrameSamples(startFrame: number, endFrame: number, interval = FRAME_INTERVAL): number[] {
  const frames: number[] = [startFrame];

  let current = startFrame + interval;
  while (current < endFrame) {
    frames.push(current);
    current += interval;
  }

  // Always include end frame if not already added
  if (frames[frames.length - 1] !== endFrame) {
    frames.push(endFrame);
  }

  return frames;
}

/**
 * Get frames directory path for a video
 */
function getFramesDir(videoPath: string): string {
  const videoName = videoPath.replace(/\.[^.]+$/, "");
  return `${videoName}-frames`;
}

/**
 * Extract ALL frames from video to directory (single ffmpeg process)
 */
async function extractAllFramesToDir(videoPath: string, framesDir: string): Promise<void> {
  const { mkdir } = await import("node:fs/promises");
  await mkdir(framesDir, { recursive: true });

  console.error(`Extracting all frames to ${framesDir}...`);

  const proc = Bun.spawn([
    "ffmpeg",
    "-i", videoPath,
    "-q:v", "2",
    "-start_number", "0",
    `${framesDir}/frame_%06d.jpg`
  ], { stdout: "pipe", stderr: "pipe" });

  const exitCode = await proc.exited;
  if (exitCode !== 0) {
    const stderr = await new Response(proc.stderr).text();
    throw new Error(`ffmpeg frame extraction failed: ${stderr}`);
  }

  console.error(`Frame extraction complete.`);
}

/**
 * Check if frames directory exists and has frames
 */
async function framesExist(framesDir: string): Promise<boolean> {
  const dir = Bun.file(`${framesDir}/frame_000000.jpg`);
  return dir.exists();
}

/**
 * Read a frame from the frames directory
 */
async function readFrame(framesDir: string, frameNum: number): Promise<string> {
  const framePath = `${framesDir}/frame_${frameNum.toString().padStart(6, "0")}.jpg`;
  const file = Bun.file(framePath);
  const buffer = await file.arrayBuffer();
  return Buffer.from(buffer).toString("base64");
}

/**
 * Get frames for a shot from the frames directory
 */
async function getFramesForShot(
  framesDir: string,
  startFrame: number,
  endFrame: number
): Promise<{ frame: number; base64: string }[]> {
  const frameNums = getFrameSamples(startFrame, endFrame);
  const results = await Promise.all(
    frameNums.map(async (frame) => ({
      frame,
      base64: await readFrame(framesDir, frame),
    }))
  );
  return results;
}

/**
 * Build the prompt for Gemini
 */
function buildPrompt(sovCues: SovCue[]): string {
  // Get unique speakers (excluding default "큐")
  const otherSpeakers = [...new Set(sovCues.map(c => c.speaker))].filter(s => s !== "큐");
  const speakerContext = otherSpeakers.length > 0
    ? `Other speakers in this scene: ${otherSpeakers.join(", ")}`
    : "";

  return `Analyze these video frames and generate ONE single-line Korean scene description.

The main host is "큐" (큐레이터/curator). ${speakerContext}

## IMPORTANT RULES:
1. **ONE description per scene** - describe only what you see in these frames
2. **DO NOT include speech content** - never quote or paraphrase what is being said
3. **Describe VISUAL actions only** - what subjects are doing physically, not what they're saying

## Description Format:

Include relevant details:
- Camera type: 드론 (if aerial)
- Shot type: fs, ws, bs, cs, ls, ins, sk, ks
- Subject & visible physical action
- Angle: 정면, 후면, 측면 (if relevant)
- Effects: 슬로모션, 타임랩스, CG (if applicable)

## Examples:
### 풍경 및 특수 촬영
- 드론, 콜로세움 외관
- 콜로세움 위 구름 지나가는 타임랩스
- 드론, 포도밭 걸어오는 큐
- 카이사르 석고상, 가이우스 그림 CG
- 걷는 병사들 다리 슬로모션 cs CG
- 드론, 루비콘 강 하이앵글

### 인물 단독 행동
- 길 걸으며 말하는 큐 측면 bs
- 스푼을 입에 넣고 음미하는 큐 cs
- 찡그리며 음미하는 큐 슬로모션 bs
- 팔 뻗으며 외치는 큐 bs
- 돌 던지는 큐 측면 ws 슬로모션
- 루비콘 강 건너는 큐의 발 cs 슬로모션

### 인물 상호작용
- 큐 반기는 마르코 ws
- 큐, 마르코 악수하며 인사
- 함께 걷는 두 사람 측면 fs
- 피자를 먹는 마, 큐 bs

### 사물 및 디테일
- 스푼에 발사믹 한방울씩 떨어지는 ins
- 테이블 위 피자 ins
- 활활 타는 모닥불 sk

Return ONLY a JSON object with a single "description" field.`;
}

/**
 * Generate description for a single shot using Gemini
 */
async function describeShot(
  ai: GoogleGenAI,
  framesDir: string,
  shot: Shot,
  shotIndex: number
): Promise<string> {
  // Get frames from directory
  const frames = await getFramesForShot(framesDir, shot.start_frame, shot.end_frame);

  console.error(`  Shot ${shotIndex + 1}: ${frames.length} frames`);

  // Build content for Gemini
  const contents = createUserContent([
    ...frames.map(f => ({
      inlineData: {
        mimeType: "image/jpeg" as const,
        data: f.base64,
      },
    })),
    { text: buildPrompt(shot.sov_cues || []) },
  ]);

  // Call Gemini
  try {
    const response = await ai.models.generateContent({
      model: GEMINI_MODEL,
      contents,
      config: {
        responseMimeType: "application/json",
        responseSchema: z.toJSONSchema(DescriptionSchema),
      },
    });

    if (!response.text) {
      console.error(`    Shot ${shotIndex + 1}: Empty response`);
      return "생성 실패";
    }

    const result = DescriptionSchema.parse(JSON.parse(response.text));
    return result.description;
  } catch (err: any) {
    console.error(`    Shot ${shotIndex + 1}: ${err.message}`);
    return "생성 실패";
  }
}

async function main() {
  const args = Bun.argv.slice(2);

  if (args.length === 0 || args.includes("--help") || args.includes("-h")) {
    console.log(`Usage: bun run scripts/describe.ts [options]

Options:
  --video <file>    Path to video file (required)
  --final <file>    Path to final.json file (required)
  --output <file>   Output file path (required)
  --batch <n>       Parallel batch size (default: ${BATCH_SIZE})
  --help, -h        Show this help`);
    process.exit(0);
  }

  let videoPath: string | null = null;
  let finalPath: string | null = null;
  let outputPath: string | null = null;
  let batchSize = BATCH_SIZE;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--video" && args[i + 1]) {
      videoPath = args[++i]!;
    } else if (arg === "--final" && args[i + 1]) {
      finalPath = args[++i]!;
    } else if (arg === "--output" && args[i + 1]) {
      outputPath = args[++i]!;
    } else if (arg === "--batch" && args[i + 1]) {
      batchSize = parseInt(args[++i]!);
    }
  }

  if (!videoPath) {
    console.error("Error: --video is required");
    process.exit(1);
  }
  if (!finalPath) {
    console.error("Error: --final is required");
    process.exit(1);
  }
  if (!outputPath) {
    console.error("Error: --output is required");
    process.exit(1);
  }

  // Check files exist
  const videoFile = Bun.file(videoPath);
  if (!(await videoFile.exists())) {
    console.error(`Error: Video file not found: ${videoPath}`);
    process.exit(1);
  }

  const finalFile = Bun.file(finalPath);
  if (!(await finalFile.exists())) {
    console.error(`Error: Final file not found: ${finalPath}`);
    process.exit(1);
  }

  // Check API key
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    console.error("Error: GEMINI_API_KEY environment variable is not set");
    process.exit(1);
  }

  // Load final.json
  const finalData: FinalJson = await finalFile.json();
  console.error(`Loaded ${finalData.shots.length} shots from ${finalPath}`);
  console.error(`Video FPS: ${finalData.fps}`);

  // Check/extract frames
  const framesDir = getFramesDir(videoPath!);
  if (await framesExist(framesDir)) {
    console.error(`Using existing frames from ${framesDir}`);
  } else {
    await extractAllFramesToDir(videoPath!, framesDir);
  }

  // Initialize Gemini
  const ai = new GoogleGenAI({ apiKey });

  // Process shots in batches
  const shotsWithDescriptions: (Shot & { description: string })[] = [];

  for (let i = 0; i < finalData.shots.length; i += batchSize) {
    const batch = finalData.shots.slice(i, i + batchSize);
    const batchNum = Math.floor(i / batchSize) + 1;
    const totalBatches = Math.ceil(finalData.shots.length / batchSize);

    console.error(`\nBatch ${batchNum}/${totalBatches} (shots ${i + 1}-${i + batch.length})`);

    const batchResults = await Promise.all(
      batch.map(async (shot, j) => {
        const description = await describeShot(ai, framesDir, shot, i + j);
        return { ...shot, description };
      })
    );

    shotsWithDescriptions.push(...batchResults);

    // Log progress
    for (const shot of batchResults) {
      console.error(`    → ${shot.description}`);
    }
  }

  // Build output
  const output = {
    ...finalData,
    shots: shotsWithDescriptions,
  };

  // Build TXT output
  const txtLines: string[] = [];
  for (const shot of shotsWithDescriptions) {
    // - {{description}} {{duration}}
    txtLines.push(`- ${shot.description} ${shot.rounded_duration}`);

    // {{speaker}} sov/ {{text}} for each cue
    if (shot.sov_cues && shot.sov_cues.length > 0) {
      for (const cue of shot.sov_cues) {
        txtLines.push(`${cue.speaker} sov/ ${cue.text}`);
      }
    }

    txtLines.push(""); // Empty line between scenes
  }

  // Determine output paths
  const jsonPath = outputPath.endsWith(".json") ? outputPath : `${outputPath}.json`;
  const txtPath = outputPath.endsWith(".json")
    ? outputPath.replace(/\.json$/, ".txt")
    : `${outputPath}.txt`;

  // Write outputs
  await Bun.write(jsonPath, JSON.stringify(output, null, 2));
  console.error(`\nJSON written to ${jsonPath}`);

  await Bun.write(txtPath, txtLines.join("\n"));
  console.error(`TXT written to ${txtPath}`);

  console.error(`Total shots processed: ${shotsWithDescriptions.length}`);
}

main().catch((err) => {
  console.error(`Error: ${err.message}`);
  process.exit(1);
});
