import type { DetectionResult, TranscriptionResult, TranscriptionWord } from "./types";
import { formatKoreanTxt } from "./utils";
import { detectShots } from "./.shot-detect";

function parseArgs(): { videoPath: string; threshold: number; json: boolean; output: string | null; transcriptionFile: string | null } {
  const args = Bun.argv.slice(2);

  if (args.length === 0 || args.includes("--help") || args.includes("-h")) {
    console.log(`Usage: bun run src/cli.ts <video-path> [options]

Options:
  --threshold <0-1>           Detection threshold (default: 0.5)
  --output <file>             Write JSON result to file
  --transcription-file <file> Transcription JSON for SOV cues in .txt output
  --json                      Output raw JSON to stdout
  --help, -h                  Show this help`);
    process.exit(0);
  }

  let videoPath = "";
  let threshold = 0.5;
  let json = false;
  let output: string | null = null;
  let transcriptionFile: string | null = null;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--threshold" && args[i + 1]) {
      threshold = parseFloat(args[++i]);
    } else if (arg === "--output" && args[i + 1]) {
      output = args[++i];
    } else if (arg === "--transcription-file" && args[i + 1]) {
      transcriptionFile = args[++i];
    } else if (arg === "--json") {
      json = true;
    } else if (!arg.startsWith("-")) {
      videoPath = arg;
    }
  }

  if (!videoPath) {
    console.error("Error: Video path is required");
    process.exit(1);
  }

  return { videoPath, threshold, json, output, transcriptionFile };
}

function formatOutput(result: DetectionResult): void {
  console.log(`\nVideo: ${result.video}`);
  console.log(`FPS: ${result.fps}`);
  console.log(`Total Frames: ${result.total_frames}`);
  console.log(`Detected Shots: ${result.shots.length}\n`);

  if (result.shots.length > 0) {
    console.log("Shots:");
    console.log("─".repeat(60));

    for (let i = 0; i < result.shots.length; i++) {
      const shot = result.shots[i];
      const duration = shot.end_time - shot.start_time;
      console.log(
        `  ${(i + 1).toString().padStart(3)}. ` +
        `${shot.start_time.toFixed(2)}s - ${shot.end_time.toFixed(2)}s ` +
        `(${duration.toFixed(2)}s) ` +
        `[frames ${shot.start_frame}-${shot.end_frame}]`
      );
    }
  }
}

async function loadTranscriptionWords(filePath: string): Promise<TranscriptionWord[]> {
  const file = Bun.file(filePath);
  if (!(await file.exists())) {
    console.error(`Error: Transcription file not found: ${filePath}`);
    process.exit(1);
  }
  const data = await file.json() as TranscriptionResult;
  return data.results.flatMap((r) => r.alternatives[0]?.words ?? []);
}

async function main(): Promise<void> {
  const { videoPath, threshold, json, output, transcriptionFile } = parseArgs();

  const result = await detectShots(videoPath, threshold);

  if ("error" in result) {
    console.error(`Error: ${result.error}`);
    process.exit(1);
  }

  let transcriptionWords: TranscriptionWord[] | undefined;
  if (transcriptionFile) {
    transcriptionWords = await loadTranscriptionWords(transcriptionFile);
  }

  if (output) {
    const txtOutput = output.replace(/\.json$/, ".txt");
    await Promise.all([
      Bun.write(output, JSON.stringify(result, null, 2)),
      Bun.write(txtOutput, formatKoreanTxt(result, transcriptionWords)),
    ]);
    console.log(`\nResult written to ${output} and ${txtOutput}`);
  } else if (json) {
    console.log(JSON.stringify(result, null, 2));
  } else {
    formatOutput(result);
  }
}

main();
