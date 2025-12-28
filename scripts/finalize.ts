/**
 * Finalize: Combine scenes and transcript into final output
 *
 * Usage:
 *   bun run finalize --scenes <scenes.json> --transcript <transcript.json> --output <output-prefix>
 *
 * Outputs:
 *   <output-prefix>.json - Combined JSON
 *   <output-prefix>.txt  - Korean format with sov cues
 */

import { formatKoreanTxt, getWordsForScene, groupWordsBySpeaker } from "../src/utils";
import type { DetectionResult, TranscriptionResult, Shot } from "../src/types";

async function main() {
  const args = Bun.argv.slice(2);

  if (args.length === 0 || args.includes("--help") || args.includes("-h")) {
    console.log(`Usage: bun run scripts/finalize.ts [options]

Options:
  --scenes <file>         Path to scenes JSON file (required)
  --transcript <file>     Path to transcript JSON file (required)
  --output <prefix>       Output file prefix (produces .json and .txt)
  --help, -h              Show this help`);
    process.exit(0);
  }

  let scenesPath: string | null = null;
  let transcriptPath: string | null = null;
  let outputPrefix: string | null = null;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--scenes" && args[i + 1]) {
      scenesPath = args[++i];
    } else if (arg === "--transcript" && args[i + 1]) {
      transcriptPath = args[++i];
    } else if (arg === "--output" && args[i + 1]) {
      outputPrefix = args[++i];
    }
  }

  if (!scenesPath) {
    console.error("Error: --scenes is required");
    process.exit(1);
  }

  if (!transcriptPath) {
    console.error("Error: --transcript is required");
    process.exit(1);
  }

  if (!outputPrefix) {
    console.error("Error: --output is required");
    process.exit(1);
  }

  // Read scenes
  const scenesFile = Bun.file(scenesPath);
  if (!(await scenesFile.exists())) {
    console.error(`Error: Scenes file not found: ${scenesPath}`);
    process.exit(1);
  }
  const scenes: DetectionResult = await scenesFile.json();

  // Read transcript
  const transcriptFile = Bun.file(transcriptPath);
  if (!(await transcriptFile.exists())) {
    console.error(`Error: Transcript file not found: ${transcriptPath}`);
    process.exit(1);
  }
  const transcript: TranscriptionResult = await transcriptFile.json();

  // Extract words from transcript
  const words = transcript.results.flatMap((r) => r.alternatives[0]?.words ?? []);

  console.error(`Loaded ${scenes.shots.length} scenes and ${words.length} words`);

  // Generate Korean TXT
  const koreanTxt = formatKoreanTxt(scenes, words);

  // Write outputs
  const jsonPath = outputPrefix.endsWith(".json") ? outputPrefix : `${outputPrefix}.json`;
  const txtPath = outputPrefix.endsWith(".json")
    ? outputPrefix.replace(/\.json$/, ".txt")
    : `${outputPrefix}.txt`;

  // Build shots with transcription groups
  const shotsWithText = scenes.shots.map((shot) => {
    const sceneWords = getWordsForScene(words, shot.start_time, shot.end_time);
    const groups = groupWordsBySpeaker(sceneWords);
    const sovCues = groups.map((g) => ({
      speaker: g.speaker,
      text: g.text,
    }));
    return {
      ...shot,
      sov_cues: sovCues,
    };
  });

  // Write JSON (scenes with transcript)
  const finalJson = {
    ...scenes,
    shots: shotsWithText,
    transcript: {
      words_count: words.length,
      source: transcriptPath,
    },
  };
  await Bun.write(jsonPath, JSON.stringify(finalJson, null, 2));
  console.error(`JSON written to ${jsonPath}`);

  // Write Korean TXT
  await Bun.write(txtPath, koreanTxt);
  console.error(`TXT written to ${txtPath}`);

  console.error("Done!");
}

main().catch((err) => {
  console.error(`Error: ${err.message}`);
  process.exit(1);
});
