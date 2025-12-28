/**
 * Post-process transcript to assign speaker labels based on language
 *
 * Usage:
 *   bun run scripts/postprocess-transcript.ts <transcript.json>
 *
 * Rules:
 *   - If word contains Korean → speakerLabel: "큐"
 *   - Otherwise → speakerLabel: "Other"
 */

import type { TranscriptionResult } from "../src/types";

/**
 * Check if text contains Korean characters (Hangul syllables)
 */
function containsKorean(text: string): boolean {
  return /[\uAC00-\uD7AF]/.test(text);
}

/**
 * Post-process transcript words to assign speaker labels
 * - Korean text → "큐"
 * - Otherwise → keep original speakerLabel
 */
export function postprocessTranscript(transcript: TranscriptionResult): TranscriptionResult {
  return {
    results: transcript.results.map((result) => ({
      alternatives: result.alternatives.map((alt) => ({
        words: alt.words.map((word) => ({
          ...word,
          speakerLabel: containsKorean(word.word) ? "큐" : word.speakerLabel,
        })),
      })),
    })),
  };
}

async function main() {
  const args = Bun.argv.slice(2);

  if (args.length === 0 || args.includes("--help") || args.includes("-h")) {
    console.log(`Usage: bun run scripts/postprocess-transcript.ts <transcript.json>

Post-processes transcript to assign speaker labels:
  - Korean text → "큐"
  - Other text → keep original

The file is modified in-place.`);
    process.exit(0);
  }

  const inputPath = args[0]!;

  // Read transcript
  const file = Bun.file(inputPath);
  if (!(await file.exists())) {
    console.error(`Error: File not found: ${inputPath}`);
    process.exit(1);
  }

  const transcript: TranscriptionResult = await file.json();

  // Count words before
  const wordCount = transcript.results.reduce(
    (sum, r) => sum + r.alternatives.reduce((s, a) => s + a.words.length, 0),
    0
  );

  console.error(`Processing ${wordCount} words...`);

  // Post-process
  const processed = postprocessTranscript(transcript);

  // Count speakers
  let koreanCount = 0;
  let otherCount = 0;
  for (const result of processed.results) {
    for (const alt of result.alternatives) {
      for (const word of alt.words) {
        if (word.speakerLabel === "큐") {
          koreanCount++;
        } else {
          otherCount++;
        }
      }
    }
  }

  console.error(`  큐 (Korean): ${koreanCount} words`);
  console.error(`  Other: ${otherCount} words`);

  // Write back
  await Bun.write(inputPath, JSON.stringify(processed, null, 2));
  console.error(`Updated: ${inputPath}`);
}

// Only run main when executed directly (not when imported)
if (import.meta.main) {
  main().catch((err) => {
    console.error(`Error: ${err.message}`);
    process.exit(1);
  });
}
