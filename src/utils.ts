import type { DetectionResult, TranscriptionWord } from "./types";

export function toMMSS(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  const secsInt = Math.floor(secs);
  const secsDec = secs % 1;
  const secsStr = secsDec > 0
    ? `${secsInt.toString().padStart(2, "0")}${secsDec.toFixed(1).slice(1)}`
    : secsInt.toString().padStart(2, "0");
  return `${mins.toString().padStart(2, "0")}:${secsStr}`;
}

export function parseOffset(offset: string | undefined): number {
  if (!offset) return 0;
  return parseFloat(offset.replace("s", ""));
}

export function getWordsForScene(
  words: TranscriptionWord[],
  startTime: number,
  endTime: number
): TranscriptionWord[] {
  return words.filter((w) => {
    if (!w.startOffset) return false;
    const wordStart = parseOffset(w.startOffset);
    // Assign word to scene based on where it starts (not where it ends)
    return wordStart >= startTime && wordStart < endTime;
  });
}

export function groupWordsBySpeaker(
  words: TranscriptionWord[]
): { speaker: string; text: string }[] {
  if (words.length === 0) return [];

  const groups: { speaker: string; text: string }[] = [];
  let currentSpeaker = words[0].speakerLabel ?? "0";
  let currentWords: string[] = [words[0].word];

  for (let i = 1; i < words.length; i++) {
    const word = words[i];
    const speaker = word.speakerLabel ?? "0";
    if (speaker === currentSpeaker) {
      currentWords.push(word.word);
    } else {
      groups.push({ speaker: currentSpeaker, text: currentWords.join(" ") });
      currentSpeaker = speaker;
      currentWords = [word.word];
    }
  }
  groups.push({ speaker: currentSpeaker, text: currentWords.join(" ") });

  return groups;
}

export function formatKoreanTxt(
  result: DetectionResult,
  transcriptionWords?: TranscriptionWord[]
): string {
  const lines: string[] = [];

  for (let i = 0; i < result.shots.length; i++) {
    const shot = result.shots[i];
    lines.push(`씬:${i + 1}`);
    lines.push(`시작시간: ${toMMSS(shot?.start_time)}`);
    lines.push(`끝시간: ${toMMSS(shot?.end_time)}`);
    lines.push(`${shot.rounded_duration}`);

    if (transcriptionWords) {
      const sceneWords = getWordsForScene(
        transcriptionWords,
        shot.start_time,
        shot.end_time
      );
      const groups = groupWordsBySpeaker(sceneWords);
      for (const group of groups) {
        lines.push(`sov ${group.speaker} \n ${group.text}`);
      }
    }

    lines.push("");
  }

  return lines.join("\n");
}
