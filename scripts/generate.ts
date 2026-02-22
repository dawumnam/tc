/**
 * Generate: Single Gemini video call for scene descriptions, SOV cues, and captions
 *
 * Usage:
 *   bun run generate --video <video.mp4> --scenes <scenes.json> --output <prefix>
 *
 * Process:
 *   1. Upload video to Gemini File API
 *   2. Divide shots into ~5-minute segments
 *   3. For each segment, send video (clipped via startOffset/endOffset) + shot timestamps
 *   4. Gemini returns structured JSON: description, sov_cues[], captions[]
 *   5. Write .json + .txt + .hwpx
 */

import { GoogleGenAI, type FileState } from "@google/genai";
import { toMMSS } from "../src/utils";
import type {
  DetectionResult,
  GeneratedShot,
  GenerateResult,
} from "../src/types";

const GEMINI_MODEL = "gemini-3.1-pro-preview";
const DEFAULT_SEGMENT_DURATION = 300; // 5 minutes
const SEGMENT_PADDING = 2; // 2 seconds padding on each side
const POLL_INTERVAL_MS = 2000;

// Hand-written JSON schema (avoids z.toJSONSchema() nesting depth issues with Gemini)
const BATCH_RESPONSE_SCHEMA = {
  type: "object",
  properties: {
    shots: {
      type: "array",
      items: {
        type: "object",
        properties: {
          shot_index: { type: "number" },
          description: { type: "string" },
          shot_label: { type: ["string", "null"], description: "Shot size/camera label (ws, bs, cs, fs, ins, TD, TU, TR, pan) or null. Default null. Only for: curating (ws/bs), key interviews (bs), close-ups (cs/BS), camera moves (TD/TU/pan), inserts (ins), travel (TR), full-frame (fs)." },
          sov_cues: {
            type: "array",
            items: {
              type: "object",
              properties: {
                speaker: { type: "string" },
                text: { type: "string" },
              },
              required: ["speaker", "text"],
            },
          },
          captions: {
            type: "array",
            items: {
              type: "object",
              properties: {
                type: { type: "string", enum: ["자막", "정보자막", "말자막"] },
                text: { type: "string" },
              },
              required: ["type", "text"],
            },
          },
        },
        required: ["shot_index", "description", "shot_label", "sov_cues", "captions"],
      },
    },
  },
  required: ["shots"],
};

interface BatchResult {
  shots: {
    shot_index: number;
    description: string;
    shot_label: string | null;
    sov_cues: { speaker: string; text: string }[];
    captions: { type: string; text: string }[];
  }[];
}

/**
 * Upload video to Gemini File API and poll until ACTIVE
 */
async function uploadVideo(
  ai: GoogleGenAI,
  videoPath: string
): Promise<string> {
  console.error(`Uploading ${videoPath} to Gemini File API...`);

  const file = await ai.files.upload({
    file: videoPath,
    config: { mimeType: "video/mp4" },
  });

  if (!file.name) {
    throw new Error("Upload returned no file name");
  }

  console.error(`Upload complete: ${file.name}, state: ${file.state}`);

  // Poll until ACTIVE
  let current = file;
  while (current.state !== ("ACTIVE" as FileState)) {
    if (current.state === ("FAILED" as FileState)) {
      throw new Error(`File processing failed: ${file.name}`);
    }
    console.error(`  Waiting for file processing (state: ${current.state})...`);
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
    current = await ai.files.get({ name: file.name });
  }

  console.error(`File ready: ${current.uri}`);
  return current.uri!;
}

/**
 * Divide shots into time-based segments
 */
function createSegments(
  shots: DetectionResult["shots"],
  segmentDuration: number
): { startIdx: number; endIdx: number; startTime: number; endTime: number }[] {
  if (shots.length === 0) return [];

  const segments: {
    startIdx: number;
    endIdx: number;
    startTime: number;
    endTime: number;
  }[] = [];

  let segStart = 0;
  for (let i = 0; i < shots.length; i++) {
    const elapsed = shots[i].end_time - shots[segStart].start_time;
    const isLast = i === shots.length - 1;

    if (elapsed >= segmentDuration || isLast) {
      segments.push({
        startIdx: segStart,
        endIdx: i,
        startTime: shots[segStart].start_time,
        endTime: shots[i].end_time,
      });
      segStart = i + 1;
    }
  }

  return segments;
}

/**
 * Build the prompt for a segment
 */
function buildSegmentPrompt(
  shots: DetectionResult["shots"],
  startIdx: number,
  endIdx: number
): string {
  const shotList = shots
    .slice(startIdx, endIdx + 1)
    .map((shot, i) => {
      const start = toMMSS(shot.start_time);
      const end = toMMSS(shot.end_time);
      return `  Shot ${i}: ${start} ~ ${end} (${shot.rounded_duration}s)`;
    })
    .join("\n");

  return `You are analyzing a Korean TV program video. For each shot listed below, provide:
1. A visual scene description in Korean
2. Transcribed speech (SOV cues) with speaker identification
3. Any on-screen captions/text overlays

## Shot Timestamps
${shotList}

## Scene Description Rules
- Describe ONLY what you see visually — do NOT include speech content in the description
- Keep descriptions SHORT and concise (under 20 characters when possible)
- ALWAYS use abbreviations, NEVER spell out full words:
  - Shot sizes: ws, bs, cs, fs, ins (NEVER write "와이드샷", "클로즈업", "바스트샷" etc.)
  - Camera: TD (tilt down), TU (tilt up), TR (tracking/travel), pan (panning)
- Do NOT add 부감 to drone shots — 드론 prefix already implies aerial: "드론, 설원 위 푸르공과 텐트"
- For NON-큐 people, use specific terms: 남자, 남, 친구, 아이 etc. Do NOT always say 현지인.
- Do NOT include 정면/측면/후면 angles — omit them
- Driving/travel shots: short description only: "시간대별 주행 TR", "비포장도로 주행"

### shot_label Rules
  - DEFAULT IS NULL — most shots should NOT have a label
  - Shot size labels describe WHERE THE FRAME CUTS THE BODY:
    - bs (bust shot): frame cuts at chest level, maximum at belly/stomach
    - ws (waist shot): frame cuts at waist/pelvis, maximum at upper thigh
    - fs (full shot): full body visible
    - cs (close-up): face or object fills the frame
    - ins (insert): small detail item
  - Only use labels for these VERY SPECIFIC cases:
    1. 큐 doing STATIONARY on-camera curating/presentation — choose bs or ws based on actual framing
    2. Stationary interviewee/person explaining directly to camera — choose bs or ws based on actual framing. ONLY for key interview moments, NOT every talking shot.
    3. Object/statue close-ups (cs or BS)
    4. Driving/travel shots (TR)
    5. Camera movements on objects/buildings: TU (tilt up), TD (tilt down), pan
    6. Insert shots of small items/details (ins) — thermometers, gauges, equipment parts
    7. Full-body framing when notable (fs) — e.g. equipment displayed full-frame
  - MUST be null for:
    - Drone shots (드론 prefix already describes these)
    - People doing activities (타는/깨는/걷는/내리는/다가가는/만지는/닦는)
    - 큐 in car, 큐 talking casually, 큐 reacting
    - Bystanders, children, people walking by
    - Landscapes, scenery, CG/graphics
    - Most conversation shots between people
    - Reaction shots, surprise shots
  - When in doubt, use null
  - IMPORTANT: Having a person talk in frame does NOT automatically mean bs. Most talking shots should be null. Only give bs/ws to KEY interview/presentation moments, and choose between bs/ws based on actual body framing.

### Description + shot_label Examples (description → shot_label)
- "지도 CG" → null
- "드론, 울란바토르 시내 전경" → null (drone = no label)
- "드론, 설원 위 푸르공과 텐트" → null (drone = no label)
- "칭기즈 칸 동상" → "BS" (object close-up)
- "칭기즈 칸 동상 앞 계단" → "TD" (camera tilt down)
- "광장, 이글루 배경 걸어오는 부녀" → null (bystander walking)
- "미끄럼틀 위, 손 흔드는 큐" → null (action shot)
- "미끄럼틀 타고 내려오는 아이" → null (action shot)
- "걸어오며 큐레이팅 큐" → "ws" (큐 on-camera curating, frame at waist)
- "큐레이팅 큐" → "bs" (큐 on-camera curating, frame at chest)
- "손짓하며 설명하는 큐" → null (큐 explaining casually, not formal curating)
- "사우나 안에서 말하는 큐" → null (큐 talking casually)
- "얼음 조각" → "cs" (object close-up)
- "시간대별 주행" → "TR" (travel/tracking)
- "하얀 설원 달리는 차" → null
- "눈밭 걷는 큐의 발" → null (body part detail)
- "드론, 설원 위 걷는 큐" → null (drone shot)
- "눈사람 코 그리는 큐" → null (action shot)
- "차에서 내려 둘러보는 큐" → null (action shot)
- "차창 밖 내다보는 큐" → null (inside car)
- "대화 나누는 큐와 남자" → null (casual talk)
- "얼음 깨는 남자" → null (action shot)
- "차에서 내려 다가가는 큐" → null (action shot)
- "눈사람 옆 큐" → null (pose, not curating)
- "손짓하며 말하는 남자" → "bs" (key interview moment, explaining to camera)
- "웃으며 말하는 친구" → null (casual reaction, not key interview)
- "친구 가리키며 소개하는 남" → null (casual talk)
- "나무 온도계, 62도" → "ins" (insert shot of instrument)
- "돌 데우는 난로" → "TU" (camera tilts up on object)
- "사우나 장비들" → "fs" (equipment displayed full-frame)
- "차량 내부, 사우나 모습 전체" → "pan" (panning shot of interior)
- "남자 팔뚝 위 땀 송글송글" → "cs" (body detail close-up)
- "남자 배-웃는 얼굴" → "TU" (camera tilt up from belly to face)

## SOV Cue Rules
- The main host is "큐" (큐레이터/curator)
- For other speakers, use their name or role as shown on screen
- Transcribe ALL speech as Korean. If speakers talk in English or other languages, TRANSLATE to Korean in the SOV cue.
- Each continuous utterance by one speaker = one sov_cue entry
- Do NOT include speech in the description field
- **Line wrapping**: Break SOV text into lines of approximately 20-25 Korean characters using \\n. Example:
  "아 너무 재미있는데요? 몽골의 겨울은\\n보통 10월부터 4월까지 이어집니다.\\n반 년 가까이 땅이 얼어있고, 한 번 쌓인\\n눈은 좀처럼 녹지 않습니다."

## Caption Rules
- "자막": general text overlays, titles, decorative/editorial captions. Also for informational text that does NOT start with a proper name/title.
  - Examples: "이예지\\n여행 작가", "2026 최저 기온\\n영하 45.2도 기록 극한의 땅", "(깜짝CG)", "(땀뻘뻘)", "초원, 사막, 호수\\n천혜의 자연을 품은 낭만의 나라"
- "정보자막": ONLY for captions that start with a proper name/title (명칭) as a heading, followed by factual information.
  - Examples: "몽골 [Mongolia]\\n수도: 울란바토르 언어: 할흐 몽골어\\n인구: 354만 명(2024) 면적: 약 156만 ㎢", "우스콜\\n산악인", "얼음 수영 마니아"
  - NOT 정보자막: "2026 최저 기온\\n영하 45.2도 기록" (no proper name heading → use 자막)
  - NOT 정보자막: "현재 기온\\n영하 32도" (no proper name heading → use 자막)
  - Rule of thumb: if the first line is a proper noun (person name, place name, organization), it's 정보자막. Otherwise 자막.
- "말자막": ONLY when there is a visible on-screen text subtitle that directly echoes what someone says. If someone speaks but there is NO on-screen subtitle text for it, do NOT create a 말자막 — it should only be an SOV cue.
  - Example: person says "수영을 하려고요" and on-screen text shows "수영을 하려고요" → 말자막
  - Example: person says "뭐 하고 계신 거예요?" but NO on-screen text → sov_cue only, NO 말자막
- **One caption entry per visual caption block**. If a caption has multiple lines on screen, join them with \\n in a SINGLE entry. Do NOT split into separate entries.
  - CORRECT: one entry with text "이예지\\n여행 작가"
  - WRONG: two entries "이예지" and "여행 작가"
  - CORRECT: one entry with text "몽골[Mongolia]\\n수도: 울란바토르 언어: 할흐 몽골어\\n인구: 354만 명(2024) 면적: 약 156만 ㎢"
  - WRONG: three separate entries

Return a JSON object with a "shots" array. Each element must have:
- shot_index (number): the shot's index from the list above (0-based)
- description (string): Korean visual description WITHOUT shot label (e.g. "걸어오며 큐레이팅 큐", NOT "걸어오며 큐레이팅 큐 ws")
- shot_label (string or null): Shot size/camera abbreviation (ws, bs, cs, fs, ins, TD, TU, TR, pan) or null if not applicable. Default is null. Use null for drone shots, landscapes, most talking shots, action shots. Only use for: 큐 curating (ws/bs), key interviews (bs), object close-ups (cs/BS), camera moves (TD/TU/pan), inserts (ins), travel (TR), full-frame (fs).
- sov_cues (array): [{speaker, text}] — use \\n for line breaks in text
- captions (array): [{type, text}] — use \\n for multi-line captions

You MUST return exactly ${endIdx - startIdx + 1} shots in the array, one per shot listed above.`;
}

/**
 * Process a single segment with retries
 */
async function processSegment(
  ai: GoogleGenAI,
  fileUri: string,
  shots: DetectionResult["shots"],
  startIdx: number,
  endIdx: number,
  segmentStartTime: number,
  segmentEndTime: number,
  retries: number
): Promise<GeneratedShot[]> {
  const shotCount = endIdx - startIdx + 1;
  const paddedStart = Math.max(0, segmentStartTime - SEGMENT_PADDING);
  const paddedEnd = segmentEndTime + SEGMENT_PADDING;

  // Convert to "Xs" format for Gemini offsets
  const startOffsetStr = `${Math.floor(paddedStart)}s`;
  const endOffsetStr = `${Math.ceil(paddedEnd)}s`;

  const prompt = buildSegmentPrompt(shots, startIdx, endIdx);

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      if (attempt > 0) {
        const delay = Math.pow(2, attempt) * 1000;
        console.error(`  Retry ${attempt}/${retries} after ${delay}ms...`);
        await new Promise((r) => setTimeout(r, delay));
      }

      const response = await ai.models.generateContent({
        model: GEMINI_MODEL,
        contents: [
          {
            role: "user",
            parts: [
              {
                fileData: {
                  fileUri,
                  mimeType: "video/mp4",
                },
                videoMetadata: {
                  startOffset: startOffsetStr,
                  endOffset: endOffsetStr,
                },
              } as any,
              { text: prompt },
            ],
          },
        ],
        config: {
          responseMimeType: "application/json",
          responseSchema: BATCH_RESPONSE_SCHEMA as any,
        },
      });

      if (!response.text) {
        throw new Error("Empty response from Gemini");
      }

      const parsed: BatchResult = JSON.parse(response.text);

      // Map results back to shots
      return shots.slice(startIdx, endIdx + 1).map((shot, i) => {
        const result = parsed.shots.find((s) => s.shot_index === i);
        let desc = result?.description ?? "생성 실패";
        const label = result?.shot_label;
        // Avoid duplication: strip label from description if already there
        if (label && desc.endsWith(` ${label}`)) {
          desc = desc.slice(0, -(label.length + 1));
        }
        return {
          ...shot,
          description: label ? `${desc} ${label}` : desc,
          sov_cues: result?.sov_cues ?? [],
          captions: result?.captions ?? [],
        };
      });
    } catch (err: any) {
      console.error(
        `  Segment error (attempt ${attempt + 1}/${retries + 1}): ${err.message}`
      );
      if (attempt === retries) {
        // Return failed shots
        return shots.slice(startIdx, endIdx + 1).map((shot) => ({
          ...shot,
          description: "생성 실패",
          sov_cues: [],
          captions: [],
        }));
      }
    }
  }

  // Should never reach here
  return [];
}

/**
 * Format shots as cue sheet TXT
 */
function formatCueSheetTxt(shots: GeneratedShot[]): string {
  const lines: string[] = [];

  for (const shot of shots) {
    // Description line: description + duration (no "- " prefix)
    lines.push(`${shot.description} ${shot.rounded_duration}`);

    // Captions (before SOV cues)
    for (const caption of shot.captions) {
      const captionLines = caption.text.split("\n");
      lines.push(`${caption.type}/ ${captionLines[0]}`);
      for (let i = 1; i < captionLines.length; i++) {
        lines.push(` ${captionLines[i]}`);
      }
    }

    // SOV cues
    for (const cue of shot.sov_cues) {
      const sovLines = cue.text.split("\n");
      lines.push(`${cue.speaker} sov/ ${sovLines[0]}`);
      for (let i = 1; i < sovLines.length; i++) {
        lines.push(` ${sovLines[i]}`);
      }
    }

    lines.push(""); // Blank separator
  }

  return lines.join("\n");
}

/**
 * Build HWPX from empty.hwpx template + shot content
 */
async function buildHwpx(
  templatePath: string,
  shots: GeneratedShot[],
  outputPath: string
): Promise<void> {
  const { mkdtemp, rm, readdir, readFile, writeFile } = await import(
    "node:fs/promises"
  );
  const { join } = await import("node:path");
  const { tmpdir } = await import("node:os");

  const tmpDir = await mkdtemp(join(tmpdir(), "hwpx-"));

  try {
    // Unzip template
    const unzip = Bun.spawn(["unzip", "-o", templatePath, "-d", tmpDir], {
      stdout: "pipe",
      stderr: "pipe",
    });
    await unzip.exited;

    // Read section0.xml
    const sectionPath = join(tmpDir, "Contents", "section0.xml");
    let sectionXml = await readFile(sectionPath, "utf-8");

    // Build paragraph elements for each shot
    const paragraphs: string[] = [];
    const txtLines: string[] = [];

    for (const shot of shots) {
      // All text lines for this shot
      const shotLines: string[] = [];

      // Description + duration
      shotLines.push(`${shot.description} ${shot.rounded_duration}`);

      // Captions
      for (const caption of shot.captions) {
        const captionParts = caption.text.split("\n");
        shotLines.push(`${caption.type}/ ${captionParts[0]}`);
        for (let i = 1; i < captionParts.length; i++) {
          shotLines.push(` ${captionParts[i]}`);
        }
      }

      // SOV cues
      for (const cue of shot.sov_cues) {
        const sovParts = cue.text.split("\n");
        shotLines.push(`${cue.speaker} sov/ ${sovParts[0]}`);
        for (let i = 1; i < sovParts.length; i++) {
          shotLines.push(` ${sovParts[i]}`);
        }
      }

      // Add each line as a paragraph
      for (let li = 0; li < shotLines.length; li++) {
        const line = shotLines[li];
        const isDescription = li === 0; // First line is always the description
        const isIndented = line.startsWith(" ");

        const paraPrIDRef = isDescription ? "13" : "12";
        // Bullet paragraphs (paraPr 13) need flags=2490368; regular use 393216
        const flags = isDescription ? "2490368" : "393216";

        let runContent: string;
        if (isIndented) {
          // Tab goes INSIDE <hp:t> (matching reference hwpx structure)
          const text = line.slice(1);
          runContent =
            `<hp:t><hp:tab width="4000" leader="0" type="1"/>${escapeXml(text)}</hp:t>`;
        } else {
          runContent = `<hp:t>${escapeXml(line)}</hp:t>`;
        }

        paragraphs.push(
          `<hp:p id="0" paraPrIDRef="${paraPrIDRef}" styleIDRef="0" pageBreak="0" columnBreak="0" merged="0">` +
            `<hp:run charPrIDRef="9">${runContent}</hp:run>` +
            `<hp:linesegarray>` +
            `<hp:lineseg textpos="0" vertpos="0" vertsize="1100" textheight="1100" ` +
            `baseline="935" spacing="660" horzpos="0" horzsize="42520" flags="${flags}"/>` +
            `</hp:linesegarray>` +
            `</hp:p>`
        );
      }

      // Empty paragraph as separator between shots
      paragraphs.push(
        `<hp:p id="0" paraPrIDRef="12" styleIDRef="0" pageBreak="0" columnBreak="0" merged="0">` +
          `<hp:run charPrIDRef="9"/>` +
          `<hp:linesegarray>` +
          `<hp:lineseg textpos="0" vertpos="0" vertsize="1100" textheight="1100" ` +
          `baseline="935" spacing="660" horzpos="0" horzsize="42520" flags="393216"/>` +
          `</hp:linesegarray>` +
          `</hp:p>`
      );

      txtLines.push(...shotLines, "");
    }

    // Add a blank separator paragraph before the first shot
    const blankPara =
      `<hp:p id="0" paraPrIDRef="12" styleIDRef="0" pageBreak="0" columnBreak="0" merged="0">` +
      `<hp:run charPrIDRef="9"/>` +
      `<hp:linesegarray>` +
      `<hp:lineseg textpos="0" vertpos="0" vertsize="1100" textheight="1100" ` +
      `baseline="935" spacing="660" horzpos="0" horzsize="42520" flags="393216"/>` +
      `</hp:linesegarray>` +
      `</hp:p>`;

    // Inject paragraphs before </hs:sec>
    const closingTag = "</hs:sec>";
    const insertPos = sectionXml.lastIndexOf(closingTag);
    if (insertPos === -1) {
      throw new Error("Could not find </hs:sec> in section0.xml");
    }
    sectionXml =
      sectionXml.slice(0, insertPos) +
      blankPara +
      paragraphs.join("") +
      sectionXml.slice(insertPos);

    await writeFile(sectionPath, sectionXml, "utf-8");

    // Update Preview/PrvText.txt (preserve template header, append shot content)
    const prvTextPath = join(tmpDir, "Preview", "PrvText.txt");
    const existingPrvText = await readFile(prvTextPath, "utf-8");
    const separator = existingPrvText.endsWith("\n") ? "\n" : "\n\n";
    await writeFile(prvTextPath, existingPrvText + separator + txtLines.join("\n"), "utf-8");

    // Re-zip as .hwpx
    // First write mimetype uncompressed, then add the rest
    const zipProc = Bun.spawn(
      [
        "bash",
        "-c",
        `cd "${tmpDir}" && zip -0 -X "${outputPath}" mimetype && zip -r -X "${outputPath}" . -x mimetype`,
      ],
      { stdout: "pipe", stderr: "pipe" }
    );
    const zipExit = await zipProc.exited;
    if (zipExit !== 0) {
      const stderr = await new Response(zipProc.stderr).text();
      throw new Error(`zip failed: ${stderr}`);
    }

    console.error(`HWPX written to ${outputPath}`);
  } finally {
    await rm(tmpDir, { recursive: true, force: true });
  }
}

function escapeXml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

async function main() {
  const args = Bun.argv.slice(2);

  if (args.length === 0 || args.includes("--help") || args.includes("-h")) {
    console.log(`Usage: bun run scripts/generate.ts [options]

Options:
  --video <file>            Path to video file (required)
  --scenes <file>           Path to scenes.json file (required)
  --output <prefix>         Output file prefix (produces .json, .txt, .hwpx)
  --segment-duration <s>    Segment duration in seconds (default: ${DEFAULT_SEGMENT_DURATION})
  --file-uri <uri>          Skip upload, use existing Gemini file URI
  --retries <n>             Retries per segment (default: 2)
  --max-segments <n>        Only process first N segments (for testing)
  --rebuild-hwpx            Rebuild HWPX from existing JSON (skip Gemini)
  --help, -h                Show this help`);
    process.exit(0);
  }

  let videoPath: string | null = null;
  let scenesPath: string | null = null;
  let outputPrefix: string | null = null;
  let segmentDuration = DEFAULT_SEGMENT_DURATION;
  let fileUri: string | null = null;
  let retries = 2;
  let maxSegments = Infinity;
  let rebuildHwpx = false;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--video" && args[i + 1]) {
      videoPath = args[++i]!;
    } else if (arg === "--scenes" && args[i + 1]) {
      scenesPath = args[++i]!;
    } else if (arg === "--output" && args[i + 1]) {
      outputPrefix = args[++i]!;
    } else if (arg === "--segment-duration" && args[i + 1]) {
      segmentDuration = parseInt(args[++i]!);
    } else if (arg === "--file-uri" && args[i + 1]) {
      fileUri = args[++i]!;
    } else if (arg === "--retries" && args[i + 1]) {
      retries = parseInt(args[++i]!);
    } else if (arg === "--max-segments" && args[i + 1]) {
      maxSegments = parseInt(args[++i]!);
    } else if (arg === "--rebuild-hwpx") {
      rebuildHwpx = true;
    }
  }

  if (!outputPrefix) {
    console.error("Error: --output is required");
    process.exit(1);
  }
  if (!rebuildHwpx && !videoPath) {
    console.error("Error: --video is required");
    process.exit(1);
  }
  if (!rebuildHwpx && !scenesPath) {
    console.error("Error: --scenes is required");
    process.exit(1);
  }

  // Determine output paths
  const jsonPath = outputPrefix.endsWith(".json")
    ? outputPrefix
    : `${outputPrefix}.json`;
  const txtPath = outputPrefix.endsWith(".json")
    ? outputPrefix.replace(/\.json$/, ".txt")
    : `${outputPrefix}.txt`;
  const hwpxPath = outputPrefix.endsWith(".json")
    ? outputPrefix.replace(/\.json$/, ".hwpx")
    : `${outputPrefix}.hwpx`;

  // Rebuild HWPX only mode: read existing JSON and rebuild HWPX
  if (rebuildHwpx) {
    const jsonFile = Bun.file(jsonPath);
    if (!(await jsonFile.exists())) {
      console.error(`Error: JSON file not found: ${jsonPath}`);
      process.exit(1);
    }
    const result: GenerateResult = await jsonFile.json();
    console.error(`Rebuilding HWPX from ${jsonPath} (${result.shots.length} shots)`);

    const { resolve } = await import("node:path");
    const templatePath = resolve(import.meta.dir, "..", "empty.hwpx");
    const hwpxAbsPath = resolve(hwpxPath);
    await buildHwpx(templatePath, result.shots, hwpxAbsPath);

    // Also regenerate TXT
    const txtContent = formatCueSheetTxt(result.shots);
    await Bun.write(txtPath, txtContent);
    console.error(`TXT written to ${txtPath}`);

    console.error(`Done.`);
    process.exit(0);
  }

  // Check files exist
  const videoFile = Bun.file(videoPath);
  if (!(await videoFile.exists())) {
    console.error(`Error: Video file not found: ${videoPath}`);
    process.exit(1);
  }

  const scenesFile = Bun.file(scenesPath);
  if (!(await scenesFile.exists())) {
    console.error(`Error: Scenes file not found: ${scenesPath}`);
    process.exit(1);
  }

  // Check API key
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    console.error("Error: GEMINI_API_KEY environment variable is not set");
    process.exit(1);
  }

  // Load scenes
  const scenes: DetectionResult = await scenesFile.json();
  console.error(`Loaded ${scenes.shots.length} shots from ${scenesPath}`);
  console.error(`Video FPS: ${scenes.fps}`);

  // Initialize Gemini
  const ai = new GoogleGenAI({ apiKey });

  // Upload or reuse file URI
  const uriCachePath = `${outputPrefix}.file-uri`;
  if (!fileUri) {
    // Check for cached URI from previous run
    const cached = Bun.file(uriCachePath);
    if (await cached.exists()) {
      fileUri = (await cached.text()).trim();
      console.error(`Using cached file URI from ${uriCachePath}: ${fileUri}`);
    } else {
      const { resolve } = await import("node:path");
      fileUri = await uploadVideo(ai, resolve(videoPath));
      // Save URI for future runs
      await Bun.write(uriCachePath, fileUri);
      console.error(`File URI saved to ${uriCachePath}`);
    }
  } else {
    console.error(`Using provided file URI: ${fileUri}`);
  }

  // Create segments
  const allSegments = createSegments(scenes.shots, segmentDuration);
  const segments = allSegments.slice(0, maxSegments);
  const totalShots = segments.length > 0
    ? segments[segments.length - 1].endIdx - segments[0].startIdx + 1
    : 0;
  console.error(
    `\nProcessing ${totalShots} shots in ${segments.length} segment(s)${maxSegments < Infinity ? ` (limited from ${allSegments.length})` : ""}`
  );

  // Process each segment
  const allShots: GeneratedShot[] = [];

  for (let s = 0; s < segments.length; s++) {
    const seg = segments[s];
    const shotCount = seg.endIdx - seg.startIdx + 1;
    console.error(
      `\nSegment ${s + 1}/${segments.length}: shots ${seg.startIdx + 1}-${seg.endIdx + 1} (${toMMSS(seg.startTime)} ~ ${toMMSS(seg.endTime)}, ${shotCount} shots)`
    );

    const segmentShots = await processSegment(
      ai,
      fileUri,
      scenes.shots,
      seg.startIdx,
      seg.endIdx,
      seg.startTime,
      seg.endTime,
      retries
    );

    allShots.push(...segmentShots);

    // Log descriptions
    for (let i = 0; i < segmentShots.length; i++) {
      const shot = segmentShots[i];
      const sovCount = shot.sov_cues.length;
      const capCount = shot.captions.length;
      console.error(
        `  Shot ${seg.startIdx + i + 1}: ${shot.description} (${sovCount} sov, ${capCount} cap)`
      );
    }
  }

  // Build output
  const result: GenerateResult = {
    video: scenes.video,
    fps: scenes.fps,
    total_frames: scenes.total_frames,
    shots: allShots,
  };

  // Write JSON
  await Bun.write(jsonPath, JSON.stringify(result, null, 2));
  console.error(`\nJSON written to ${jsonPath}`);

  // Write TXT
  const txtContent = formatCueSheetTxt(allShots);
  await Bun.write(txtPath, txtContent);
  console.error(`TXT written to ${txtPath}`);

  // Write HWPX
  const { resolve } = await import("node:path");
  const templatePath = resolve(
    import.meta.dir,
    "..",
    "empty.hwpx"
  );
  const hwpxAbsPath = resolve(hwpxPath);
  await buildHwpx(templatePath, allShots, hwpxAbsPath);

  console.error(`\nTotal shots processed: ${allShots.length}`);
  console.error(
    `Output: ${jsonPath}, ${txtPath}, ${hwpxPath}`
  );
}

main().catch((err) => {
  console.error(`Error: ${err.message}`);
  process.exit(1);
});
