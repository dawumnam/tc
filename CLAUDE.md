# TC-1: Korean TV Cue Sheet Generator

## Pipeline

```
video → detect_shots.py → scenes.json
video + scenes.json → generate.ts → .json + .txt + .hwpx
```

### Step 1: Shot Detection
```bash
bun run detect <video.mp4> --output scenes.json
```
Uses TransNetV2 (deep learning) via `scripts/detect_shots.py` (Python venv at `.venv/`).

### Step 2: Generate (single Gemini call per segment)
```bash
bun run generate --video <video.mp4> --scenes scenes.json --output result
```
Options:
- `--segment-duration <s>` — Segment duration in seconds (default: 300)
- `--file-uri <uri>` — Skip upload, reuse existing Gemini file URI
- `--retries <n>` — Retries per segment (default: 2)
- `--max-segments <n>` — Only process first N segments (for testing)
- `--rebuild-hwpx` — Rebuild HWPX + TXT from existing JSON without calling Gemini (only `--output` required)

Outputs: `result.json`, `result.txt`, `result.hwpx`

File URI caching: After first upload, the Gemini file URI is saved to `{output}.file-uri` and reused on subsequent runs automatically.

## Key Files

- `scripts/generate.ts` — Main generation script (Gemini video API, cue sheet TXT/HWPX output)
- `scripts/detect_shots.py` — Shot detection (TransNetV2, TensorFlow + Metal GPU)
- `src/types.ts` — Shared types: `Shot`, `SovCue`, `Caption`, `GeneratedShot`, `GenerateResult`, `DetectionResult`
- `src/utils.ts` — Helpers: `toMMSS()`, `formatKoreanTxt()` (legacy scene format), `groupWordsBySpeaker()`, `parseOffset()`
- `src/cli.ts` + `src/.shot-detect.ts` — Bun CLI wrapper for shot detection (spawns Python subprocess)
- `empty.hwpx` — HWPX template for output generation (root-level, used by generate.ts)

## Output Format (TXT/HWPX Cue Sheet)

Matches the style in `mongolia/*/tc-mongolia-*.hwpx`:
```
드론, 울란바토르 시내 전경 7
자막/ 울란바토르

드론, 울란바토르 건물들 7
정보자막/ 몽골[Mongolia]
 수도: 울란바토르 언어: 할흐 몽골어
 인구: 354만 명(2024) 면적: 약 156만 ㎢

걸어오며 큐레이팅 큐 ws 11
자막/ 이예지
 여행 작가
큐 sov/ 아 너무 재미있는데요? 몽골의 겨울은
 보통 10월부터 4월까지 이어집니다.
```

### Format Rules
- Description + `shot_label` (if any) + `rounded_duration` on same line (e.g. `걸어오며 큐레이팅 큐 ws 11`)
- No `- ` prefix on description lines
- Captions before SOV cues
- Multi-line continuation uses leading space indent
- Blank line between shots
- Caption types: `자막/`, `정보자막/`, `말자막/`
- SOV format: `{speaker} sov/ {text}`

### shot_label
Appended to description before duration. Values: `ws`, `bs`, `cs`, `fs`, `ins`, `TD`, `TU`, `TR`, `pan`, or omitted (null).
- **Default is null** — most shots have no label
- Only used for: stationary curating (ws/bs), key interviews (bs/ws), object close-ups (cs/BS), camera moves (TD/TU/pan), inserts (ins), travel (TR), full-frame (fs)
- Must be null for: drone shots, action shots, casual talking, landscapes, CG/graphics

### Caption Type Rules
- `자막` — general text overlays, titles, editorial captions, informational text NOT starting with a proper name
- `정보자막` — ONLY when first line is a proper name/title (person, place, org) followed by factual info
- `말자막` — ONLY when visible on-screen subtitle text directly echoes spoken dialogue

## Gemini Integration

- Model: `gemini-3.1-pro-preview`
- Uses `@google/genai` SDK with `generateContent()` and video `fileData` with `startOffset`/`endOffset`
- Hand-written JSON schema for `responseSchema` (NOT zod's `z.toJSONSchema()` — causes Gemini nesting depth error)
- Segment padding: 2 seconds on each side of the time window sent to Gemini
- Response: structured JSON with `shots[]` containing `shot_index`, `description`, `shot_label`, `sov_cues[]`, `captions[]`
- `shot_label` returned separately from description — merged in code as `${description} ${shot_label}` (with dedup check)
- Exponential backoff retry on failure (2^attempt * 1000ms)

## HWPX Generation

Built from `empty.hwpx` template:
1. Unzip template to temp dir
2. Inject `<hp:p>` paragraphs before `</hs:sec>` in `Contents/section0.xml`
3. Update `Preview/PrvText.txt` — preserve template header, append shot content
4. Re-zip with mimetype as first uncompressed entry (`zip -0` then `zip -r`)

### Paragraph Styles
- `paraPrIDRef="13"` — Description lines (with `-` dash bullet via `<hh:heading type="BULLET" idRef="1" level="0"/>`)
- `paraPrIDRef="12"` — Content lines (captions, SOV cues, continuations, blank separators)
- `charPrIDRef="9"` — All runs
- `flags="2490368"` for bullet paragraphs (paraPr 13), `flags="393216"` for regular (paraPr 12)

### Tab Indentation
- `<hp:tab>` must be INSIDE `<hp:t>`, not outside
- Correct: `<hp:t><hp:tab width="4000" leader="0" type="1"/>text</hp:t>`

### Blank separators
- Empty `<hp:run charPrIDRef="9"/>` paragraph between shots
- One blank separator paragraph prepended before first shot content

## Shot Detection Details

- Model: TransNetV2 (weights auto-downloaded to `.models/transnetv2/`)
- TF 2.16.2 + tensorflow-metal 1.2.0 on Apple M1 Pro
- **Requires `TF_USE_LEGACY_KERAS=1`** — Keras 3 (TF 2.16 default) breaks Metal GPU; legacy Keras 2 restores GPU acceleration
- Batched inference: 128-window batches, 100-frame windows with 50-frame stride
- Frame extraction: ffmpeg downscales to 48x27 for TransNetV2 input
- Caching: `{video}.transnet_frames.npz` (extracted frames) and `{video}.transnet_preds.npy` (predictions) stored next to source video; invalidated on video file change
- Changing `--threshold` reuses cached predictions (no re-extraction or re-inference)
- Duration rounding: nearest integer, but 0.5 rounds down

## Environment

- Runtime: Bun
- Python: 3.12 (`.venv/`), deps: tensorflow-macos 2.16.2, tensorflow-metal 1.2.0, tf_keras 2.16.0
- `GEMINI_API_KEY` env var required for generate
- Dependencies: `@google/genai` (Gemini SDK), `openai` (legacy scripts), `zod` (listed but unused — JSON schema is hand-written)
- Reference cue sheets: `mongolia/01/tc-mongolia-01.hwpx`, `mongolia/02/tc-mongolia-02.hwpx`
- Setup: `bun run setup` (creates `.venv/`, installs tensorflow-macos + tensorflow-metal)

## Legacy Scripts (replaced by generate.ts)

- `scripts/transcribe.ts` — RTZR transcription
- `scripts/transcribe-openai.ts` — OpenAI transcription
- `scripts/transcription.ts` — Transcription processing
- `scripts/postprocess-transcript.ts` — Post-process transcripts
- `scripts/finalize.ts` — Merge scenes + transcript
- `scripts/describe.ts` — Gemini frame-based descriptions (per-shot image upload approach, superseded by video-based generate.ts)
