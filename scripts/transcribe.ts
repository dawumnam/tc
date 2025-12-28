/**
 * RTZR STT API client for transcription with word timestamps and diarization.
 *
 * Usage:
 *   bun run transcribe <audio-file> [--output <file.json>]
 *
 * Environment variables:
 *   RTZR_CLIENT_ID     - API client ID
 *   RTZR_CLIENT_SECRET - API client secret
 */

const BASE_URL = "https://openapi.vito.ai";

interface TokenResponse {
  access_token: string;
  expire_at: number;
}

interface TranscribeResponse {
  id: string;
}

interface Word {
  word: string;
  startOffset?: string;
  endOffset?: string;
  speakerLabel?: string;
}

interface Utterance {
  start_at: number;
  duration: number;
  msg: string;
  spk: number;
  lang: string;
  words?: Word[];
}

interface TranscriptionResult {
  id: string;
  status: "transcribing" | "completed" | "failed";
  results?: {
    utterances: Utterance[];
  };
  error?: {
    code: string;
    message: string;
  };
}

let cachedToken: TokenResponse | null = null;

async function getToken(): Promise<string> {
  const clientId = process.env.RTZR_CLIENT_ID;
  const clientSecret = process.env.RTZR_CLIENT_SECRET;

  if (!clientId || !clientSecret) {
    throw new Error("Missing RTZR_CLIENT_ID or RTZR_CLIENT_SECRET environment variables");
  }

  // Reuse token if not expired (with 30 min buffer)
  if (cachedToken && cachedToken.expire_at > Date.now() / 1000 + 1800) {
    return cachedToken.access_token;
  }

  const resp = await fetch(`${BASE_URL}/v1/authenticate`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ client_id: clientId, client_secret: clientSecret }),
  });

  if (!resp.ok) {
    throw new Error(`Authentication failed: ${resp.status} ${await resp.text()}`);
  }

  cachedToken = (await resp.json()) as TokenResponse;
  return cachedToken.access_token;
}

async function submitTranscription(filePath: string): Promise<string> {
  const token = await getToken();
  const file = Bun.file(filePath);

  if (!(await file.exists())) {
    throw new Error(`File not found: ${filePath}`);
  }

  const config = {
    model_name: "sommers",
    use_diarization: true,
    use_word_timestamp: true,
    use_itn: true,
    use_disfluency_filter: true,
    use_paragraph_splitter: true,
    paragraph_splitter: { max: 50 },
    domain: "GENERAL",
  };

  const formData = new FormData();
  formData.append("file", file);
  formData.append("config", JSON.stringify(config));

  console.error(`Submitting ${filePath} for transcription...`);

  const resp = await fetch(`${BASE_URL}/v1/transcribe`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: formData,
  });

  if (!resp.ok) {
    throw new Error(`Submit failed: ${resp.status} ${await resp.text()}`);
  }

  const result = (await resp.json()) as TranscribeResponse;
  console.error(`Transcription ID: ${result.id}`);
  return result.id;
}

async function pollTranscription(transcribeId: string): Promise<TranscriptionResult> {
  const token = await getToken();
  const pollInterval = 5000; // 5 seconds
  const timeout = 3600000; // 1 hour
  const deadline = Date.now() + timeout;

  while (Date.now() < deadline) {
    const resp = await fetch(`${BASE_URL}/v1/transcribe/${transcribeId}`, {
      headers: { Authorization: `Bearer ${token}` },
    });

    if (!resp.ok) {
      throw new Error(`Poll failed: ${resp.status} ${await resp.text()}`);
    }

    const result = (await resp.json()) as TranscriptionResult;

    if (result.status === "completed" || result.status === "failed") {
      return result;
    }

    console.error(`Status: ${result.status}, waiting ${pollInterval / 1000}s...`);
    await Bun.sleep(pollInterval);
  }

  throw new Error("Timeout waiting for transcription");
}

function convertToTranscriptionFormat(result: TranscriptionResult): object {
  // Convert RTZR format to our expected format for --transcription-file
  const words: Word[] = [];

  if (result.results?.utterances) {
    for (const utterance of result.results.utterances) {
      if (utterance.words) {
        for (const word of utterance.words) {
          words.push({
            word: word.word,
            startOffset: word.startOffset,
            endOffset: word.endOffset,
            speakerLabel: String(utterance.spk),
          });
        }
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

async function main() {
  const args = Bun.argv.slice(2);

  if (args.length === 0 || args.includes("--help") || args.includes("-h")) {
    console.log(`Usage: bun run scripts/transcribe.ts <audio-file> [options]

Options:
  --output <file>  Write result to JSON file
  --raw            Output raw RTZR response (not converted)
  --help, -h       Show this help

Environment:
  RTZR_CLIENT_ID     API client ID
  RTZR_CLIENT_SECRET API client secret`);
    process.exit(0);
  }

  let audioPath = "";
  let outputPath: string | null = null;
  let raw = false;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "--output" && args[i + 1]) {
      outputPath = args[++i];
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

  const transcribeId = await submitTranscription(audioPath);
  const result = await pollTranscription(transcribeId);

  if (result.status === "failed") {
    console.error(`Transcription failed: ${result.error?.code} - ${result.error?.message}`);
    process.exit(1);
  }

  const output = raw ? result : convertToTranscriptionFormat(result);
  const json = JSON.stringify(output, null, 2);

  if (outputPath) {
    await Bun.write(outputPath, json);
    console.error(`Result written to ${outputPath}`);
  } else {
    console.log(json);
  }
}

main().catch((err) => {
  console.error(`Error: ${err.message}`);
  process.exit(1);
});
