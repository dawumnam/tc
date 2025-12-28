import type { DetectionOutput } from "./types";

const PYTHON_PATH = ".venv/bin/python";
const SCRIPT_PATH = "scripts/detect_shots.py";

export async function detectShots(videoPath: string, threshold: number): Promise<DetectionOutput> {
  const file = Bun.file(videoPath);
  if (!(await file.exists())) {
    return { error: `Video file not found: ${videoPath}` };
  }

  const proc = Bun.spawn([PYTHON_PATH, SCRIPT_PATH, videoPath, "--threshold", threshold.toString()], {
    stdout: "pipe",
    stderr: "inherit",
  });

  const stdout = await new Response(proc.stdout).text();
  const exitCode = await proc.exited;

  if (exitCode !== 0) {
    return { error: `Python process exited with code ${exitCode}` };
  }

  try {
    return JSON.parse(stdout) as DetectionOutput;
  } catch {
    return { error: `Failed to parse output: ${stdout}` };
  }
}
