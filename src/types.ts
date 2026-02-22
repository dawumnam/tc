export interface SovCue {
  speaker: string;
  text: string;
}

export interface Shot {
  start_frame: number;
  end_frame: number;
  start_time: number;
  end_time: number;
  duration: number;
  rounded_duration: number;
  sov_cues?: SovCue[];
  description?: string;
}

export interface DetectionResult {
  video: string;
  fps: number;
  total_frames: number;
  shots: Shot[];
}

export interface Caption {
  type: "자막" | "정보자막" | "말자막";
  text: string;
}

export interface GeneratedShot extends Shot {
  description: string;
  sov_cues: SovCue[];
  captions: Caption[];
}

export interface GenerateResult {
  video: string;
  fps: number;
  total_frames: number;
  shots: GeneratedShot[];
}

export interface DetectionError {
  error: string;
}

export type DetectionOutput = DetectionResult | DetectionError;

export interface TranscriptionWord {
  word: string;
  startOffset?: string;
  endOffset?: string;
  speakerLabel?: string;
}

export interface TranscriptionResult {
  results: {
    alternatives: {
      words: TranscriptionWord[];
    }[];
  }[];
}
