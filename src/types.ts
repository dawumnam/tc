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
