#!/usr/bin/env python3
"""
TransNetV2 shot boundary detection script.
Outputs JSON to stdout for Bun CLI consumption.
"""

import argparse
import json
import sys
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF logging

import numpy as np
import tensorflow as tf


class TransNetV2:
    """TransNetV2 model wrapper for shot boundary detection."""

    MODEL_URL = "https://github.com/soCzech/TransNetV2/raw/master/inference/transnetv2-weights/"
    WEIGHTS_FILES = [
        "saved_model.pb",
        "variables/variables.index",
        "variables/variables.data-00000-of-00001",
    ]

    def __init__(self, model_dir: str | None = None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "..", ".models", "transnetv2")

        self.model_dir = model_dir
        self._model = None

    def _ensure_model(self):
        """Download and load model weights if not present."""
        if self._model is not None:
            return

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "variables"), exist_ok=True)

        model_pb = os.path.join(self.model_dir, "saved_model.pb")
        if not os.path.exists(model_pb):
            self._download_weights()

        self._model = tf.saved_model.load(self.model_dir)

    def _download_weights(self):
        """Download TransNetV2 weights from GitHub."""
        import urllib.request

        print("Downloading TransNetV2 weights...", file=sys.stderr)
        base_url = self.MODEL_URL

        for weight_file in self.WEIGHTS_FILES:
            url = base_url + weight_file
            dest = os.path.join(self.model_dir, weight_file)
            print(f"  Downloading {weight_file}...", file=sys.stderr)
            urllib.request.urlretrieve(url, dest)

        print("Download complete.", file=sys.stderr)

    def _log(self, msg: str):
        """Log progress to stderr."""
        print(msg, file=sys.stderr, flush=True)

    def _extract_frames(self, video_path: str) -> tuple[np.ndarray, float, int]:
        """Extract frames from FHD video, downscale to 48x27 for model input."""
        import subprocess

        self._log("Probing video...")

        # Get video info using ffprobe
        probe_cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,nb_frames,width,height",
            "-of", "json",
            video_path,
        ]

        try:
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(probe_result.stdout)
            stream = probe_data["streams"][0]

            fps_parts = stream["r_frame_rate"].split("/")
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
            total_frames = int(stream.get("nb_frames", 0))

        except (subprocess.CalledProcessError, FileNotFoundError, KeyError):
            fps = 30.0
            total_frames = 0

        self._log("Extracting frames...")

        # Extract and downscale frames: FHD -> 48x27 for TransNetV2
        extract_cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", "scale=48:27",
            "-pix_fmt", "rgb24",
            "-f", "rawvideo",
            "-",
        ]

        result = subprocess.run(extract_cmd, capture_output=True, check=True)
        raw_frames = result.stdout

        frame_size = 48 * 27 * 3
        num_frames = len(raw_frames) // frame_size

        if total_frames == 0:
            total_frames = num_frames

        frames = np.frombuffer(raw_frames, dtype=np.uint8)
        frames = frames.reshape((num_frames, 27, 48, 3))

        self._log(f"Extracted {num_frames} frames ({fps:.2f} fps)")

        return frames, fps, total_frames

    def predict(self, video_path: str, threshold: float = 0.5) -> dict:
        """Run shot boundary detection on a video."""
        self._log("Loading model...")
        self._ensure_model()

        frames, fps, total_frames = self._extract_frames(video_path)
        num_frames = len(frames)

        # Cast to float32 but keep 0-255 range (model handles normalization internally)
        frames_float = frames.astype(np.float32)

        window_size = 100
        step_size = 50  # Overlap by 50, take center predictions

        # Pad frames for clean windowing
        pad_start = 25
        pad_end = 25 + (window_size - (num_frames % window_size)) % window_size
        frames_padded = np.pad(
            frames_float,
            ((pad_start, pad_end), (0, 0), (0, 0), (0, 0)),
            mode="edge"
        )

        padded_len = len(frames_padded)
        total_windows = (padded_len - window_size) // step_size + 1
        all_predictions = []

        self._log("Running inference...")

        for window_idx in range(total_windows):
            start = window_idx * step_size
            end = start + window_size
            batch = frames_padded[start:end]

            batch_tensor = tf.constant(batch[np.newaxis, ...], dtype=tf.float32)
            result = self._model(batch_tensor)

            # Debug first batch
            if window_idx == 0:
                self._log(f"Model output type: {type(result)}")
                if isinstance(result, tuple):
                    self._log(f"Tuple length: {len(result)}")
                    for i, r in enumerate(result):
                        if hasattr(r, 'shape'):
                            self._log(f"  [{i}] shape: {r.shape}, min: {float(tf.reduce_min(r)):.4f}, max: {float(tf.reduce_max(r)):.4f}")
                        elif isinstance(r, dict):
                            self._log(f"  [{i}] dict keys: {list(r.keys())}")

            # Extract logits (first element of tuple)
            if isinstance(result, tuple):
                logits = result[0]
            else:
                logits = result

            # Apply sigmoid to convert logits to probabilities
            pred = tf.sigmoid(logits).numpy()

            # Debug shape
            if window_idx == 0:
                self._log(f"Pred shape after sigmoid: {pred.shape}")

            # Handle different possible shapes
            if pred.ndim == 3:
                pred = pred[0, :, 0]
            elif pred.ndim == 2:
                pred = pred[0, :]

            # Take center 50 frames (indices 25-74)
            center_pred = pred[25:75]
            all_predictions.append(center_pred)

            progress = int(((window_idx + 1) / total_windows) * 100)
            self._log(f"Progress: {progress}% ({window_idx + 1}/{total_windows})")

        # Concatenate and trim to original length
        predictions = np.concatenate(all_predictions)[:num_frames]

        if not all_predictions:
            self._log("Done. No shots detected.")
            return {
                "video": video_path,
                "fps": fps,
                "total_frames": total_frames,
                "shots": [],
            }

        predictions = np.concatenate(all_predictions)[:num_frames]

        # Debug: show prediction stats
        self._log(f"Predictions min={predictions.min():.4f} max={predictions.max():.4f} mean={predictions.mean():.4f}")

        shot_boundaries = np.where(predictions > threshold)[0].tolist()
        self._log(f"Found {len(shot_boundaries)} boundaries above threshold {threshold}")

        def round_nearest(x):
            """Round to nearest 1, but .5 rounds down"""
            import math
            if x - math.floor(x) == 0.5:
                return int(math.floor(x))
            return round(x)

        def make_shot(start_f, end_f):
            start_t = round(start_f / fps, 3)
            end_t = round(end_f / fps, 3)
            duration = round(end_t - start_t, 3)
            return {
                "start_frame": int(start_f),
                "end_frame": int(end_f),
                "start_time": start_t,
                "end_time": end_t,
                "duration": duration,
                "rounded_duration": round_nearest(duration),
            }

        shots = []
        start_frame = 0

        for boundary in shot_boundaries:
            if boundary > start_frame:
                shots.append(make_shot(start_frame, boundary))
            start_frame = boundary + 1

        if start_frame < num_frames:
            shots.append(make_shot(start_frame, num_frames - 1))

        self._log(f"Done. Found {len(shots)} shots.")

        return {
            "video": video_path,
            "fps": fps,
            "total_frames": total_frames,
            "shots": shots,
        }


def main():
    parser = argparse.ArgumentParser(description="Detect shot boundaries using TransNetV2")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (0-1)")
    parser.add_argument("--model-dir", help="Custom model directory")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(json.dumps({"error": f"Video file not found: {args.video}"}))
        sys.exit(1)

    try:
        model = TransNetV2(model_dir=args.model_dir)
        result = model.predict(args.video, threshold=args.threshold)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
