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
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Keras 3 breaks Metal GPU; use Keras 2

import numpy as np
import tensorflow as tf

# Ensure Metal GPU is used when available
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"Using GPU: {gpus[0].name}", file=sys.stderr)
else:
    print("WARNING: No GPU found, running on CPU", file=sys.stderr)


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

    def _cache_path(self, video_path: str, suffix: str) -> str:
        """Return cache file path next to the source video."""
        return video_path + suffix

    def _cache_valid(self, video_path: str, cache_path: str) -> bool:
        """Check if cache exists and is newer than the video file."""
        if not os.path.exists(cache_path):
            return False
        return os.path.getmtime(cache_path) >= os.path.getmtime(video_path)

    def _extract_frames(self, video_path: str) -> tuple[np.ndarray, float, int]:
        """Extract frames from FHD video, downscale to 48x27 for model input."""
        import subprocess

        frames_cache = self._cache_path(video_path, ".transnet_frames.npz")

        if self._cache_valid(video_path, frames_cache):
            self._log("Loading cached frames...")
            data = np.load(frames_cache)
            frames = data["frames"]
            fps = float(data["fps"])
            total_frames = int(data["total_frames"])
            self._log(f"Loaded {len(frames)} cached frames ({fps:.2f} fps)")
            return frames, fps, total_frames

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

        # Cache frames next to video
        np.savez_compressed(frames_cache, frames=frames, fps=fps, total_frames=total_frames)
        self._log(f"Cached frames to {frames_cache}")

        return frames, fps, total_frames

    def _run_inference(self, video_path: str, frames: np.ndarray) -> np.ndarray:
        """Run batched TransNetV2 inference. Returns per-frame predictions."""
        num_frames = len(frames)
        frames_float = frames.astype(np.float32)

        window_size = 100
        step_size = 50

        pad_start = 25
        pad_end = 25 + (window_size - (num_frames % window_size)) % window_size
        frames_padded = np.pad(
            frames_float,
            ((pad_start, pad_end), (0, 0), (0, 0), (0, 0)),
            mode="edge"
        )

        padded_len = len(frames_padded)
        total_windows = (padded_len - window_size) // step_size + 1

        self._log("Running inference...")

        max_batch = 128
        all_predictions = []

        for batch_start in range(0, total_windows, max_batch):
            batch_end = min(batch_start + max_batch, total_windows)

            windows = np.stack([
                frames_padded[i * step_size : i * step_size + window_size]
                for i in range(batch_start, batch_end)
            ])

            with tf.device("/GPU:0" if gpus else "/CPU:0"):
                batch_tensor = tf.constant(windows, dtype=tf.float32)
                result = self._model(batch_tensor)

            if isinstance(result, tuple):
                logits = result[0]
            else:
                logits = result

            pred = tf.sigmoid(logits).numpy()

            if pred.ndim == 3:
                pred = pred[:, :, 0]

            center_preds = pred[:, 25:75]
            all_predictions.append(center_preds.reshape(-1))

            self._log(f"Progress: {batch_end}/{total_windows} windows")

        predictions = np.concatenate(all_predictions)[:num_frames]
        return predictions

    def predict(self, video_path: str, threshold: float = 0.5) -> dict:
        """Run shot boundary detection on a video."""
        frames, fps, total_frames = self._extract_frames(video_path)
        num_frames = len(frames)

        preds_cache = self._cache_path(video_path, ".transnet_preds.npy")

        if self._cache_valid(video_path, preds_cache):
            self._log("Loading cached predictions...")
            predictions = np.load(preds_cache)
            if len(predictions) == num_frames:
                self._log(f"Loaded cached predictions ({len(predictions)} frames)")
            else:
                self._log("Cache frame count mismatch, re-running inference...")
                self._log("Loading model...")
                self._ensure_model()
                predictions = self._run_inference(video_path, frames)
                np.save(preds_cache, predictions)
                self._log(f"Cached predictions to {preds_cache}")
        else:
            self._log("Loading model...")
            self._ensure_model()
            predictions = self._run_inference(video_path, frames)
            np.save(preds_cache, predictions)
            self._log(f"Cached predictions to {preds_cache}")

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
