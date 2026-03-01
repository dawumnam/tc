#!/usr/bin/env python3
"""
TransNetV2 shot boundary detection script (PyTorch).
Outputs JSON to stdout for Bun CLI consumption.
"""

import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch
from transnetv2_pytorch import TransNetV2


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def cache_path(video_path: str, suffix: str) -> str:
    return video_path + suffix


def cache_valid(video_path: str, cp: str) -> bool:
    if not os.path.exists(cp):
        return False
    return os.path.getmtime(cp) >= os.path.getmtime(video_path)


def get_fps(video_path: str) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "json", video_path],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(result.stdout)
        parts = data["streams"][0]["r_frame_rate"].split("/")
        return float(parts[0]) / float(parts[1]) if len(parts) == 2 else float(parts[0])
    except Exception:
        return 30.0


def round_nearest(x):
    """Round to nearest 1, but .5 rounds down."""
    if x - math.floor(x) == 0.5:
        return int(math.floor(x))
    return round(x)


def predict(video_path: str, threshold: float = 0.5) -> dict:
    fps = get_fps(video_path)
    log(f"Video FPS: {fps:.2f}")

    preds_cache = cache_path(video_path, ".transnet_preds.npy")

    if cache_valid(video_path, preds_cache):
        log("Loading cached predictions...")
        predictions = np.load(preds_cache)
        log(f"Loaded cached predictions ({len(predictions)} frames)")
    else:
        log("Loading model...")
        model = TransNetV2(device="cpu")
        model.eval()

        log("Running inference...")
        # Redirect stdout to stderr so library print() calls don't corrupt JSON output
        _orig_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            video_frames, single_frame_pred, all_frame_pred = model.predict_video(video_path)
        finally:
            sys.stdout = _orig_stdout
        predictions = single_frame_pred.numpy() if hasattr(single_frame_pred, 'numpy') else np.array(single_frame_pred)
        if predictions.ndim > 1:
            predictions = predictions[:, 0]
        log(f"Inference done. {len(predictions)} frames")

        np.save(preds_cache, predictions)
        log(f"Cached predictions to {preds_cache}")

    num_frames = len(predictions)
    total_frames = num_frames

    log(f"Predictions min={predictions.min():.4f} max={predictions.max():.4f} mean={predictions.mean():.4f}")

    shot_boundaries = np.where(predictions > threshold)[0].tolist()
    log(f"Found {len(shot_boundaries)} boundaries above threshold {threshold}")

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

    log(f"Done. Found {len(shots)} shots.")

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

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(json.dumps({"error": f"Video file not found: {args.video}"}))
        sys.exit(1)

    try:
        result = predict(args.video, threshold=args.threshold)
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
