#!/bin/bash
set -e

echo "Setting up Python environment for TransNetV2..."

# Find compatible Python (3.9-3.12)
PYTHON=""
for v in python3.12 python3.11 python3.10 python3.9; do
  if command -v $v &> /dev/null; then
    PYTHON=$v
    break
  fi
done

if [ -z "$PYTHON" ]; then
  echo "Error: Python 3.9-3.12 required. Install with: brew install python@3.12"
  exit 1
fi

echo "Using $PYTHON"

# Create virtual environment
$PYTHON -m venv .venv
source .venv/bin/activate

# Install M1-optimized TensorFlow
echo "Installing tensorflow-macos and tensorflow-metal..."
pip install --upgrade pip
pip install tensorflow-macos tensorflow-metal numpy pillow

echo ""
echo "Setup complete! Run with:"
echo "  bun run detect <video-path>"
