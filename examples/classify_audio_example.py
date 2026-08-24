"""Example: classify a .wav audio clip and print the top predicted sounds.

Usage:
    python examples/classify_audio_example.py path/to/clip.wav
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from audio_classification.classifier import AudioClassifier  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Classify a .wav clip using MediaPipe AudioClassifier")
    parser.add_argument("wav_path")
    parser.add_argument("--max-results", type=int, default=5)
    args = parser.parse_args()

    with AudioClassifier(max_results=args.max_results) as classifier:
        label, score = classifier.classify_top_label(args.wav_path)
        print(f"Top prediction for '{args.wav_path}': {label} ({score:.3f})")


if __name__ == "__main__":
    main()
