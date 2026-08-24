"""Example: compare two images and print their embedding similarity.

Usage:
    python examples/embed_images_example.py path/to/a.jpg path/to/b.jpg
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from image_embedding.embedder import ImageEmbedder  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Compare two images using MediaPipe ImageEmbedder")
    parser.add_argument("image_a")
    parser.add_argument("image_b")
    args = parser.parse_args()

    with ImageEmbedder() as embedder:
        similarity = embedder.compare(args.image_a, args.image_b)
        print(f"'{args.image_a}' vs '{args.image_b}'")
        print(f"Cosine similarity: {similarity:.4f}")


if __name__ == "__main__":
    main()
