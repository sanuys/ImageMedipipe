"""Wrapper around MediaPipe's ImageEmbedder task."""
import argparse
import os

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "mobilenet_v3_small.tflite",
)


class ImageEmbedder:
    """Embeds images into feature vectors and compares them for similarity."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, quantize: bool = False):
        options = vision.ImageEmbedderOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            quantize=quantize,
            running_mode=vision.RunningMode.IMAGE,
        )
        self._embedder = vision.ImageEmbedder.create_from_options(options)

    def embed(self, image_path: str):
        mp_image = mp.Image.create_from_file(image_path)
        result = self._embedder.embed(mp_image)
        return result.embeddings[0]

    @staticmethod
    def cosine_similarity(embedding_a, embedding_b) -> float:
        return vision.ImageEmbedder.cosine_similarity(embedding_a, embedding_b)

    def compare(self, image_path_a: str, image_path_b: str) -> float:
        embedding_a = self.embed(image_path_a)
        embedding_b = self.embed(image_path_b)
        return self.cosine_similarity(embedding_a, embedding_b)

    def close(self):
        self._embedder.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    parser = argparse.ArgumentParser(description="Compare two images with MediaPipe ImageEmbedder")
    parser.add_argument("image_a", help="Path to the first image")
    parser.add_argument("image_b", help="Path to the second image")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to the .tflite embedder model")
    args = parser.parse_args()

    with ImageEmbedder(model_path=args.model) as embedder:
        similarity = embedder.compare(args.image_a, args.image_b)
        print(f"Cosine similarity: {similarity:.4f}")


if __name__ == "__main__":
    main()
