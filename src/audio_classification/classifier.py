"""Wrapper around MediaPipe's AudioClassifier task."""
import argparse
import os

from mediapipe.tasks.python import audio
from mediapipe.tasks.python.components import containers
from mediapipe.tasks.python.core.base_options import BaseOptions

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models",
    "yamnet.tflite",
)


class AudioClassifier:
    """Classifies audio clips (.wav) into sound event categories."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, max_results: int = 5):
        options = audio.AudioClassifierOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            max_results=max_results,
        )
        self._classifier = audio.AudioClassifier.create_from_options(options)

    def classify(self, wav_path: str):
        audio_data = containers.AudioData.create_from_wav_file(wav_path)
        results = self._classifier.classify(audio_data)
        return results

    def classify_top_label(self, wav_path: str):
        results = self.classify(wav_path)
        top_category = results[0].classifications[0].categories[0]
        return top_category.category_name, top_category.score

    def close(self):
        self._classifier.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main():
    parser = argparse.ArgumentParser(description="Classify a .wav clip with MediaPipe AudioClassifier")
    parser.add_argument("wav_path", help="Path to a mono 16-bit PCM .wav file")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to the .tflite classifier model")
    parser.add_argument("--max-results", type=int, default=5, help="Number of top categories to show")
    args = parser.parse_args()

    with AudioClassifier(model_path=args.model, max_results=args.max_results) as classifier:
        results = classifier.classify(args.wav_path)
        for classification_result in results:
            timestamp = classification_result.timestamp_ms
            categories = classification_result.classifications[0].categories
            print(f"Segment @ {timestamp}ms:")
            for category in categories:
                print(f"  {category.category_name}: {category.score:.3f}")


if __name__ == "__main__":
    main()
