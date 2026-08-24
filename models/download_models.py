"""Download the pretrained .tflite models used by this project.

Run:
    python models/download_models.py
"""
import os
import urllib.request

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = {
    "mobilenet_v3_small.tflite": (
        "https://storage.googleapis.com/mediapipe-models/image_embedder/"
        "mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite"
    ),
    "yamnet.tflite": (
        "https://storage.googleapis.com/mediapipe-models/audio_classifier/"
        "yamnet/float32/1/yamnet.tflite"
    ),
}


def download(name: str, url: str) -> None:
    dest = os.path.join(MODELS_DIR, name)
    if os.path.exists(dest):
        print(f"[skip] {name} already exists")
        return
    print(f"[download] {name} <- {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"[done] saved to {dest}")


if __name__ == "__main__":
    for filename, url in MODELS.items():
        download(filename, url)
