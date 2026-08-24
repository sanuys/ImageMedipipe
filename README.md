# ImageMedipipe

Python utilities built on [MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/guide) for:

- **Image Embedding** — turn images into feature vectors and compare them by cosine similarity.
- **Audio Classification** — classify `.wav` audio clips into sound event categories (e.g. speech, music, animal sounds) using the YAMNet model.

## Project structure

```
ImageMedipipe/
├── models/
│   └── download_models.py     # downloads the .tflite models used below
├── src/
│   ├── image_embedding/
│   │   └── embedder.py        # ImageEmbedder wrapper + CLI
│   └── audio_classification/
│       └── classifier.py      # AudioClassifier wrapper + CLI
└── examples/
    ├── embed_images_example.py
    └── classify_audio_example.py
```

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   ```

2. Download the pretrained models (saved to `models/`, not committed to git):

   ```bash
   python models/download_models.py
   ```

   This fetches:
   - `mobilenet_v3_small.tflite` for image embedding
   - `yamnet.tflite` for audio classification

## Usage

### Image embedding

Compare two images and print their cosine similarity:

```bash
python src/image_embedding/embedder.py path/to/a.jpg path/to/b.jpg
```

Or from Python:

```python
from src.image_embedding.embedder import ImageEmbedder

with ImageEmbedder() as embedder:
    similarity = embedder.compare("a.jpg", "b.jpg")
    print(similarity)
```

### Audio classification

Classify a mono 16-bit PCM `.wav` file:

```bash
python src/audio_classification/classifier.py path/to/clip.wav
```

Or from Python:

```python
from src.audio_classification.classifier import AudioClassifier

with AudioClassifier() as classifier:
    label, score = classifier.classify_top_label("clip.wav")
    print(label, score)
```

### Examples

Runnable end-to-end scripts live in `examples/`:

```bash
python examples/embed_images_example.py path/to/a.jpg path/to/b.jpg
python examples/classify_audio_example.py path/to/clip.wav
```

Drop sample images/audio into `examples/sample_data/` for quick local testing.

## References

- [MediaPipe Image Embedder guide](https://ai.google.dev/edge/mediapipe/solutions/vision/image_embedder)
- [MediaPipe Audio Classifier guide](https://ai.google.dev/edge/mediapipe/solutions/audio/audio_classifier)
