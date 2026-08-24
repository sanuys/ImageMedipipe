"""
MediaPipe Tasks - Flask Web UI
=========================================
เว็บแอปที่รวมสอง demo ของ MediaPipe Tasks ไว้ในที่เดียว:

1. Image Embedder - เปรียบเทียบความคล้ายกันของภาพ 2 ภาพ (โมเดล MobileNet V3 Small)
   https://github.com/googlesamples/mediapipe/blob/main/examples/image_embedder/python/image_embedder.ipynb
2. Audio Classifier - จำแนกประเภทเสียงตามช่วงเวลา (โมเดล YAMNet / AudioSet)
   https://github.com/googlesamples/mediapipe/blob/main/examples/audio_classifier/python/audio_classification.ipynb

วิธีรัน:
    pip install -r requirements.txt
    python app.py

หมายเหตุ: ฟีเจอร์ Audio Classifier ต้องติดตั้ง ffmpeg ไว้ในเครื่อง/เซิร์ฟเวอร์ด้วย
(ใช้แปลงไฟล์เสียงหลากหลายฟอร์แมตให้เป็น WAV ก่อนส่งเข้า MediaPipe) ดูวิธีติดตั้งใน README

ครั้งแรกที่รัน แอปจะดาวน์โหลดไฟล์โมเดลและไฟล์ตัวอย่างของทั้งสองฟีเจอร์มาเก็บไว้
ในเครื่องโดยอัตโนมัติ (ต้องมีอินเทอร์เน็ตในตอนนั้น) หลังจากนั้นจะใช้ไฟล์ที่
ดาวน์โหลดไว้แล้วโดยไม่ต้องดาวน์โหลดซ้ำ
"""

import math
import os
import subprocess
import threading
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from scipy.io import wavfile
from werkzeug.utils import secure_filename

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import audio
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components import containers

# ---------------------------------------------------------------------------
# ค่าคงที่ / พาธไฟล์
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
SAMPLES_DIR = BASE_DIR / "static" / "samples"
UPLOAD_DIR = BASE_DIR / "uploads"

MODEL_PATH = MODEL_DIR / "embedder.tflite"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite"
)

# รูปภาพตัวอย่าง เหมือนกับที่ใช้ใน notebook ต้นฉบับของ Google
SAMPLE_ASSETS_BASE_URL = "https://storage.googleapis.com/mediapipe-assets/"
SAMPLE_IMAGES = ["burger.jpg", "burger_crop.jpg"]

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # จำกัดขนาด request รวม 10 MB

# --- Audio Classifier (YAMNet / AudioSet) -----------------------------------

AUDIO_MODEL_PATH = MODEL_DIR / "classifier.tflite"
AUDIO_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "audio_classifier/yamnet/float32/1/yamnet.tflite"
)

AUDIO_SAMPLES_DIR = BASE_DIR / "static" / "audio_samples"
AUDIO_SAMPLE_FILES = ["speech_16000_hz_mono.wav"]

# ffmpeg รองรับฟอร์แมตเสียงหลากหลาย ลิสต์นี้ใช้ตรวจเบื้องต้นจากนามสกุลไฟล์เท่านั้น
# (ตัวตัดสินจริงคือ ffmpeg ตอนแปลงไฟล์)
ALLOWED_AUDIO_EXTENSIONS = {
    "wav", "mp3", "m4a", "aac", "ogg", "oga", "flac",
    "wma", "opus", "webm", "mp4", "3gp", "aiff", "aif",
}
AUDIO_MAX_RESULTS = 4
# ปกติใช้แค่ "ffmpeg" เฉย ๆ (ต้องอยู่ใน PATH) แต่ถ้า PATH มีปัญหา (พบบ่อยบน Windows
# หลังติดตั้งด้วย winget/choco ที่ shell เดิมยังไม่เห็น PATH ใหม่) สามารถตั้งค่า
# environment variable FFMPEG_BIN ให้ชี้ไปที่ ffmpeg.exe แบบเต็มพาธได้โดยตรง เช่น:
#   $env:FFMPEG_BIN = "C:\ffmpeg\bin\ffmpeg.exe"   (PowerShell)
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")

_setup_lock = threading.Lock()
_setup_done = False


# ---------------------------------------------------------------------------
# เตรียมไฟล์ที่จำเป็น (โมเดล + รูปตัวอย่าง) แบบดาวน์โหลดอัตโนมัติ
# ---------------------------------------------------------------------------

def ensure_assets():
    """ดาวน์โหลดไฟล์โมเดลและรูปภาพตัวอย่าง หากยังไม่มีในเครื่อง (ทำครั้งเดียว)."""
    global _setup_done
    with _setup_lock:
        if _setup_done:
            return

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        AUDIO_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        if not MODEL_PATH.exists():
            print(f"[setup] กำลังดาวน์โหลดโมเดล image embedder -> {MODEL_PATH}")
            tmp_path = MODEL_PATH.with_suffix(".tmp")
            urllib.request.urlretrieve(MODEL_URL, tmp_path)
            tmp_path.rename(MODEL_PATH)
            print("[setup] ดาวน์โหลดโมเดล image embedder เสร็จแล้ว")
        else:
            print(f"[setup] พบไฟล์โมเดล image embedder อยู่แล้วที่ {MODEL_PATH}")

        for name in SAMPLE_IMAGES:
            dest = SAMPLES_DIR / name
            if not dest.exists():
                print(f"[setup] กำลังดาวน์โหลดรูปตัวอย่าง -> {dest}")
                tmp_path = dest.with_suffix(dest.suffix + ".tmp")
                urllib.request.urlretrieve(SAMPLE_ASSETS_BASE_URL + name, tmp_path)
                tmp_path.rename(dest)
        print("[setup] รูปตัวอย่างพร้อมใช้งาน")

        if not AUDIO_MODEL_PATH.exists():
            print(f"[setup] กำลังดาวน์โหลดโมเดล audio classifier (YAMNet) -> {AUDIO_MODEL_PATH}")
            tmp_path = AUDIO_MODEL_PATH.with_suffix(".tmp")
            urllib.request.urlretrieve(AUDIO_MODEL_URL, tmp_path)
            tmp_path.rename(AUDIO_MODEL_PATH)
            print("[setup] ดาวน์โหลดโมเดล audio classifier เสร็จแล้ว")
        else:
            print(f"[setup] พบไฟล์โมเดล audio classifier อยู่แล้วที่ {AUDIO_MODEL_PATH}")

        for name in AUDIO_SAMPLE_FILES:
            dest = AUDIO_SAMPLES_DIR / name
            if not dest.exists():
                print(f"[setup] กำลังดาวน์โหลดไฟล์เสียงตัวอย่าง -> {dest}")
                tmp_path = dest.with_suffix(dest.suffix + ".tmp")
                urllib.request.urlretrieve(SAMPLE_ASSETS_BASE_URL + name, tmp_path)
                tmp_path.rename(dest)
        print("[setup] ไฟล์เสียงตัวอย่างพร้อมใช้งาน")

        _setup_done = True


# ---------------------------------------------------------------------------
# ตัวช่วยจัดการไฟล์รูปภาพที่รับเข้ามา
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def resolve_image(prefix: str, form, files):
    """
    แปลง slot รูปภาพ ('image1' / 'image2') จาก request ให้เป็นพาธไฟล์จริง
    ลำดับความสำคัญ: ไฟล์ที่อัปโหลด > รูปตัวอย่างที่เลือก

    คืนค่า: (พาธไฟล์บนดิสก์, url สำหรับแสดงตัวอย่างในหน้าเว็บ, เป็นไฟล์ชั่วคราวหรือไม่)
    """
    file_field = f"{prefix}_file"
    sample_field = f"{prefix}_sample"

    upload = files.get(file_field)
    if upload and upload.filename:
        if not allowed_file(upload.filename):
            raise ValueError(f'ไฟล์ "{upload.filename}" ไม่ใช่รูปภาพที่รองรับ (jpg, png, bmp, webp)')
        ext = upload.filename.rsplit(".", 1)[1].lower()
        temp_name = f"{uuid.uuid4().hex}.{ext}"
        temp_path = UPLOAD_DIR / temp_name
        upload.save(temp_path)
        return temp_path, f"/uploads/{temp_name}", True

    sample_name = form.get(sample_field)
    if sample_name:
        sample_name = secure_filename(sample_name)
        if sample_name not in SAMPLE_IMAGES:
            raise ValueError("รูปตัวอย่างที่เลือกไม่ถูกต้อง")
        path = SAMPLES_DIR / sample_name
        if not path.exists():
            raise ValueError("ไม่พบไฟล์รูปตัวอย่างบนเซิร์ฟเวอร์ กรุณารีสตาร์ทแอปเพื่อดาวน์โหลดใหม่")
        return path, f"/static/samples/{sample_name}", False

    raise ValueError("กรุณาอัปโหลดรูปภาพ หรือเลือกจากตัวอย่าง")


# ---------------------------------------------------------------------------
# แกนหลัก: คำนวณ embedding และ cosine similarity ด้วย MediaPipe
# (โครงสร้างเดียวกับใน image_embedder.ipynb ต้นฉบับ)
# ---------------------------------------------------------------------------

def embed_and_compare(path1: Path, path2: Path, l2_normalize: bool, quantize: bool) -> float:
    base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.ImageEmbedderOptions(
        base_options=base_options,
        l2_normalize=l2_normalize,
        quantize=quantize,
    )

    with vision.ImageEmbedder.create_from_options(options) as embedder:
        first_image = mp.Image.create_from_file(str(path1))
        second_image = mp.Image.create_from_file(str(path2))

        first_embedding_result = embedder.embed(first_image)
        second_embedding_result = embedder.embed(second_image)

        similarity = vision.ImageEmbedder.cosine_similarity(
            first_embedding_result.embeddings[0],
            second_embedding_result.embeddings[0],
        )

    if similarity is None or math.isnan(similarity):
        raise ValueError("ไม่สามารถคำนวณค่าความคล้ายได้ กรุณาลองรูปภาพอื่น")

    return float(similarity)


# ---------------------------------------------------------------------------
# Audio Classifier (YAMNet / AudioSet) — จัดการไฟล์เสียงที่รับเข้ามา
# ---------------------------------------------------------------------------

def allowed_audio_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_AUDIO_EXTENSIONS


def resolve_audio(form, files):
    """
    แปลง slot ไฟล์เสียง ('audio_file' / 'audio_sample') จาก request ให้เป็นพาธไฟล์จริง
    ลำดับความสำคัญ: ไฟล์ที่อัปโหลด > ไฟล์ตัวอย่างที่เลือก

    คืนค่า: (พาธไฟล์บนดิสก์, url สำหรับเล่นไฟล์ในหน้าเว็บ, เป็นไฟล์ชั่วคราวหรือไม่)
    """
    upload = files.get("audio_file")
    if upload and upload.filename:
        if not allowed_audio_file(upload.filename):
            raise ValueError(
                f'ไฟล์ "{upload.filename}" ไม่ใช่ไฟล์เสียงที่รองรับ '
                f'(รองรับ: {", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))})'
            )
        ext = upload.filename.rsplit(".", 1)[1].lower()
        temp_name = f"{uuid.uuid4().hex}.{ext}"
        temp_path = UPLOAD_DIR / temp_name
        upload.save(temp_path)
        return temp_path, f"/uploads/{temp_name}", True

    sample_name = form.get("audio_sample")
    if sample_name:
        sample_name = secure_filename(sample_name)
        if sample_name not in AUDIO_SAMPLE_FILES:
            raise ValueError("ไฟล์เสียงตัวอย่างที่เลือกไม่ถูกต้อง")
        path = AUDIO_SAMPLES_DIR / sample_name
        if not path.exists():
            raise ValueError("ไม่พบไฟล์เสียงตัวอย่างบนเซิร์ฟเวอร์ กรุณารีสตาร์ทแอปเพื่อดาวน์โหลดใหม่")
        return path, f"/static/audio_samples/{sample_name}", False

    raise ValueError("กรุณาอัปโหลดไฟล์เสียง หรือเลือกจากตัวอย่าง")


def convert_to_wav(src_path: Path) -> Path:
    """
    แปลงไฟล์เสียงฟอร์แมตใดก็ได้ (mp3, m4a, ogg, flac, ...) ให้เป็น WAV
    mono, 16kHz, 16-bit PCM ด้วย ffmpeg เพื่อให้อ่านด้วย scipy.io.wavfile ได้แน่นอน
    (จำเป็นต้องติดตั้ง ffmpeg ไว้ในเครื่อง/เซิร์ฟเวอร์ — ดู README)
    """
    out_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_conv.wav"
    cmd = [
        FFMPEG_BIN, "-y", "-i", str(src_path),
        "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    except FileNotFoundError as exc:
        raise ValueError(
            "ไม่พบโปรแกรม ffmpeg บนเซิร์ฟเวอร์ กรุณาติดตั้ง ffmpeg ก่อนใช้งานฟีเจอร์ Audio Classifier "
            "(ดูวิธีติดตั้งใน README)"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        raise ValueError(
            "ไม่สามารถแปลงไฟล์เสียงได้ ไฟล์อาจเสียหายหรือฟอร์แมตไม่รองรับ "
            f"({stderr.strip()[-200:] or 'ไม่ทราบสาเหตุ'})"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("แปลงไฟล์เสียงใช้เวลานานเกินไป กรุณาลองไฟล์ที่สั้นกว่านี้") from exc

    return out_path


def classify_audio(path: Path, max_results: int = AUDIO_MAX_RESULTS) -> list:
    """
    จำแนกประเภทเสียงด้วย MediaPipe AudioClassifier (โมเดล YAMNet / AudioSet)
    โครงสร้างเดียวกับใน audio_classification.ipynb ต้นฉบับ ต่างกันตรงที่เราใช้
    result.timestamp_ms ที่ MediaPipe คืนมาให้โดยตรง แทนการ hardcode ช่วงเวลา
    เพื่อให้รองรับไฟล์เสียงความยาวเท่าไหร่ก็ได้
    """
    base_options = mp_python.BaseOptions(model_asset_path=str(AUDIO_MODEL_PATH))
    options = audio.AudioClassifierOptions(base_options=base_options, max_results=max_results)

    try:
        sample_rate, wav_data = wavfile.read(str(path))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"ไม่สามารถอ่านไฟล์เสียงที่แปลงแล้วได้: {exc}") from exc

    if wav_data.ndim > 1:
        wav_data = wav_data.mean(axis=1)

    if np.issubdtype(wav_data.dtype, np.integer):
        max_val = float(np.iinfo(wav_data.dtype).max)
        normalized = wav_data.astype(np.float64) / max_val
    else:
        normalized = wav_data.astype(np.float64)

    with audio.AudioClassifier.create_from_options(options) as classifier:
        audio_clip = containers.AudioData.create_from_array(
            normalized.astype(np.float32), sample_rate
        )
        classification_result_list = classifier.classify(audio_clip)

    if not classification_result_list:
        raise ValueError("ไม่สามารถจำแนกเสียงได้ กรุณาลองไฟล์เสียงอื่น (ไฟล์อาจสั้นเกินไป)")

    segments = []
    for idx, result in enumerate(classification_result_list):
        timestamp_ms = result.timestamp_ms if result.timestamp_ms is not None else idx * 975
        categories = []
        if result.classifications:
            for cat in result.classifications[0].categories:
                categories.append(
                    {
                        "name": cat.category_name or cat.display_name or "ไม่ทราบชื่อหมวดหมู่",
                        "score": float(cat.score) if cat.score is not None else 0.0,
                    }
                )
        segments.append({"timestamp_ms": int(timestamp_ms), "categories": categories})

    return segments


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


@app.route("/")
def index():
    ensure_assets()
    return render_template("index.html", samples=SAMPLE_IMAGES, active_page="image")


@app.route("/audio-classifier")
def audio_page():
    ensure_assets()
    return render_template("audio.html", samples=AUDIO_SAMPLE_FILES, active_page="audio")


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/api/compare", methods=["POST"])
def api_compare():
    try:
        ensure_assets()

        path1, url1, _ = resolve_image("image1", request.form, request.files)
        path2, url2, _ = resolve_image("image2", request.form, request.files)

        l2_normalize = request.form.get("l2_normalize", "true").lower() == "true"
        quantize = request.form.get("quantize", "true").lower() == "true"

        similarity = embed_and_compare(path1, path2, l2_normalize, quantize)

        return jsonify(
            {
                "ok": True,
                "similarity": similarity,
                "image1_url": url1,
                "image2_url": url2,
                "options": {"l2_normalize": l2_normalize, "quantize": quantize},
            }
        )

    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001 - แสดงข้อความ error ให้ผู้ใช้เห็นแบบอ่านง่าย
        return jsonify({"ok": False, "error": f"เกิดข้อผิดพลาดที่เซิร์ฟเวอร์: {exc}"}), 500


@app.route("/api/classify-audio", methods=["POST"])
def api_classify_audio():
    converted_path = None
    try:
        ensure_assets()

        src_path, display_url, _ = resolve_audio(request.form, request.files)
        converted_path = convert_to_wav(src_path)
        segments = classify_audio(converted_path, AUDIO_MAX_RESULTS)

        return jsonify(
            {
                "ok": True,
                "audio_url": display_url,
                "segments": segments,
            }
        )

    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"เกิดข้อผิดพลาดที่เซิร์ฟเวอร์: {exc}"}), 500
    finally:
        # ไฟล์ wav ที่แปลงแล้วเป็นไฟล์ชั่วคราวใช้ครั้งเดียว ลบทิ้งเสมอหลังประมวลผล
        if converted_path is not None and converted_path.exists():
            try:
                converted_path.unlink()
            except OSError:
                pass


@app.errorhandler(413)
def too_large(_exc):
    return jsonify({"ok": False, "error": "ไฟล์รูปภาพมีขนาดใหญ่เกินไป (จำกัดไม่เกิน 10MB ต่อคำขอ)"}), 413


if __name__ == "__main__":
    ensure_assets()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
