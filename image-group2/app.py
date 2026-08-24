"""
MediaPipe Image Embedder - Flask Web UI
=========================================
เว็บแอปสำหรับเปรียบเทียบความคล้ายกันของภาพ 2 ภาพ ด้วย MediaPipe Tasks
"Image Embedder" (โมเดล MobileNet V3 Small) โดยอิงจากตัวอย่างโค้ดทางการของ
Google MediaPipe:
https://github.com/googlesamples/mediapipe/blob/main/examples/image_embedder/python/image_embedder.ipynb

วิธีรัน:
    pip install -r requirements.txt
    python app.py

ครั้งแรกที่รัน แอปจะดาวน์โหลดไฟล์โมเดล (embedder.tflite) และรูปภาพตัวอย่าง
มาเก็บไว้ในเครื่องโดยอัตโนมัติ (ต้องมีอินเทอร์เน็ตในตอนนั้น) หลังจากนั้นจะใช้
ไฟล์ที่ดาวน์โหลดไว้แล้วโดยไม่ต้องดาวน์โหลดซ้ำ
"""

import math
import threading
import urllib.request
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

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
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        if not MODEL_PATH.exists():
            print(f"[setup] กำลังดาวน์โหลดโมเดล embedder -> {MODEL_PATH}")
            tmp_path = MODEL_PATH.with_suffix(".tmp")
            urllib.request.urlretrieve(MODEL_URL, tmp_path)
            tmp_path.rename(MODEL_PATH)
            print("[setup] ดาวน์โหลดโมเดลเสร็จแล้ว")
        else:
            print(f"[setup] พบไฟล์โมเดลอยู่แล้วที่ {MODEL_PATH}")

        for name in SAMPLE_IMAGES:
            dest = SAMPLES_DIR / name
            if not dest.exists():
                print(f"[setup] กำลังดาวน์โหลดรูปตัวอย่าง -> {dest}")
                tmp_path = dest.with_suffix(dest.suffix + ".tmp")
                urllib.request.urlretrieve(SAMPLE_ASSETS_BASE_URL + name, tmp_path)
                tmp_path.rename(dest)
        print("[setup] รูปตัวอย่างพร้อมใช้งาน")

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
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


@app.route("/")
def index():
    ensure_assets()
    return render_template("index.html", samples=SAMPLE_IMAGES)


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


@app.errorhandler(413)
def too_large(_exc):
    return jsonify({"ok": False, "error": "ไฟล์รูปภาพมีขนาดใหญ่เกินไป (จำกัดไม่เกิน 10MB ต่อคำขอ)"}), 413


if __name__ == "__main__":
    ensure_assets()
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
