# MediaPipe Tasks — Web UI (Flask)

เว็บแอปที่รวมสอง demo ของ [MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/guide)
ไว้ในที่เดียว โดยใช้ Flask เป็นเว็บเซิร์ฟเวอร์ และสลับหน้าได้จากแท็บเมนูด้านบน:

1. **Image Embedder** — เปรียบเทียบความคล้ายกันของภาพ 2 ภาพ (MobileNet V3 Small)
2. **Audio Classifier** — จำแนกประเภทเสียงตามช่วงเวลา (YAMNet / AudioSet)

โค้ดพื้นฐานอ้างอิงจากตัวอย่างทางการของ Google:
- https://github.com/googlesamples/mediapipe/blob/main/examples/image_embedder/python/image_embedder.ipynb
- https://github.com/googlesamples/mediapipe/blob/main/examples/audio_classifier/python/audio_classification.ipynb

หน้าตาและการทำงานอ้างอิงแนวทางจาก MediaPipe Web Samples Demo:
https://google-ai-edge.github.io/mediapipe-samples-web/

## ฟีเจอร์

### 🖼️ Image Embedder
- อัปโหลดรูปภาพของตัวเอง 2 รูป (คลิกหรือลากไฟล์มาวาง) หรือเลือกจากรูปตัวอย่าง
  ที่มีให้ (burger.jpg / burger_crop.jpg — เหมือนใน notebook ต้นฉบับ)
- กด "เปรียบเทียบ" เพื่อคำนวณ **Cosine Similarity** ระหว่างภาพทั้งสอง
  (ค่าอยู่ระหว่าง -1 ถึง 1) พร้อมแสดงเป็นแถบวัดผล
- ตัวเลือกขั้นสูง: เปิด/ปิด `l2_normalize` และ `quantize`

### 🔊 Audio Classifier
- สลับได้ 2 โหมด (เหมือน MediaPipe Web Samples Demo):
  - **🎙️ Microphone** — กด "เริ่มอัดเสียง" พูด/เปิดเสียงใส่ไมค์ แล้วกด "หยุดอัดเสียง"
    ระบบจะส่งเสียงที่อัดได้ทั้งหมดไปจำแนกให้อัตโนมัติทันที (ต้องเปิดหน้าเว็บผ่าน
    `http://localhost:5000` หรือ `http://127.0.0.1:5000` เท่านั้น — เบราว์เซอร์
    บล็อกการเข้าถึงไมโครโฟนถ้าเข้าผ่าน IP อื่นที่ไม่ใช่ HTTPS/localhost)
  - **📁 Audio File** — อัปโหลดไฟล์เสียงของตัวเอง (รองรับหลายฟอร์แมต เช่น mp3, wav,
    m4a, ogg, flac ฯลฯ — แปลงเป็น WAV mono 16kHz ให้อัตโนมัติด้วย ffmpeg ก่อน
    ประมวลผล) หรือเลือกจากไฟล์เสียงตัวอย่าง (speech_16000_hz_mono.wav — เหมือนใน
    notebook ต้นฉบับ) แล้วกด "จำแนกประเภทเสียง"
- จำแนกเสียงด้วยโมเดล **YAMNet (AudioSet)** ผลลัพธ์จะถูกแบ่งเป็นช่วงเวลา (segment)
  พร้อมคะแนนความมั่นใจของแต่ละช่วง (ใช้ `timestamp_ms` ที่ MediaPipe คืนมาโดยตรง
  จึงรองรับไฟล์เสียงความยาวเท่าไหร่ก็ได้ ไม่ผูกกับความยาวไฟล์ตัวอย่างใน notebook)
- ตัวเลือกขั้นสูง (ปรับได้จากหน้าเว็บ ตรงกับ SETTINGS ใน MediaPipe Web Samples Demo):
  - **Max Results** (1–10, ค่าเริ่มต้น 4) — จำนวนหมวดหมู่สูงสุดต่อช่วงเวลา
  - **Score Threshold** (0–1, ค่าเริ่มต้น 0) — ตัดหมวดหมู่ที่คะแนนต่ำกว่าค่านี้ทิ้ง
- แสดงสถานะ (Ready / Processing / Error) และ **Inference Time** (เวลาที่โมเดลใช้
  จำแนกจริง หน่วย ms) เหมือนแถบสถานะใน MediaPipe Web Samples Demo

ทั้งสองฟีเจอร์ดาวน์โหลดโมเดลและไฟล์ตัวอย่างให้อัตโนมัติในการรันครั้งแรก
ไม่ต้องเตรียมไฟล์เอง

## โครงสร้างโปรเจกต์

```
mediapipe-image-embedder-webui/
├── app.py                     # Flask backend + MediaPipe inference (ทั้งสองฟีเจอร์)
├── requirements.txt
├── templates/
│   ├── base.html               # เลย์เอาต์กลาง + แท็บเมนู
│   ├── index.html               # หน้า Image Embedder
│   └── audio.html               # หน้า Audio Classifier
├── static/
│   ├── css/style.css
│   ├── js/main.js               # ตรรกะหน้า Image Embedder
│   ├── js/audio.js              # ตรรกะหน้า Audio Classifier
│   ├── samples/                  # รูปตัวอย่าง (ดาวน์โหลดอัตโนมัติ)
│   └── audio_samples/            # ไฟล์เสียงตัวอย่าง (ดาวน์โหลดอัตโนมัติ)
├── models/                      # embedder.tflite, classifier.tflite (ดาวน์โหลดอัตโนมัติ)
└── uploads/                     # ไฟล์ที่ผู้ใช้อัปโหลด (ชั่วคราว)
```

## สิ่งที่ต้องติดตั้งก่อน

- Python 3.9+
- **ffmpeg** (จำเป็นสำหรับฟีเจอร์ Audio Classifier เพื่อแปลงไฟล์เสียงหลากหลาย
  ฟอร์แมตให้เป็น WAV ก่อนส่งเข้า MediaPipe) — ติดตั้งผ่านตัวจัดการแพ็กเกจของระบบ
  ไม่ใช่ pip:
  ```bash
  # macOS (Homebrew)
  brew install ffmpeg

  # Ubuntu / Debian
  sudo apt-get update && sudo apt-get install -y ffmpeg

  # Windows (Chocolatey)
  choco install ffmpeg
  ```
  ตรวจสอบว่าติดตั้งสำเร็จด้วยคำสั่ง `ffmpeg -version`
- การเชื่อมต่ออินเทอร์เน็ต (สำหรับดาวน์โหลดโมเดล/ไฟล์ตัวอย่างในการรันครั้งแรกเท่านั้น)

## วิธีติดตั้งและรัน

```bash
# 1) สร้างและเปิดใช้งาน virtual environment (แนะนำ)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2) ติดตั้ง dependencies
pip install -r requirements.txt

# 3) รันแอป
python app.py
```

จากนั้นเปิดเบราว์เซอร์ไปที่ **http://localhost:5000** แล้วสลับหน้าได้จากแท็บเมนู
ด้านบน ("Image Embedder" / "Audio Classifier")

> ⚠️ **สำคัญสำหรับฟีเจอร์อัดเสียงจากไมโครโฟน:** ต้องเปิดผ่าน `http://localhost:5000`
> หรือ `http://127.0.0.1:5000` เท่านั้น ถ้าเปิดผ่าน IP อื่น (เช่น `http://172.x.x.x:5000`
> ที่เห็นในแถบที่อยู่เวลาเข้าจากเครื่องอื่นในวง LAN) เบราว์เซอร์จะบล็อกการขอสิทธิ์
> ไมโครโฟนโดยอัตโนมัติ เพราะเป็นข้อกำหนดด้านความปลอดภัยของเบราว์เซอร์เอง (ต้องเป็น
> HTTPS หรือ localhost) — ฟีเจอร์อัปโหลดไฟล์เสียง/เลือกตัวอย่าง และ Image Embedder
> ไม่ติดข้อจำกัดนี้ ใช้ผ่าน IP ไหนก็ได้ตามปกติ

การรันครั้งแรกจะใช้เวลาสักครู่เพื่อดาวน์โหลด:
- `models/embedder.tflite` (MobileNet V3 Small)
- `models/classifier.tflite` (YAMNet)
- `static/samples/burger.jpg`, `burger_crop.jpg`
- `static/audio_samples/speech_16000_hz_mono.wav`

รันครั้งถัดไปจะใช้ไฟล์ที่ดาวน์โหลดไว้แล้วทันที ไม่ต้องดาวน์โหลดซ้ำ

## การทำงานของระบบ (โดยสรุป)

### Image Embedder
1. ผู้ใช้เลือกภาพ 2 ภาพ แล้วกด "เปรียบเทียบ" → หน้าเว็บส่งข้อมูลไปที่ `POST /api/compare`
2. Backend โหลดโมเดลผ่าน `vision.ImageEmbedder`, แปลงภาพทั้งสองเป็น embedding,
   คำนวณ `cosine_similarity(...)` แล้วส่งค่ากลับเป็น JSON
3. หน้าเว็บแสดงผลค่า similarity พร้อมแถบวัดผล (gauge)

### Audio Classifier
1. ผู้ใช้เลือกไฟล์เสียง (อัปโหลด/ตัวอย่าง) หรืออัดเสียงจากไมค์ (`getUserMedia` +
   `MediaRecorder` ฝั่งเบราว์เซอร์ ได้ไฟล์ webm/ogg) → หน้าเว็บส่งไฟล์ไปที่
   `POST /api/classify-audio` พร้อมค่า `max_results` / `score_threshold` ที่ตั้งไว้
   (โหมดไมค์จะส่งให้อัตโนมัติทันทีที่กด "หยุดอัดเสียง" ไม่ต้องกดปุ่มอื่นเพิ่ม)
2. Backend แปลงไฟล์เป็น WAV mono 16kHz ด้วย `ffmpeg`, อ่านด้วย `scipy.io.wavfile`,
   สร้าง `AudioData` แล้วส่งเข้า `audio.AudioClassifier` (โมเดล YAMNet) พร้อมจับเวลา
   เฉพาะขั้นตอน `classifier.classify(...)` เพื่อรายงานเป็น `inference_time_ms`
3. ผลลัพธ์แต่ละช่วงเวลา (segment) จะมี `timestamp_ms` และหมวดหมู่ (ตามจำนวน
   `max_results` และกรองด้วย `score_threshold` ที่ตั้งไว้) พร้อมคะแนน ส่งกลับเป็น
   JSON แล้วแสดงเป็นการ์ดต่อช่วงเวลาในหน้าเว็บ พร้อมสถานะและ inference time
4. ไฟล์ WAV ที่แปลงแล้วเป็นไฟล์ชั่วคราว จะถูกลบทิ้งทันทีหลังประมวลผลเสร็จ

## หมายเหตุ

- แอปนี้เหมาะสำหรับใช้งาน/เดโมในเครื่อง (development) เท่านั้น หากต้องการ
  deploy ใช้งานจริง ควรรันผ่าน WSGI server เช่น gunicorn, ปิด debug mode,
  และตรวจสอบให้แน่ใจว่า ffmpeg ติดตั้งอยู่บนเซิร์ฟเวอร์ด้วย
- จำกัดขนาดไฟล์อัปโหลดไว้ที่ 10MB ต่อคำขอ (ปรับได้ที่ `MAX_CONTENT_LENGTH` ใน `app.py`)
- โฟลเดอร์ `uploads/` จะสะสมไฟล์ที่ผู้ใช้อัปโหลดไว้เรื่อย ๆ (เพื่อให้เล่น/แสดง
  ตัวอย่างย้อนหลังในหน้าเว็บได้) สามารถลบไฟล์เก่าทิ้งเป็นระยะได้ตามต้องการ
- เกณฑ์ข้อความ "คล้ายกันมาก / ค่อนข้างคล้ายกัน / ไม่ค่อยคล้ายกัน" ในหน้า Image
  Embedder เป็นเพียงการจัดกลุ่มแบบคร่าว ๆ เพื่อให้อ่านง่ายเท่านั้น ไม่ใช่ค่า
  มาตรฐานจาก MediaPipe
