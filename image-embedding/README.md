# MediaPipe Image Embedder — Web UI (Flask)

เว็บแอปสำหรับเปรียบเทียบความคล้ายกันของภาพ 2 ภาพ โดยใช้ [MediaPipe Tasks:
Image Embedder](https://ai.google.dev/edge/mediapipe/solutions/vision/image_embedder)
(โมเดล MobileNet V3 Small) เป็น backend และ Flask เป็นเว็บเซิร์ฟเวอร์

โค้ดพื้นฐานอ้างอิงจากตัวอย่างทางการของ Google:
https://github.com/googlesamples/mediapipe/blob/main/examples/image_embedder/python/image_embedder.ipynb

หน้าตาและการทำงานอ้างอิงแนวทางจาก MediaPipe Web Samples Demo:
https://google-ai-edge.github.io/mediapipe-samples-web/#/vision/image_embedder

## ฟีเจอร์

- อัปโหลดรูปภาพของตัวเอง 2 รูป (คลิกหรือลากไฟล์มาวาง) หรือเลือกจากรูปตัวอย่าง
  ที่มีให้ (burger.jpg / burger_crop.jpg — เหมือนใน notebook ต้นฉบับ)
- กด "เปรียบเทียบ" เพื่อคำนวณ **Cosine Similarity** ระหว่างภาพทั้งสอง
  (ค่าอยู่ระหว่าง -1 ถึง 1 ยิ่งใกล้ 1 ยิ่งคล้ายกันมาก) พร้อมแสดงเป็นแถบวัดผล
- ตัวเลือกขั้นสูง: เปิด/ปิด `l2_normalize` และ `quantize` (ตรงกับพารามิเตอร์ใน
  notebook ต้นฉบับ)
- ดาวน์โหลดโมเดลและรูปตัวอย่างให้อัตโนมัติในการรันครั้งแรก ไม่ต้องเตรียมไฟล์เอง

## โครงสร้างโปรเจกต์

```
mediapipe-image-embedder-webui/
├── app.py                  # Flask backend + MediaPipe inference
├── requirements.txt
├── templates/
│   └── index.html          # หน้าเว็บหลัก
├── static/
│   ├── css/style.css
│   ├── js/main.js          # อัปโหลด/เลือกตัวอย่าง/เรียก API/แสดงผล
│   └── samples/             # รูปตัวอย่าง (ดาวน์โหลดอัตโนมัติ)
├── models/                  # embedder.tflite (ดาวน์โหลดอัตโนมัติ)
└── uploads/                 # ไฟล์ที่ผู้ใช้อัปโหลด (ชั่วคราว)
```

## วิธีติดตั้งและรัน

ต้องมี Python 3.9+ และการเชื่อมต่ออินเทอร์เน็ต (สำหรับดาวน์โหลดโมเดล/รูปตัวอย่าง
ในการรันครั้งแรกเท่านั้น)

```bash
# 1) สร้างและเปิดใช้งาน virtual environment (แนะนำ)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2) ติดตั้ง dependencies
pip install -r requirements.txt

# 3) รันแอป
python app.py
```

จากนั้นเปิดเบราว์เซอร์ไปที่ **http://localhost:5000**

การรันครั้งแรกจะใช้เวลาสักครู่เพื่อดาวน์โหลด:
- ไฟล์โมเดล `embedder.tflite` (MobileNet V3 Small, ~ไม่กี่ MB) เก็บไว้ที่ `models/`
- รูปตัวอย่าง `burger.jpg`, `burger_crop.jpg` เก็บไว้ที่ `static/samples/`

รันครั้งถัดไปจะใช้ไฟล์ที่ดาวน์โหลดไว้แล้วทันที ไม่ต้องดาวน์โหลดซ้ำ

## การทำงานของระบบ (โดยสรุป)

1. ผู้ใช้เลือกภาพ 2 ภาพ (อัปโหลดเอง หรือเลือกจากตัวอย่าง) แล้วกด "เปรียบเทียบ"
2. หน้าเว็บส่งข้อมูลรูปภาพไปที่ endpoint `POST /api/compare`
3. ฝั่ง backend (`app.py`) จะ:
   - โหลดโมเดลผ่าน `mediapipe.tasks.python.vision.ImageEmbedder`
   - แปลงภาพทั้งสองเป็น embedding ด้วย `embedder.embed(...)`
   - คำนวณ `ImageEmbedder.cosine_similarity(...)` ระหว่าง embedding ทั้งสอง
   - ส่งค่า similarity กลับเป็น JSON
4. หน้าเว็บแสดงผลค่า similarity พร้อมแถบวัดผล (gauge)

## หมายเหตุ

- แอปนี้เหมาะสำหรับใช้งาน/เดโมในเครื่อง (development) เท่านั้น หากต้องการ
  deploy ใช้งานจริง ควรรันผ่าน WSGI server เช่น gunicorn และปิด debug mode
- จำกัดขนาดไฟล์อัปโหลดไว้ที่ 10MB ต่อคำขอ (ปรับได้ที่ `MAX_CONTENT_LENGTH` ใน `app.py`)
- เกณฑ์ข้อความ "คล้ายกันมาก / ค่อนข้างคล้ายกัน / ไม่ค่อยคล้ายกัน" เป็นเพียงการจัดกลุ่ม
  แบบคร่าว ๆ เพื่อให้อ่านง่ายในหน้าเว็บเท่านั้น ไม่ใช่ค่ามาตรฐานจาก MediaPipe
