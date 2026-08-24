# กฎการ Add & Commit สำหรับโปรเจกต์ ImageMedipipe

เอกสารนี้กำหนดแนวทางการใช้ `git add` และ `git commit` สำหรับ repo นี้ เพื่อให้ประวัติ commit อ่านง่าย ตรวจสอบย้อนหลังได้ และไม่หลุดไฟล์ที่ไม่ควร push ขึ้น remote

## 1. ก่อน commit ทุกครั้ง

1. รัน `git status` ก่อนเสมอ เพื่อดูว่ามีไฟล์อะไรเปลี่ยนแปลง/ยังไม่ได้ track บ้าง
2. รัน `git diff` (หรือ `git diff --staged` หลัง add แล้ว) เพื่อตรวจสอบเนื้อหาที่จะ commit จริง ๆ ว่าตรงกับที่ตั้งใจ
3. ตรวจสอบว่าไม่มีไฟล์ที่ไม่ควร commit หลุดเข้ามา เช่น:
   - ไฟล์ model `*.tflite` ใน `models/` (ให้ใช้ `models/download_models.py` แทนการ commit ตัวไฟล์)
   - ไฟล์ environment เช่น `.venv/`, `.env`
   - ไฟล์ข้อมูลทดสอบขนาดใหญ่ใน `examples/sample_data/`
   - ไฟล์ที่มี secret/credential ใด ๆ (API key, token, password)

## 2. การ `git add`

- **ห้าม** ใช้ `git add .` หรือ `git add -A` พร่ำเพรื่อ ให้ระบุชื่อไฟล์ที่ต้องการ add ทีละรายการหรือทีละกลุ่มที่เกี่ยวข้องกัน เช่น:
  ```bash
  git add src/image_embedding/embedder.py
  git add README.md COMMIT_RULES.md
  ```
- ถ้าจำเป็นต้อง add ทั้งหมดจริง ๆ ให้รัน `git status` ตรวจดูรายการก่อนเสมอ แล้วค่อยยืนยัน add
- 1 commit ควรมีการเปลี่ยนแปลงเรื่องเดียว (one logical change) ไม่รวมหลายฟีเจอร์/หลาย bug fix ไว้ใน commit เดียว

## 3. รูปแบบข้อความ commit (Conventional Commits)

ใช้รูปแบบ:

```
<type>: <คำอธิบายสั้น ๆ ภาษาไทยหรืออังกฤษก็ได้>

<รายละเอียดเพิ่มเติม (ถ้ามี)>
```

### ประเภท (`type`) ที่ใช้ได้

| type       | ใช้เมื่อ                                                        |
|------------|-------------------------------------------------------------------|
| `feat`     | เพิ่มฟีเจอร์ใหม่ (เช่น เพิ่ม module audio classification)         |
| `fix`      | แก้บั๊ก                                                            |
| `docs`     | แก้ไข/เพิ่มเอกสาร เช่น README, COMMIT_RULES.md                    |
| `refactor` | ปรับโครงสร้างโค้ดโดยไม่เปลี่ยนพฤติกรรม                             |
| `test`     | เพิ่ม/แก้ไข test                                                   |
| `chore`    | งานจุกจิก เช่น แก้ `.gitignore`, จัดการ dependency, ปรับ config    |
| `perf`     | ปรับปรุงประสิทธิภาพ                                                |

### ตัวอย่าง commit message ที่ดี

```bash
git commit -m "feat: เพิ่ม ImageEmbedder wrapper สำหรับเปรียบเทียบความคล้ายรูปภาพ"
git commit -m "fix: แก้ path model ผิดใน audio classifier"
git commit -m "docs: เพิ่มตัวอย่างการใช้งานใน README"
git commit -m "chore: เพิ่ม models/*.tflite ใน .gitignore"
```

### ข้อความ commit ที่ไม่ควรใช้

```bash
git commit -m "update"
git commit -m "fix bug"
git commit -m "wip"
git commit -m "แก้ไข"
```
(สั้นเกินไป ไม่บอกว่าเปลี่ยนอะไร ทำไมต้องเปลี่ยน)

## 4. ข้อห้าม

- ห้าม `git commit --amend` กับ commit ที่ push ขึ้น remote ไปแล้ว (ยกเว้นได้รับอนุญาตชัดเจน)
- ห้าม `git push --force` ไปที่ branch `main` โดยไม่ได้รับอนุญาต
- ห้ามใช้ `--no-verify` เพื่อข้าม pre-commit hook เว้นแต่จำเป็นจริง ๆ และแจ้งเหตุผล
- ห้าม commit ไฟล์ที่มี secret/credential แม้จะลบออกใน commit ถัดไปก็ยังอยู่ใน git history

## 5. Checklist ก่อนกด commit

- [ ] รัน `git status` แล้วเข้าใจว่าไฟล์ไหนจะถูก commit
- [ ] รัน `git diff --staged` แล้วตรวจว่าเนื้อหาถูกต้อง ไม่มีของหลุด
- [ ] ไม่มีไฟล์ model (`.tflite`), `.venv`, secret หลุดเข้ามาใน staging
- [ ] ข้อความ commit ใช้รูปแบบ `<type>: <คำอธิบาย>` และสื่อสารว่า "ทำไม" ไม่ใช่แค่ "ทำอะไร"
- [ ] 1 commit = 1 การเปลี่ยนแปลงเชิงตรรกะ (logical change)
