(() => {
  "use strict";

  // ---- state สำหรับโหมด "Audio File" (อัปโหลด / เลือกตัวอย่าง) ----
  const fileState = { type: null, file: null, sampleName: null };

  let currentMode = "mic"; // 'mic' | 'file'

  // ---- mic recording state ----
  let mediaRecorder = null;
  let mediaStream = null;
  let recordedChunks = [];
  let isRecording = false;
  let isBusy = false; // กำลังส่งไปจำแนกอยู่หรือไม่ (กันกดซ้อน)

  // ---- DOM refs ----
  const form = document.getElementById("audio-form");
  const errorBox = document.getElementById("audio-error-box");
  const loadingOverlay = document.getElementById("audio-loading-overlay");
  const emptyHint = document.getElementById("audio-empty-hint");
  const segmentsList = document.getElementById("segments-list");
  const statusLabel = document.getElementById("status-label");
  const inferenceTimeEl = document.getElementById("inference-time");

  const modeButtons = document.querySelectorAll(".mode-toggle__btn");
  const micPanel = document.getElementById("mic-panel");
  const filePanel = document.getElementById("file-panel");

  const recordBtn = document.getElementById("record-btn");
  const recordBtnLabel = document.getElementById("record-btn-label");
  const micStatus = document.getElementById("mic-status");
  const micPlayer = document.getElementById("mic_player");
  const micUnsupported = document.getElementById("mic-unsupported");

  const dropzone = document.querySelector(".audio-dropzone");
  const fileInput = document.getElementById("audio_file");
  const sampleHidden = document.getElementById("audio_sample");
  const filePlayer = document.getElementById("audio_player");
  const filePreview = document.getElementById("audio_preview");
  const classifyBtn = document.getElementById("classify-btn");
  const resetBtn = document.getElementById("audio-reset-btn");

  const maxResultsInput = document.getElementById("max_results");
  const maxResultsValue = document.getElementById("max_results_value");
  const scoreThresholdInput = document.getElementById("score_threshold");
  const scoreThresholdValue = document.getElementById("score_threshold_value");

  // ---------------------------------------------------------------------
  // Helpers ทั่วไป
  // ---------------------------------------------------------------------

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  function clearError() {
    errorBox.hidden = true;
    errorBox.textContent = "";
  }

  function setStatus(text) {
    statusLabel.textContent = text;
  }

  function setInferenceTime(ms) {
    if (typeof ms === "number") {
      inferenceTimeEl.textContent = `Inference Time: ${ms.toFixed(1)} ms`;
    } else {
      inferenceTimeEl.textContent = "Inference Time: - ms";
    }
  }

  function setLoading(loading) {
    isBusy = loading;
    loadingOverlay.hidden = !loading;
    recordBtn.disabled = loading;
    classifyBtn.disabled = loading || fileState.type === null;
  }

  function clearResults() {
    segmentsList.innerHTML = "";
    emptyHint.hidden = false;
    setInferenceTime(null);
  }

  // ---------------------------------------------------------------------
  // สลับโหมด Microphone / Audio File
  // ---------------------------------------------------------------------

  function switchMode(mode) {
    if (mode === currentMode) return;
    if (isRecording) {
      stopRecording();
    }
    currentMode = mode;
    modeButtons.forEach((btn) => {
      const active = btn.dataset.mode === mode;
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    });
    micPanel.hidden = mode !== "mic";
    filePanel.hidden = mode !== "file";
    clearError();
  }

  modeButtons.forEach((btn) => {
    btn.addEventListener("click", () => switchMode(btn.dataset.mode));
  });

  // ---------------------------------------------------------------------
  // ตัวเลือกขั้นสูง: Max Results / Score Threshold
  // ---------------------------------------------------------------------

  maxResultsInput.addEventListener("input", () => {
    maxResultsValue.textContent = maxResultsInput.value;
  });

  scoreThresholdInput.addEventListener("input", () => {
    scoreThresholdValue.textContent = parseFloat(scoreThresholdInput.value).toFixed(2);
  });

  // ---------------------------------------------------------------------
  // โหมด Microphone
  // ---------------------------------------------------------------------

  function micSupported() {
    return (
      window.isSecureContext &&
      navigator.mediaDevices &&
      typeof navigator.mediaDevices.getUserMedia === "function"
    );
  }

  if (!micSupported()) {
    micUnsupported.hidden = false;
    recordBtn.disabled = true;
    micStatus.textContent = "ไม่สามารถใช้ไมโครโฟนได้บนหน้านี้";
  }

  function updateRecordButtonUI() {
    recordBtn.classList.toggle("is-recording", isRecording);
    recordBtnLabel.textContent = isRecording ? "หยุดอัดเสียง" : "เริ่มอัดเสียง";
  }

  async function startRecording() {
    if (!micSupported() || isBusy) return;

    clearError();
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      showError("ไม่สามารถเข้าถึงไมโครโฟนได้: " + err.message);
      return;
    }

    recordedChunks = [];
    let chosenMimeType = "";
    const candidates = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/ogg",
    ];
    for (const candidate of candidates) {
      if (window.MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(candidate)) {
        chosenMimeType = candidate;
        break;
      }
    }

    try {
      mediaRecorder = chosenMimeType
        ? new MediaRecorder(mediaStream, { mimeType: chosenMimeType })
        : new MediaRecorder(mediaStream);
    } catch (err) {
      showError("ไม่สามารถเริ่มอัดเสียงได้: " + err.message);
      mediaStream.getTracks().forEach((t) => t.stop());
      return;
    }

    mediaRecorder.addEventListener("dataavailable", (e) => {
      if (e.data && e.data.size > 0) {
        recordedChunks.push(e.data);
      }
    });
    mediaRecorder.addEventListener("stop", onRecordingStopped);

    mediaRecorder.start();
    isRecording = true;
    updateRecordButtonUI();
    micStatus.textContent = "🔴 กำลังอัดเสียง... พูดหรือเปิดเสียงใกล้ไมโครโฟน แล้วกด \"หยุดอัดเสียง\"";
    clearResults();
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
    isRecording = false;
    updateRecordButtonUI();
  }

  async function onRecordingStopped() {
    const mimeType = (mediaRecorder && mediaRecorder.mimeType) || "audio/webm";
    const blob = new Blob(recordedChunks, { type: mimeType });

    if (blob.size === 0) {
      micStatus.textContent = "พร้อมอัดเสียง — กดปุ่มด้านล่างเพื่อเริ่ม";
      showError("ไม่ได้บันทึกเสียงไว้เลย กรุณาลองอัดใหม่อีกครั้ง");
      return;
    }

    const objectUrl = URL.createObjectURL(blob);
    micPlayer.src = objectUrl;
    micPlayer.hidden = false;

    micStatus.textContent = "อัดเสียงเสร็จแล้ว กำลังส่งไปจำแนกประเภทเสียง...";

    const ext = mimeType.includes("ogg") ? "ogg" : "webm";
    const file = new File([blob], `recording.${ext}`, { type: mimeType });

    await classifyFile(file);

    if (!isBusy) {
      micStatus.textContent = "พร้อมอัดเสียง — กดปุ่มด้านล่างเพื่อเริ่มใหม่อีกครั้ง";
    }
  }

  recordBtn.addEventListener("click", () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  });

  // ---------------------------------------------------------------------
  // โหมด Audio File (อัปโหลด / เลือกตัวอย่าง)
  // ---------------------------------------------------------------------

  function clearSampleSelection() {
    document
      .querySelectorAll("#audio_samples .sample-thumb--audio")
      .forEach((btn) => btn.classList.remove("is-selected"));
  }

  function showFilePlayer(src) {
    filePlayer.src = src;
    filePlayer.hidden = false;
    filePreview.hidden = true;
  }

  function hideFilePlayer() {
    filePlayer.removeAttribute("src");
    filePlayer.hidden = true;
    filePreview.hidden = false;
  }

  function updateClassifyButton() {
    classifyBtn.disabled = isBusy || fileState.type === null;
  }

  function selectUploadedFile(file) {
    const looksLikeAudio =
      file.type.startsWith("audio/") ||
      /\.(wav|mp3|m4a|aac|ogg|oga|flac|wma|opus|webm|mp4|3gp|aiff|aif)$/i.test(file.name);
    if (!looksLikeAudio) {
      showError("กรุณาเลือกไฟล์เสียงเท่านั้น");
      return;
    }
    fileState.type = "file";
    fileState.file = file;
    fileState.sampleName = null;
    sampleHidden.value = "";
    clearSampleSelection();

    showFilePlayer(URL.createObjectURL(file));

    clearError();
    clearResults();
    updateClassifyButton();
  }

  function selectSample(sampleName) {
    fileState.type = "sample";
    fileState.file = null;
    fileState.sampleName = sampleName;
    sampleHidden.value = sampleName;
    fileInput.value = "";

    showFilePlayer(`/static/audio_samples/${encodeURIComponent(sampleName)}`);

    document.querySelectorAll("#audio_samples .sample-thumb--audio").forEach((btn) => {
      btn.classList.toggle("is-selected", btn.dataset.sample === sampleName);
    });

    clearError();
    clearResults();
    updateClassifyButton();
  }

  function resetFilePanel() {
    fileState.type = null;
    fileState.file = null;
    fileState.sampleName = null;
    fileInput.value = "";
    sampleHidden.value = "";
    hideFilePlayer();
    clearSampleSelection();
    clearError();
    clearResults();
    updateClassifyButton();
  }

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) {
      selectUploadedFile(fileInput.files[0]);
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("is-dragover");
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("is-dragover");
  });
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("is-dragover");
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) selectUploadedFile(file);
  });

  document.querySelectorAll("#audio_samples .sample-thumb--audio").forEach((btn) => {
    btn.addEventListener("click", () => selectSample(btn.dataset.sample));
  });

  resetBtn.addEventListener("click", resetFilePanel);

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (currentMode !== "file" || isBusy) return;
    clearError();

    if (fileState.type === null) {
      showError("กรุณาอัปโหลดไฟล์เสียง หรือเลือกจากตัวอย่างก่อน");
      return;
    }

    if (fileState.type === "file") {
      await classifyFile(fileState.file);
    } else {
      await classifySample(fileState.sampleName);
    }
  });

  // ---------------------------------------------------------------------
  // เรียก backend เพื่อจำแนกเสียง (ใช้ร่วมกันทั้งโหมด mic และ file)
  // ---------------------------------------------------------------------

  async function classifyFile(file) {
    const formData = new FormData();
    formData.append("audio_file", file);
    await runClassify(formData);
  }

  async function classifySample(sampleName) {
    const formData = new FormData();
    formData.append("audio_sample", sampleName);
    await runClassify(formData);
  }

  async function runClassify(formData) {
    formData.append("max_results", maxResultsInput.value);
    formData.append("score_threshold", scoreThresholdInput.value);

    clearError();
    setLoading(true);
    setStatus("Processing...");

    try {
      const resp = await fetch("/api/classify-audio", {
        method: "POST",
        body: formData,
      });
      const data = await resp.json();

      if (!resp.ok || !data.ok) {
        showError(data.error || "เกิดข้อผิดพลาดที่ไม่ทราบสาเหตุ");
        setStatus("Error");
        setInferenceTime(null);
        return;
      }

      renderSegments(data.segments);
      setInferenceTime(data.inference_time_ms);
      setStatus("Ready");
    } catch (err) {
      showError("ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์ได้: " + err.message);
      setStatus("Error");
      setInferenceTime(null);
    } finally {
      setLoading(false);
      updateClassifyButton();
    }
  }

  // ---------------------------------------------------------------------
  // แสดงผลลัพธ์
  // ---------------------------------------------------------------------

  function formatTimestamp(ms) {
    return `${(ms / 1000).toFixed(2)} วินาที`;
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  function renderSegments(segments) {
    segmentsList.innerHTML = "";

    if (!segments || segments.length === 0) {
      emptyHint.hidden = false;
      emptyHint.textContent = "ไม่พบผลลัพธ์การจำแนกเสียง";
      return;
    }

    emptyHint.hidden = true;

    segments.forEach((segment, idx) => {
      const card = document.createElement("div");
      card.className = "segment-card";

      const header = document.createElement("div");
      header.className = "segment-card__header";
      header.innerHTML = `<span class="segment-card__index">ช่วงที่ ${idx + 1}</span><span class="segment-card__time">เริ่มที่ ${formatTimestamp(segment.timestamp_ms)}</span>`;
      card.appendChild(header);

      const catList = document.createElement("div");
      catList.className = "category-list";

      if (segment.categories.length === 0) {
        const emptyRow = document.createElement("p");
        emptyRow.className = "hint";
        emptyRow.textContent = "ไม่มีหมวดหมู่ผ่านเกณฑ์ Score Threshold ที่ตั้งไว้";
        catList.appendChild(emptyRow);
      }

      const maxScore = segment.categories.length > 0 ? segment.categories[0].score : 1;

      segment.categories.forEach((cat) => {
        const row = document.createElement("div");
        row.className = "category-row";

        const percent = Math.max(0, Math.min(100, cat.score * 100));
        const relativeWidth = maxScore > 0 ? Math.max(4, (cat.score / maxScore) * 100) : 4;

        row.innerHTML = `
          <div class="category-row__label">
            <span>${escapeHtml(cat.name)}</span>
            <span class="category-row__score">${percent.toFixed(1)}%</span>
          </div>
          <div class="category-row__track">
            <div class="category-row__fill" style="width: ${relativeWidth}%"></div>
          </div>
        `;
        catList.appendChild(row);
      });

      card.appendChild(catList);
      segmentsList.appendChild(card);
    });
  }

  // ---------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------

  updateClassifyButton();
  setInferenceTime(null);
})();
