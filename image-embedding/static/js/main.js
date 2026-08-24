(() => {
  "use strict";

  const slots = ["image1", "image2"];

  // เก็บสถานะการเลือกรูปของแต่ละ slot: { type: 'file' | 'sample', file, sampleName }
  const state = {
    image1: { type: null, file: null, sampleName: null },
    image2: { type: null, file: null, sampleName: null },
  };

  const form = document.getElementById("compare-form");
  const compareBtn = document.getElementById("compare-btn");
  const resetBtn = document.getElementById("reset-btn");
  const errorBox = document.getElementById("error-box");
  const resultSection = document.getElementById("result");
  const loadingOverlay = document.getElementById("loading-overlay");

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  function clearError() {
    errorBox.hidden = true;
    errorBox.textContent = "";
  }

  function setLoading(isLoading) {
    loadingOverlay.hidden = !isLoading;
    compareBtn.disabled = isLoading || !bothSlotsFilled();
  }

  function bothSlotsFilled() {
    return slots.every((slot) => state[slot].type !== null);
  }

  function updateCompareButton() {
    compareBtn.disabled = !bothSlotsFilled();
  }

  function renderPreview(slot, src) {
    const preview = document.getElementById(`${slot}_preview`);
    preview.innerHTML = "";
    const img = document.createElement("img");
    img.src = src;
    img.alt = `${slot} preview`;
    preview.appendChild(img);
  }

  function clearPreview(slot) {
    const preview = document.getElementById(`${slot}_preview`);
    preview.innerHTML = `
      <span class="dropzone__icon">🖼️</span>
      <span class="dropzone__text">คลิกหรือลากไฟล์รูปภาพมาวางที่นี่</span>
    `;
  }

  function selectFile(slot, file) {
    if (!file.type.startsWith("image/")) {
      showError("กรุณาเลือกไฟล์รูปภาพเท่านั้น");
      return;
    }
    state[slot] = { type: "file", file, sampleName: null };
    document.getElementById(`${slot}_sample`).value = "";
    clearSampleSelection(slot);

    const reader = new FileReader();
    reader.onload = (e) => renderPreview(slot, e.target.result);
    reader.readAsDataURL(file);

    clearError();
    hideResult();
    updateCompareButton();
  }

  function selectSample(slot, sampleName, imgSrc) {
    state[slot] = { type: "sample", file: null, sampleName };
    document.getElementById(`${slot}_sample`).value = sampleName;
    document.getElementById(`${slot}_file`).value = "";

    renderPreview(slot, imgSrc);
    highlightSampleSelection(slot, sampleName);

    clearError();
    hideResult();
    updateCompareButton();
  }

  function clearSampleSelection(slot) {
    document
      .querySelectorAll(`#${slot}_samples .sample-thumb`)
      .forEach((btn) => btn.classList.remove("is-selected"));
  }

  function highlightSampleSelection(slot, sampleName) {
    document.querySelectorAll(`#${slot}_samples .sample-thumb`).forEach((btn) => {
      btn.classList.toggle("is-selected", btn.dataset.sample === sampleName);
    });
  }

  function hideResult() {
    resultSection.hidden = true;
  }

  function resetAll() {
    slots.forEach((slot) => {
      state[slot] = { type: null, file: null, sampleName: null };
      document.getElementById(`${slot}_file`).value = "";
      document.getElementById(`${slot}_sample`).value = "";
      clearPreview(slot);
      clearSampleSelection(slot);
    });
    clearError();
    hideResult();
    updateCompareButton();
  }

  // ---- ตั้งค่า event listener ให้แต่ละ slot ----
  slots.forEach((slot) => {
    const dropzone = document.querySelector(`.image-card[data-slot="${slot}"] .dropzone`);
    const fileInput = document.getElementById(`${slot}_file`);

    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files[0]) {
        selectFile(slot, fileInput.files[0]);
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
      if (file) {
        selectFile(slot, file);
      }
    });

    document.querySelectorAll(`#${slot}_samples .sample-thumb`).forEach((btn) => {
      btn.addEventListener("click", () => {
        const sampleName = btn.dataset.sample;
        const imgSrc = btn.querySelector("img").src;
        selectSample(slot, sampleName, imgSrc);
      });
    });
  });

  resetBtn.addEventListener("click", resetAll);

  // ---- ส่งฟอร์มไปยัง backend ----
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearError();

    if (!bothSlotsFilled()) {
      showError("กรุณาเลือกรูปภาพทั้งสองฝั่งก่อนเปรียบเทียบ");
      return;
    }

    const formData = new FormData();

    slots.forEach((slot) => {
      const s = state[slot];
      if (s.type === "file") {
        formData.append(`${slot}_file`, s.file);
      } else if (s.type === "sample") {
        formData.append(`${slot}_sample`, s.sampleName);
      }
    });

    formData.append("l2_normalize", document.getElementById("l2_normalize").checked ? "true" : "false");
    formData.append("quantize", document.getElementById("quantize").checked ? "true" : "false");

    setLoading(true);
    try {
      const resp = await fetch("/api/compare", {
        method: "POST",
        body: formData,
      });
      const data = await resp.json();

      if (!resp.ok || !data.ok) {
        showError(data.error || "เกิดข้อผิดพลาดที่ไม่ทราบสาเหตุ");
        return;
      }

      renderResult(data.similarity);
    } catch (err) {
      showError("ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์ได้: " + err.message);
    } finally {
      setLoading(false);
    }
  });

  function renderResult(similarity) {
    const value = document.getElementById("result-value");
    const tag = document.getElementById("result-tag");
    const marker = document.getElementById("gauge-marker");

    value.textContent = similarity.toFixed(4);

    let tagText;
    if (similarity >= 0.75) {
      tagText = "คล้ายกันมาก";
    } else if (similarity >= 0.4) {
      tagText = "ค่อนข้างคล้ายกัน";
    } else if (similarity >= 0) {
      tagText = "ไม่ค่อยคล้ายกัน";
    } else {
      tagText = "แตกต่างกัน";
    }
    tag.textContent = tagText;

    // แปลงค่า similarity (-1..1) เป็นตำแหน่ง % บน gauge (0..100)
    const clamped = Math.max(-1, Math.min(1, similarity));
    const percent = ((clamped + 1) / 2) * 100;
    marker.style.left = `${percent}%`;

    resultSection.hidden = false;
    resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  updateCompareButton();
})();
