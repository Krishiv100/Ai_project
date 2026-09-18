(() => {
  const API = (window.QCAS_API_BASE || "http://localhost:8000").replace(/\/$/, "");

  const $ = (id) => document.getElementById(id);
  const tabs = document.querySelectorAll(".tab");
  const imagePanel = $("imagePanel");
  const videoPanel = $("videoPanel");
  const imageInput = $("imageInput");
  const videoInput = $("videoInput");
  const imagePreview = $("imagePreview");
  const videoPreview = $("videoPreview");
  const imageEmpty = $("imageEmpty");
  const videoEmpty = $("videoEmpty");
  const imageDrop = $("imageDrop");
  const videoDrop = $("videoDrop");
  const analyzeImageBtn = $("analyzeImageBtn");
  const analyzeVideoBtn = $("analyzeVideoBtn");
  const clearImageBtn = $("clearImageBtn");
  const clearVideoBtn = $("clearVideoBtn");
  const progressCard = $("progressCard");
  const progressTitle = $("progressTitle");
  const progressText = $("progressText");
  const errorCard = $("errorCard");
  const results = $("results");
  const videoResults = $("videoResults");
  const faceResults = $("faceResults");

  let imageFile = null;
  let videoFile = null;
  let imageObjectUrl = null;
  let videoObjectUrl = null;

  function fmt(v, digits = 3) {
    const n = Number(v);
    return Number.isFinite(n) ? n.toFixed(digits) : "—";
  }

  function showError(message) {
    errorCard.textContent = message || "Something went wrong.";
    errorCard.classList.remove("hidden");
    progressCard.classList.add("hidden");
  }

  function hideError() {
    errorCard.classList.add("hidden");
    errorCard.textContent = "";
  }

  function setBusy(title, text) {
    hideError();
    results.classList.add("hidden");
    progressTitle.textContent = title;
    progressText.textContent = text;
    progressCard.classList.remove("hidden");
    analyzeImageBtn.disabled = true;
    analyzeVideoBtn.disabled = true;
  }

  function clearBusy() {
    progressCard.classList.add("hidden");
    analyzeImageBtn.disabled = !imageFile;
    analyzeVideoBtn.disabled = !videoFile;
  }

  function setGauge(prob) {
    const p = Math.max(0, Math.min(1, Number(prob) || 0));
    $("gauge").style.setProperty("--p", `${p * 360}deg`);
    $("fakeProbability").textContent = `${Math.round(p * 100)}%`;
  }

  function displayResult(data, mode) {
    const prediction = String(data.prediction || "—").toUpperCase();
    const predictionEl = $("prediction");
    predictionEl.textContent = prediction;
    predictionEl.className = prediction === "FAKE" ? "fake" : "real";

    const fakeProb = Number(data.fake_probability ?? data.video_fake_probability ?? 0);
    const confidence = Number(data.confidence ?? (prediction === "FAKE" ? fakeProb : 1 - fakeProb));
    $("confidenceText").textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
    setGauge(fakeProb);

    $("threshold").textContent = fmt(data.threshold);
    $("quality").textContent = fmt(data.estimated_degradation ?? data.mean_estimated_degradation);
    $("spatialGate").textContent = fmt(data.spatial_gate ?? data.mean_spatial_gate);
    $("spectralGate").textContent = fmt(data.spectral_gate ?? data.mean_spectral_gate);
    $("boundary1").textContent = fmt(data.frequency_boundary_1 ?? data.mean_b1);
    $("boundary2").textContent = fmt(data.frequency_boundary_2 ?? data.mean_b2);

    if (mode === "image") {
      videoResults.classList.add("hidden");
      faceResults.classList.remove("hidden");
      if (data.face_crop) {
        $("facePreview").src = data.face_crop;
      }
    } else {
      faceResults.classList.add("hidden");
      videoResults.classList.remove("hidden");
      $("framesAnalyzed").textContent = `${data.faces_analyzed || 0} detected faces analyzed from ${data.frames_sampled || 0} sampled frames.`;
      const gallery = $("videoGallery");
      gallery.innerHTML = "";
      (data.sample_faces || []).forEach((src, i) => {
        const img = new Image();
        img.src = src;
        img.alt = `Sampled face ${i + 1}`;
        gallery.appendChild(img);
      });
    }

    results.classList.remove("hidden");
  }

  async function request(path, formData) {
    const res = await fetch(`${API}${path}`, {
      method: "POST",
      body: formData
    });

    let body;
    try {
      body = await res.json();
    } catch {
      body = {};
    }

    if (!res.ok) {
      throw new Error(body.detail || body.error || `Server returned ${res.status}.`);
    }
    return body;
  }

  async function analyzeImage() {
    if (!imageFile) return;
    setBusy("Analyzing image…", "Detecting the main face and running QCAS-DF.");
    try {
      const form = new FormData();
      form.append("file", imageFile);
      const data = await request("/api/analyze/image", form);
      displayResult(data, "image");
    } catch (err) {
      showError(err.message);
    } finally {
      clearBusy();
    }
  }

  async function analyzeVideo() {
    if (!videoFile) return;
    setBusy("Analyzing video…", "Sampling frames, detecting faces and aggregating QCAS-DF predictions.");
    try {
      const form = new FormData();
      form.append("file", videoFile);
      const data = await request("/api/analyze/video", form);
      displayResult(data, "video");
    } catch (err) {
      showError(err.message);
    } finally {
      clearBusy();
    }
  }

  function loadImage(file) {
    if (!file || !file.type.startsWith("image/")) {
      showError("Please choose a supported image file.");
      return;
    }
    imageFile = file;
    if (imageObjectUrl) URL.revokeObjectURL(imageObjectUrl);
    imageObjectUrl = URL.createObjectURL(file);
    imagePreview.src = imageObjectUrl;
    imagePreview.style.display = "block";
    imageEmpty.style.display = "none";
    analyzeImageBtn.disabled = false;
    clearImageBtn.disabled = false;
    hideError();
    results.classList.add("hidden");
  }

  function loadVideo(file) {
    if (!file || !file.type.startsWith("video/")) {
      showError("Please choose a supported video file.");
      return;
    }
    videoFile = file;
    if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
    videoObjectUrl = URL.createObjectURL(file);
    videoPreview.src = videoObjectUrl;
    videoPreview.style.display = "block";
    videoEmpty.style.display = "none";
    analyzeVideoBtn.disabled = false;
    clearVideoBtn.disabled = false;
    hideError();
    results.classList.add("hidden");
  }

  function clearImage() {
    imageFile = null;
    imageInput.value = "";
    if (imageObjectUrl) URL.revokeObjectURL(imageObjectUrl);
    imageObjectUrl = null;
    imagePreview.removeAttribute("src");
    imagePreview.style.display = "none";
    imageEmpty.style.display = "block";
    analyzeImageBtn.disabled = true;
    clearImageBtn.disabled = true;
    results.classList.add("hidden");
    hideError();
  }

  function clearVideo() {
    videoFile = null;
    videoInput.value = "";
    if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
    videoObjectUrl = null;
    videoPreview.pause();
    videoPreview.removeAttribute("src");
    videoPreview.load();
    videoPreview.style.display = "none";
    videoEmpty.style.display = "block";
    analyzeVideoBtn.disabled = true;
    clearVideoBtn.disabled = true;
    results.classList.add("hidden");
    hideError();
  }

  function installDropzone(zone, handler) {
    ["dragenter", "dragover"].forEach((name) => {
      zone.addEventListener(name, (e) => {
        e.preventDefault();
        zone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach((name) => {
      zone.addEventListener(name, (e) => {
        e.preventDefault();
        zone.classList.remove("dragover");
      });
    });
    zone.addEventListener("drop", (e) => {
      const file = e.dataTransfer.files && e.dataTransfer.files[0];
      if (file) handler(file);
    });
  }

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => {
        const active = t === tab;
        t.classList.toggle("active", active);
        t.setAttribute("aria-selected", String(active));
      });
      const isImage = tab.dataset.tab === "image";
      imagePanel.classList.toggle("active", isImage);
      videoPanel.classList.toggle("active", !isImage);
      results.classList.add("hidden");
      hideError();
    });
  });

  $("pickImageBtn").addEventListener("click", () => imageInput.click());
  $("pickVideoBtn").addEventListener("click", () => videoInput.click());
  imageInput.addEventListener("change", () => loadImage(imageInput.files[0]));
  videoInput.addEventListener("change", () => loadVideo(videoInput.files[0]));
  analyzeImageBtn.addEventListener("click", analyzeImage);
  analyzeVideoBtn.addEventListener("click", analyzeVideo);
  clearImageBtn.addEventListener("click", clearImage);
  clearVideoBtn.addEventListener("click", clearVideo);

  installDropzone(imageDrop, loadImage);
  installDropzone(videoDrop, loadVideo);

  async function checkHealth() {
    const dot = $("apiDot");
    const label = $("apiLabel");
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 5000);
      const res = await fetch(`${API}/health`, { signal: controller.signal });
      clearTimeout(timer);
      if (!res.ok) throw new Error();
      const data = await res.json();
      dot.style.background = "var(--good)";
      dot.style.boxShadow = "0 0 0 4px rgba(85,227,156,.09)";
      label.textContent = data.model_loaded ? "Model online" : "API online";
    } catch {
      dot.style.background = "var(--bad)";
      dot.style.boxShadow = "0 0 0 4px rgba(255,107,122,.09)";
      label.textContent = "API offline";
    }
  }

  checkHealth();
})();
