/* VisionTrack AI – browser side. Plain JavaScript, no frameworks. */
(() => {
  const $ = (sel) => document.querySelector(sel);
  const el = {
    sys: $("#sys"), alert: $("#alert"),
    model: $("#model"), tracker: $("#tracker"), preset: $("#preset"),
    conf: $("#conf"), confOut: $("#confOut"), trails: $("#trails"),
    tabVideo: $("#tab-video"), tabWebcam: $("#tab-webcam"),
    panelVideo: $("#panel-video"), panelWebcam: $("#panel-webcam"),
    file: $("#file"), fileName: $("#fileName"), fileHint: $("#fileHint"), drop: $("#drop"),
    processBtn: $("#processBtn"), cancelBtn: $("#cancelBtn"), downloadBtn: $("#downloadBtn"),
    outVideo: $("#outVideo"), preview: $("#preview"), videoEmpty: $("#videoEmpty"),
    progressWrap: $("#progressWrap"), progressFill: $("#progressFill"),
    progressTrack: $("#progressTrack"), progressText: $("#progressText"),
    summary: $("#summary"),
    camStart: $("#camStart"), camStop: $("#camStop"), camFeed: $("#camFeed"),
    camEmpty: $("#camEmpty"), camStats: $("#camStats"),
  };

  let system = null;
  let jobId = null, pollTimer = null;
  let statsTimer = null, confTimer = null, webcamRunning = false;

  // ---------- helpers ----------
  function showAlert(message, kind = "error") {
    el.alert.textContent = message;
    el.alert.className = "alert" + (kind === "info" ? " info" : "");
    el.alert.hidden = false;
  }
  function clearAlert() { el.alert.hidden = true; }

  async function api(url, options) {
    let response;
    try {
      response = await fetch(url, options);
    } catch {
      throw new Error("Cannot reach the server. Is `python app.py` still running?");
    }
    let data = null;
    try { data = await response.json(); } catch { /* not JSON */ }
    if (!response.ok) throw new Error((data && data.error) || `Request failed (${response.status}).`);
    return data;
  }

  function getSettings() {
    return {
      model: el.model.value,
      tracker: el.tracker.value,
      preset: el.preset.value,
      conf: el.conf.value,
      tracking: document.querySelector('input[name="mode"]:checked').value === "track",
      trails: el.trails.checked,
    };
  }

  function fillSelect(select, options, selected) {
    select.innerHTML = "";
    Object.entries(options).forEach(([value, label]) => {
      const opt = document.createElement("option");
      opt.value = value; opt.textContent = label;
      if (value === selected) opt.selected = true;
      select.appendChild(opt);
    });
  }

  // ---------- startup ----------
  async function init() {
    try {
      system = await api("/api/system");
    } catch (err) {
      el.sys.textContent = "Server not reachable";
      showAlert(err.message);
      return;
    }
    fillSelect(el.model, system.models, system.default_model);
    fillSelect(el.tracker, system.trackers, system.default_tracker);
    fillSelect(el.preset, system.presets, "all");
    el.conf.min = system.conf.min; el.conf.max = system.conf.max; el.conf.value = system.conf.default;
    el.confOut.textContent = Number(el.conf.value).toFixed(2);
    el.sys.textContent = system.device;
    el.fileHint.textContent =
      `${system.allowed_extensions.join(", ")} · up to ${system.max_upload_mb} MB`;
    if (!system.ffmpeg) {
      showAlert("ffmpeg was not found, so the processed video may not play in the browser. You can still download it.", "info");
    }
  }

  // ---------- tabs ----------
  function selectTab(name) {
    const video = name === "video";
    el.tabVideo.classList.toggle("active", video);
    el.tabWebcam.classList.toggle("active", !video);
    el.tabVideo.setAttribute("aria-selected", video);
    el.tabWebcam.setAttribute("aria-selected", !video);
    el.panelVideo.hidden = !video;
    el.panelWebcam.hidden = video;
    clearAlert();
  }
  el.tabVideo.addEventListener("click", () => selectTab("video"));
  el.tabWebcam.addEventListener("click", () => selectTab("webcam"));

  // ---------- confidence slider ----------
  el.conf.addEventListener("input", () => {
    el.confOut.textContent = Number(el.conf.value).toFixed(2);
    if (webcamRunning) {            // live update, slightly delayed
      clearTimeout(confTimer);
      confTimer = setTimeout(() => {
        api("/api/webcam/confidence", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ conf: el.conf.value }),
        }).catch((e) => showAlert(e.message));
      }, 250);
    }
  });

  // ---------- video file ----------
  function checkFile(file) {
    if (!file) return "Choose a video file first.";
    const ext = "." + file.name.split(".").pop().toLowerCase();
    if (system && !system.allowed_extensions.includes(ext)) {
      return `Unsupported file type "${ext}". Use: ${system.allowed_extensions.join(", ")}.`;
    }
    if (system && file.size > system.max_upload_mb * 1024 * 1024) {
      return `This file is larger than the ${system.max_upload_mb} MB limit.`;
    }
    return null;
  }

  function setFile(file) {
    el.fileName.textContent = file ? file.name : "Choose a video file or drop it here";
  }
  el.file.addEventListener("change", () => setFile(el.file.files[0]));
  ["dragenter", "dragover"].forEach((t) => el.drop.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => el.drop.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.remove("over"); }));
  el.drop.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length) { el.file.files = e.dataTransfer.files; setFile(el.file.files[0]); }
  });

  function setProgress(fraction, text) {
    el.progressWrap.hidden = false;
    const pct = Math.round(Math.max(0, Math.min(1, fraction)) * 100);
    el.progressFill.style.width = pct + "%";
    el.progressTrack.setAttribute("aria-valuenow", pct);
    el.progressText.textContent = text;
  }

  function setBusy(busy) {
    el.processBtn.disabled = busy;
    el.cancelBtn.hidden = !busy;
  }

  el.processBtn.addEventListener("click", () => {
    clearAlert();
    const file = el.file.files[0];
    const problem = checkFile(file);
    if (problem) { showAlert(problem); return; }

    el.summary.hidden = true; el.downloadBtn.hidden = true;
    el.outVideo.hidden = true; el.outVideo.removeAttribute("src");
    el.preview.hidden = true; el.videoEmpty.hidden = false;
    el.videoEmpty.textContent = "Uploading…";
    setBusy(true);

    const form = new FormData();
    form.append("video", file);
    Object.entries(getSettings()).forEach(([k, v]) => form.append(k, v));

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/upload");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) setProgress(e.loaded / e.total, `Uploading… ${Math.round((e.loaded / e.total) * 100)}%`);
    };
    xhr.onerror = () => { setBusy(false); showAlert("Upload failed. Is the server still running?"); };
    xhr.onload = () => {
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch { /* ignore */ }
      if (xhr.status >= 400) {
        setBusy(false); el.progressWrap.hidden = true;
        el.videoEmpty.textContent = "Choose a video below.";
        showAlert(data.error || `Upload failed (${xhr.status}).`);
        return;
      }
      jobId = data.job_id;
      setProgress(0, "Starting…");
      pollJob();
    };
    xhr.send(form);
  });

  el.cancelBtn.addEventListener("click", () => {
    if (jobId) api(`/api/jobs/${jobId}/cancel`, { method: "POST" }).catch(() => {});
  });

  function pollJob() {
    clearTimeout(pollTimer);
    api(`/api/jobs/${jobId}`).then((job) => {
      if (job.status === "error" || job.status === "cancelled") {
        setBusy(false); el.progressWrap.hidden = true;
        el.preview.hidden = true; el.videoEmpty.hidden = false;
        el.videoEmpty.textContent = job.status === "cancelled" ? "Processing cancelled." : "Processing stopped.";
        if (job.status === "error") showAlert(job.error);
        return;
      }
      if (job.status === "done") { finishJob(job); return; }

      if (job.frames_done > 0) {
        el.preview.src = `/api/jobs/${jobId}/preview?t=${Date.now()}`;
        el.preview.hidden = false; el.videoEmpty.hidden = true;
      }
      let text = job.message;
      if (job.status === "processing") {
        text = job.total_frames
          ? `Processing frame ${job.frames_done} of ${job.total_frames} · ${job.processing_fps} frames/s`
          : `Processing frame ${job.frames_done} · ${job.processing_fps} frames/s`;
      }
      setProgress(job.progress, text);
      pollTimer = setTimeout(pollJob, 700);
    }).catch((err) => {
      setBusy(false); showAlert(err.message);
    });
  }

  function finishJob(job) {
    setBusy(false);
    setProgress(1, "Finished.");
    el.preview.hidden = true; el.videoEmpty.hidden = true;
    el.outVideo.src = `/api/jobs/${jobId}/video`;
    el.outVideo.hidden = false;
    el.downloadBtn.href = `/api/jobs/${jobId}/video?download=1`;
    el.downloadBtn.hidden = false;
    if (!job.playable) {
      showAlert("Your browser may not be able to play this file inline, but the download will work in most video players.", "info");
    }
    renderSummary(job.summary);
  }

  function renderSummary(summary) {
    el.summary.replaceChildren();
    if (!summary) return;
    if (!summary.classes.length) {
      const p = document.createElement("p");
      p.textContent = `No objects were found in ${summary.frames} frames. Try lowering the confidence threshold or choosing "All" objects.`;
      el.summary.append(p); el.summary.hidden = false; return;
    }
    const table = document.createElement("table");
    const head = table.insertRow();
    ["Object", summary.tracking ? "Track IDs assigned" : null, "Most seen in one frame"]
      .filter(Boolean).forEach((t) => { const th = document.createElement("th"); th.textContent = t; head.append(th); });
    summary.classes.forEach((c) => {
      const row = table.insertRow();
      row.insertCell().textContent = c.name;
      if (summary.tracking) row.insertCell().textContent = c.unique_ids;
      row.insertCell().textContent = c.peak_in_frame;
    });
    el.summary.append(table);
    if (summary.tracking) {
      const note = document.createElement("p");
      note.textContent = "IDs assigned can be higher than the real number of objects: a hidden or fast-moving object may get a new ID when it reappears.";
      el.summary.append(note);
    }
    el.summary.hidden = false;
  }

  // ---------- webcam ----------
  function webcamStopped(message) {
    webcamRunning = false;
    clearInterval(statsTimer);
    el.camFeed.hidden = true; el.camFeed.removeAttribute("src");
    el.camEmpty.hidden = false;
    el.camEmpty.textContent = "Press Start webcam. The camera is opened by the computer running this app.";
    el.camStart.hidden = false; el.camStart.disabled = false; el.camStop.hidden = true;
    el.camStats.textContent = "";
    if (message) showAlert(message);
  }

  el.camStart.addEventListener("click", async () => {
    clearAlert();
    el.camStart.disabled = true;
    el.camEmpty.textContent = "Opening camera and loading model…";
    try {
      await api("/api/webcam/start", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(getSettings()),
      });
    } catch (err) {
      webcamStopped(err.message);
      return;
    }
    webcamRunning = true;
    el.camFeed.src = `/api/webcam/feed?t=${Date.now()}`;
    el.camFeed.hidden = false; el.camEmpty.hidden = true;
    el.camStart.hidden = true; el.camStop.hidden = false;
    statsTimer = setInterval(updateCamStats, 1000);
  });

  el.camStop.addEventListener("click", async () => {
    try { await api("/api/webcam/stop", { method: "POST" }); } catch { /* ignore */ }
    webcamStopped();
  });

  el.camFeed.addEventListener("error", () => {
    if (webcamRunning) updateCamStats();
  });

  async function updateCamStats() {
    try {
      const s = await api("/api/webcam/status");
      if (!s.running) {
        webcamStopped(s.error || "The webcam stopped.");
        return;
      }
      el.camStats.textContent = `${s.fps} FPS · ${s.objects_now} objects now · ${s.unique_ids} track IDs so far`;
    } catch (err) {
      webcamStopped(err.message);
    }
  }

  window.addEventListener("beforeunload", () => {
    if (webcamRunning) navigator.sendBeacon("/api/webcam/stop");
  });

  init();
})();
