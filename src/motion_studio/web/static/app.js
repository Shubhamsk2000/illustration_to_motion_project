"use strict";

// ---- helpers ---------------------------------------------------------------------------------------------

const $ = (sel, el = document) => el.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const app = $("#app");
let OPTIONS = null;
let STATE = null;
let TAB = "scenes";
let pollTimer = null;

async function api(method, url, body) {
  const init = { method, headers: {} };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const res = await fetch(url, init);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `${res.status} ${res.statusText}`);
  return data;
}

let toastTimer;
function toast(msg, bad = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "toast" + (bad ? " bad" : "");
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), bad ? 6000 : 2500);
}

const fmtSecs = (s) => {
  s = Math.round(s);
  return s >= 60 ? `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}` : `${s}s`;
};
const fmtSize = (b) => (b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1e3))} KB`);

function pickFile(accept) {
  return new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = accept;
    input.onchange = () => resolve(input.files[0] || null);
    input.click();
  });
}

// ---- routing ---------------------------------------------------------------------------------------------

async function route() {
  clearInterval(pollTimer);
  pollTimer = null;
  if (!OPTIONS) OPTIONS = await api("GET", "/api/options");
  const m = location.hash.match(/^#\/ep\/([a-z0-9_-]+)(?:\/(scenes|script|settings|output))?/);
  try {
    if (m) {
      TAB = m[2] || "scenes";
      if (STATE && STATE.name === m[1] && $("#tab")) {  // tab switch only: keep any running build's progress
        renderEpisode();
        if (STATE.job && STATE.job.state === "running") startPolling();
        return;
      }
      await showEpisode(m[1]);
    }
    else await showList();
  } catch (e) {
    app.innerHTML = `<p class="error">${esc(e.message)}</p><p><a href="#/">Back to episodes</a></p>`;
  }
}
window.addEventListener("hashchange", route);

// ---- episode list ----------------------------------------------------------------------------------------

async function showList() {
  $("#crumbs").textContent = "";
  document.title = "Motion Studio";
  const eps = await api("GET", "/api/episodes");
  app.innerHTML = `
    <div class="row spread"><h1>Episodes</h1></div>
    <div class="grid">
      ${eps.map((e) => `
        <a class="card" href="#/ep/${esc(e.name)}">
          <div class="thumb" style="${e.thumb ? `background-image:url('${esc(e.thumb)}')` : ""}"></div>
          <div class="body">
            <h2>${esc(e.title)}</h2>
            <div class="muted small">${e.scenes} scenes · ${e.images}/${e.scenes} images ·
              ${e.narration ? "narration ✓" : "no narration yet"}</div>
            ${e.job && e.job.state === "running" ? `<div class="chip warn">building ${Math.round(e.job.progress * 100)}%</div>` : ""}
            ${e.exports.length ? `<div class="chip ok">${e.exports.length} video(s)</div>` : ""}
          </div>
        </a>`).join("")}
      <form class="card newcard" id="newep">
        <h2>New episode</h2>
        <input name="title" placeholder="Title, e.g. A Day in the Life of a Roman Soldier" required>
        <input name="name" placeholder="folder name, e.g. roman-soldier" pattern="[a-z0-9][a-z0-9_-]*" required>
        <div class="muted small" style="margin-top:6px">Lowercase letters, numbers and dashes.</div>
        <button class="primary" style="margin-top:10px">Create</button>
      </form>
    </div>`;
  const form = $("#newep");
  // form.name / form.title are the form's own properties, so look the inputs up explicitly
  const titleIn = form.elements.namedItem("title");
  const nameIn = form.elements.namedItem("name");
  titleIn.addEventListener("input", () => {
    if (!nameIn.dataset.touched) {
      nameIn.value = titleIn.value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 50);
    }
  });
  nameIn.addEventListener("input", () => (nameIn.dataset.touched = "1"));
  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    try {
      await api("POST", "/api/episodes", { name: nameIn.value, title: titleIn.value });
      location.hash = `#/ep/${nameIn.value}`;
    } catch (e) {
      toast(e.message, true);
    }
  });
}

// ---- episode page ----------------------------------------------------------------------------------------

async function showEpisode(name) {
  STATE = await api("GET", `/api/episodes/${name}`);
  renderEpisode();
  if (STATE.job && STATE.job.state === "running") startPolling();
}

function renderEpisode() {
  const s = STATE;
  document.title = `${s.title} · Motion Studio`;
  $("#crumbs").innerHTML = `/ <a href="#/">Episodes</a> / ${esc(s.title)}`;
  const sum = s.summary;
  app.innerHTML = `
    <div class="ephead">
      <div class="stack">
        <h1>${esc(s.title)}</h1>
        <div class="row">
          <span class="chip">${sum.scenes} scenes · ${sum.words} words · about ${sum.est_minutes} min</span>
          <span class="chip ${sum.images_found === sum.scenes ? "ok" : "warn"}">${sum.images_found}/${sum.scenes} images</span>
          <span class="chip ${sum.chapters.length >= 3 ? "ok" : sum.chapters.length ? "warn" : ""}">${sum.chapters.length} chapters</span>
          ${sum.long_scenes.length ? `<span class="chip warn" title="Over 40 words on one image">long: ${sum.long_scenes.join(", ")}</span>` : ""}
        </div>
      </div>
      <div class="panel stack" style="min-width:320px">
        ${audioRow("narration", "Narration", s.narration, "Your voice recording")}
        ${audioRow("music", "Music", s.music, "Optional, plays quietly under your voice")}
      </div>
    </div>
    <div class="panel build" id="build"></div>
    <div class="tabs">
      ${["scenes", "script", "settings", "output"].map((t) => `<button data-tab="${t}" class="${TAB === t ? "active" : ""}">${t[0].toUpperCase() + t.slice(1)}</button>`).join("")}
    </div>
    <div id="tab"></div>`;
  app.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
    location.hash = `#/ep/${s.name}/${b.dataset.tab}`;
  }));
  app.querySelectorAll("[data-upload-audio]").forEach((b) => b.addEventListener("click", () => uploadAudio(b.dataset.uploadAudio)));
  renderBuild();
  ({ scenes: renderScenes, script: renderScript, settings: renderSettings, output: renderOutput })[TAB]();
}

function audioRow(kind, label, info, hint) {
  return `
    <div>
      <div class="row spread"><strong>${label}</strong>
        <button data-upload-audio="${kind}">${info ? "Replace" : "Upload"}</button></div>
      ${info ? `<div class="muted small">${esc(info.name)}${info.seconds ? ` · ${fmtSecs(info.seconds)}` : ""}</div>
                <audio controls preload="none" src="${esc(info.url)}" style="width:100%;margin-top:4px"></audio>`
             : `<div class="muted small">${hint}</div>`}
    </div>`;
}

async function uploadAudio(kind) {
  const file = await pickFile("audio/*,.wav,.mp3,.m4a,.flac,.aac,.ogg");
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  toast(`Uploading ${file.name}…`);
  try {
    STATE = await api("POST", `/api/episodes/${STATE.name}/audio/${kind}`, fd);
    renderEpisode();
    toast(`${kind === "music" ? "Music" : "Narration"} saved`);
  } catch (e) {
    toast(e.message, true);
  }
}

// ---- build panel -----------------------------------------------------------------------------------------

function renderBuild() {
  const el = $("#build");
  if (!el) return;
  const s = STATE;
  const job = s.job;
  const missing = s.summary.missing_images.length;
  const running = job && job.state === "running";
  let status = "";
  if (running) {
    status = `
      <div class="row" style="margin-top:12px">
        <div class="bar"><div style="width:${Math.round(job.progress * 100)}%"></div></div>
        <span>${Math.round(job.progress * 100)}%</span>
        <button class="danger" id="cancel">Cancel</button>
      </div>
      <div class="muted small">${job.draft ? "Draft" : "Final"} · ${esc(job.detail || job.stage || "starting")} · ${fmtSecs(job.elapsed)}</div>`;
  } else if (job && job.state === "done" && job.result) {
    const r = job.result;
    const warn = [];
    if (r.script_match < 0.7) warn.push("the recording doesn't follow the script closely; check the timing");
    r.chapter_problems.forEach((p) => warn.push(`chapters: ${p}`));
    if (r.missing_images.length) warn.push(`${r.missing_images.length} scene(s) used title cards`);
    status = `
      <div style="margin-top:10px">✓ ${job.draft ? "Draft" : "Final video"} ready in ${fmtSecs(job.elapsed)} ·
        ${fmtSecs(r.seconds)} long · script matched ${Math.round(r.script_match * 100)}% ·
        ${r.segments_rendered} scenes rendered, ${r.segments_reused} reused
        · <button class="link" id="goout">Watch it</button></div>
      ${warn.map((w) => `<div class="small" style="color:var(--warn)">! ${esc(w)}</div>`).join("")}`;
  } else if (job && job.state === "failed") {
    status = `<div class="error" style="margin-top:10px">Build failed: ${esc(job.error)}</div>`;
  } else if (job && job.state === "cancelled") {
    status = `<div class="muted" style="margin-top:10px">Build cancelled. Finished scenes are kept for next time.</div>`;
  }
  el.innerHTML = `
    <div class="row spread">
      <div class="row">
        <button class="primary" id="draft" ${running || !s.narration ? "disabled" : ""}>Build draft</button>
        <button id="final" ${running || !s.narration ? "disabled" : ""}>Build final 1080p</button>
      </div>
      <div class="muted small">
        ${!s.narration ? "Upload your narration to build." :
          missing ? `${missing} image(s) missing: they'll show as title cards.` :
          "Draft: quick 540p check. Final: full quality for YouTube."}
      </div>
    </div>
    ${status}`;
  $("#draft", el)?.addEventListener("click", () => startBuild(true));
  $("#final", el)?.addEventListener("click", () => startBuild(false));
  $("#cancel", el)?.addEventListener("click", cancelBuild);
  $("#goout", el)?.addEventListener("click", () => { location.hash = `#/ep/${s.name}/output`; });
}

async function startBuild(draft) {
  try {
    STATE.job = await api("POST", `/api/episodes/${STATE.name}/build`, { draft });
    renderBuild();
    startPolling();
  } catch (e) {
    toast(e.message, true);
  }
}

async function cancelBuild() {
  await api("DELETE", `/api/episodes/${STATE.name}/build`);
  STATE.job = await api("GET", `/api/episodes/${STATE.name}/build`);
  renderBuild();
}

function startPolling() {
  clearInterval(pollTimer);
  const name = STATE.name;
  pollTimer = setInterval(async () => {
    if (!STATE || STATE.name !== name) return clearInterval(pollTimer);
    try {
      const job = await api("GET", `/api/episodes/${name}/build`);
      STATE.job = job;
      if (job.state !== "running") {
        clearInterval(pollTimer);
        STATE = await api("GET", `/api/episodes/${name}`);
        renderEpisode();
      } else {
        renderBuild();
      }
    } catch (e) {
      clearInterval(pollTimer);
    }
  }, 1500);
}

// ---- scenes tab ------------------------------------------------------------------------------------------

function renderScenes() {
  const tab = $("#tab");
  if (!STATE.scenes.length) {
    tab.innerHTML = `<p class="muted">No scenes yet. Write the script first (Script tab): one paragraph per image.</p>`;
    return;
  }
  let html = `<p class="muted small">Drop an image on a scene (or click its box) to add it. Looks are picked from each
    scene's words; change any of them here. "Preview" renders just that scene in a few seconds.</p>`;
  let chapter = null;
  for (const sc of STATE.scenes) {
    if (sc.chapter && sc.chapter !== chapter) {
      chapter = sc.chapter;
      html += `<div class="chapter">${esc(chapter)}</div>`;
    }
    html += `<div class="scene" data-id="${sc.id}">${sceneInner(sc)}</div>`;
  }
  tab.innerHTML = html;
  tab.querySelectorAll(".scene").forEach(bindScene);
}

function option(value, label, selected) {
  return `<option value="${esc(value)}" ${selected ? "selected" : ""}>${esc(label)}</option>`;
}

function lookSelect(sc, key, autoLabel) {
  const cur = sc.override[key] ?? "";
  return `<select data-key="${key}">${option("", autoLabel, cur === "")}${OPTIONS.looks[key]
    .map((v) => option(v, v.replace(/_/g, " "), cur === v)).join("")}</select>`;
}

function sceneInner(sc) {
  const s = STATE.settings;
  const presetAuto = sc.auto ? `Auto: ${sc.auto.look} (“${sc.auto.word}”)` : `Default: ${s.default_preset}`;
  const fromLook = (k) => `From look: ${String(sc.spec[k]).replace(/_/g, " ")}`;
  const changed = Object.keys(sc.override).length;
  return `
    <div class="media ${sc.image ? "" : "empty"}" data-media title="Click or drop an image to replace">
      ${sc.image ? `<img src="${esc(sc.image.thumb)}" alt="Scene ${sc.id}">` : "Drop the image here<br>or click to choose"}
    </div>
    <div class="text">
      <div class="id">#${sc.id} · ${sc.seconds}s · ${sc.words} words</div>
      <div>${esc(sc.text)}</div>
      ${sc.note ? `<div class="note">Image: ${esc(sc.note)}</div>` : ""}
      <div class="row" style="margin-top:10px">
        <button data-preview>▶ Preview</button>
        ${changed ? `<span class="changed">look changed</span><button class="link" data-reset>reset</button>` : ""}
      </div>
    </div>
    <div class="looks">
      <label>Look</label>${lookSelect(sc, "preset", presetAuto)}
      <label>Camera</label>${lookSelect(sc, "move", `Auto: ${sc.spec.move.replace(/_/g, " ")}`)}
      <label>Particles</label>${lookSelect(sc, "atmosphere", fromLook("atmosphere"))}
      <label>Mist</label>${lookSelect(sc, "mist", fromLook("mist"))}
      <label>Light</label>${lookSelect(sc, "glow", fromLook("glow"))}
      <label>Strength</label><input type="number" data-key="strength" min="0" max="3" step="0.1"
        value="${sc.override.strength ?? ""}" placeholder="${sc.spec.strength}">
    </div>`;
}

function bindScene(row) {
  const id = row.dataset.id;
  const media = $("[data-media]", row);
  media.addEventListener("click", async () => {
    if (media.querySelector("video")) return;
    const file = await pickFile("image/png,image/jpeg,image/webp");
    if (file) uploadImage(id, file, row);
  });
  row.addEventListener("dragover", (ev) => { ev.preventDefault(); row.classList.add("dragover"); });
  row.addEventListener("dragleave", () => row.classList.remove("dragover"));
  row.addEventListener("drop", (ev) => {
    ev.preventDefault();
    row.classList.remove("dragover");
    const file = ev.dataTransfer.files[0];
    if (file) uploadImage(id, file, row);
  });
  row.querySelectorAll("[data-key]").forEach((el) => el.addEventListener("change", () => saveLook(id, row)));
  $("[data-preview]", row).addEventListener("click", () => preview(id, row));
  $("[data-reset]", row)?.addEventListener("click", async () => {
    await putLook(id, row, {});
  });
}

function sceneById(id) {
  return STATE.scenes.find((s) => s.id === id);
}

function refreshRow(id, row) {
  row.innerHTML = sceneInner(sceneById(id));
  bindScene(row);
}

async function saveLook(id, row) {
  const values = {};
  row.querySelectorAll("[data-key]").forEach((el) => { if (el.value !== "") values[el.dataset.key] = el.value; });
  if (values.strength !== undefined) values.strength = Number(values.strength);
  await putLook(id, row, values);
}

async function putLook(id, row, values) {
  try {
    STATE = await api("PUT", `/api/episodes/${STATE.name}/scenes/${id}`, values);
    refreshRow(id, row);
  } catch (e) {
    toast(e.message, true);
  }
}

async function uploadImage(id, file, row) {
  const media = $("[data-media]", row);
  media.insertAdjacentHTML("beforeend", `<div class="busy">Uploading…</div>`);
  const fd = new FormData();
  fd.append("file", file);
  try {
    STATE = await api("POST", `/api/episodes/${STATE.name}/scenes/${id}/image`, fd);
    refreshRow(id, row);
    renderBuild();
  } catch (e) {
    toast(e.message, true);
    refreshRow(id, row);
  }
}

async function preview(id, row) {
  const media = $("[data-media]", row);
  const btn = $("[data-preview]", row);
  btn.disabled = true;
  media.insertAdjacentHTML("beforeend", `<div class="busy">Rendering preview…</div>`);
  try {
    const r = await api("POST", `/api/episodes/${STATE.name}/scenes/${id}/preview`);
    media.classList.remove("empty");
    media.innerHTML = `<video src="${esc(r.url)}" autoplay loop muted playsinline controls></video>`;
  } catch (e) {
    toast(e.message, true);
    $(".busy", media)?.remove();
  } finally {
    btn.disabled = false;
  }
}

// ---- script tab ------------------------------------------------------------------------------------------

function renderScript() {
  const tab = $("#tab");
  tab.innerHTML = `
    <div class="script stack">
      <div class="row spread">
        <div class="muted small">One paragraph = one image · <code>## Heading</code> = chapter ·
          <code>&gt; Image: …</code> = what to draw (not narrated) · Ctrl+S saves</div>
        <button class="primary" id="savescript">Save script</button>
      </div>
      <textarea id="scripttext" spellcheck="true">${esc(STATE.script)}</textarea>
    </div>`;
  const ta = $("#scripttext");
  const save = async () => {
    try {
      STATE = await api("PUT", `/api/episodes/${STATE.name}/script`, { text: ta.value });
      toast(`Saved · ${STATE.summary.scenes} scenes, about ${STATE.summary.est_minutes} min`);
      const pos = ta.selectionStart;
      renderEpisode();
      const again = $("#scripttext");
      again.focus();
      again.setSelectionRange(pos, pos);
    } catch (e) {
      toast(e.message, true);
    }
  };
  $("#savescript").addEventListener("click", save);
  ta.addEventListener("keydown", (ev) => {
    if ((ev.ctrlKey || ev.metaKey) && ev.key === "s") { ev.preventDefault(); save(); }
  });
}

// ---- settings tab ----------------------------------------------------------------------------------------

const SETTING_FIELDS = [
  ["default_preset", "Default look", "select:presets", "Used for scenes whose words don't suggest a look."],
  ["auto_look", "Pick looks from words", "bool", "fire → campfire, fog → misty, snow → winter …"],
  ["size", "Final video size", "select:1920x1080,2560x1440,3840x2160,1280x720", "Larger sizes take longer to render."],
  ["fps", "Frames per second", "select:24,25,30", ""],
  ["transition", "Crossfade (seconds)", "number:0:2:0.1", "0 = hard cuts between scenes."],
  ["enhance_voice", "Clean up my voice", "bool", "Turn off if you already cleaned the recording in another tool."],
  ["music_volume", "Music volume", "number:0:1:0.01", "Level under your voice; it also dips while you speak."],
  ["loudness", "Final loudness (LUFS)", "number:-24:-9:0.5", "YouTube plays at -14."],
  ["end_hold", "Hold last image (seconds)", "number:0:10:0.5", "After your last word."],
  ["image_style", "Image style for prompts", "text", "Added to every prompt in shotlist.md, e.g. “flat 2D storybook illustration”."],
  ["whisper_model", "Speech recognition", "select:base,small", "base is fast and usually enough; small is slower."],
  ["language", "Narration language", "text", "Two-letter code, e.g. en."],
];

function renderSettings() {
  const st = STATE.settings;
  const field = ([key, label, type, help]) => {
    let input;
    if (type === "bool") input = `<input type="checkbox" name="${key}" ${st[key] ? "checked" : ""} style="justify-self:start">`;
    else if (type.startsWith("select:")) {
      const list = type === "select:presets" ? OPTIONS.looks.preset : type.slice(7).split(",");
      input = `<select name="${key}">${list.map((v) => option(v, v, String(st[key]) === v)).join("")}</select>`;
    } else if (type.startsWith("number:")) {
      const [, min, max, step] = type.split(":");
      input = `<input type="number" name="${key}" min="${min}" max="${max}" step="${step}" value="${esc(st[key])}">`;
    } else input = `<input type="text" name="${key}" value="${esc(st[key])}">`;
    return `<label for="${key}">${label}</label>${input}${help ? `<div class="help">${esc(help)}</div>` : ""}`;
  };
  $("#tab").innerHTML = `
    <form class="panel" id="settings">
      <div class="form">${SETTING_FIELDS.map(field).join("")}</div>
      <div style="margin-top:16px"><button class="primary">Save settings</button></div>
    </form>`;
  $("#settings").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const f = ev.target;
    const values = {};
    for (const [key, , type] of SETTING_FIELDS) {
      const el = f.elements.namedItem(key);
      values[key] = type === "bool" ? el.checked : el.value;
    }
    try {
      STATE = await api("PUT", `/api/episodes/${STATE.name}/settings`, values);
      toast("Settings saved");
      renderEpisode();
    } catch (e) {
      toast(e.message, true);
    }
  });
}

// ---- output tab ------------------------------------------------------------------------------------------

function renderOutput() {
  const ex = STATE.exports;
  const videos = ex.filter((e) => e.name.endsWith(".mp4"));
  const latest = videos[0];
  $("#tab").innerHTML = !ex.length ? `<p class="muted">Nothing built yet. Use “Build draft” above.</p>` : `
    <div class="stack">
      ${latest ? `<div><div class="muted small">${esc(latest.name)} · built ${new Date(latest.mtime * 1000).toLocaleString()}</div>
        <video class="final" controls preload="metadata" src="${esc(latest.url)}"></video></div>` : ""}
      <div class="panel">
        <h2>Files</h2>
        <ul class="files">${ex.map((e) => `<li><a href="${esc(e.url)}" download="${esc(e.name)}">${esc(e.name)}</a>
          <span class="muted small">${fmtSize(e.size)}</span></li>`).join("")}</ul>
        <div class="muted small">Upload subtitles.srt to YouTube as captions. Paste the chapters below into the description.</div>
      </div>
      ${STATE.chapters ? `<div class="panel"><div class="row spread"><h2>Chapters</h2><button id="copych">Copy</button></div>
        <pre class="chapters">${esc(STATE.chapters)}</pre></div>` : ""}
    </div>`;
  $("#copych")?.addEventListener("click", async () => {
    await navigator.clipboard.writeText(STATE.chapters);
    toast("Chapters copied");
  });
}

route();
