/*
 * NetraChat browser code.
 *
 * What happens:
 *   1. When the page opens, we ask the backend for a new chat "session".
 *   2. When you choose an image, we upload it to /api/image. The backend
 *      returns the opening message and the list of detected objects, which
 *      we draw as boxes on top of the picture.
 *   3. Every question you type is sent to /api/chat and the reply is shown.
 *
 * Messages are always inserted with textContent (never innerHTML), so text
 * typed by a user can never inject code into the page.
 */

const PALETTE = ["#e6194b", "#2f9e44", "#3b5bdb", "#f08c00", "#9c36b5",
                 "#0c8599", "#d6336c", "#5c940d", "#1971c2", "#a35a1f"];

const GENERAL_SUGGESTIONS = [
  "Describe the image", "What objects can you see?", "How many people are there?",
  "Where was this photo taken?", "What is the main object?", "How sure are you?",
];

const $ = (id) => document.getElementById(id);
const els = {
  status: $("status"), statusText: $("status-text"),
  dropzone: $("dropzone"), fileInput: $("file-input"),
  stage: $("stage"), preview: $("preview"), boxes: $("boxes"), scanning: $("scanning"),
  objects: $("objects"), changeImage: $("change-image"),
  boxToggleWrap: $("box-toggle-wrap"), boxToggle: $("box-toggle"),
  messages: $("messages"), suggestions: $("suggestions"),
  composer: $("composer"), input: $("message"), send: $("send"), reset: $("reset"),
};

const state = { sessionId: null, detections: [], colors: {}, busy: false, imageUrl: null };

// ------------------------------------------------------------ backend calls
async function api(path, options = {}) {
  const res = await fetch(`/api/${path}`, options);
  let data = {};
  try { data = await res.json(); } catch { /* empty or non-JSON body */ }
  if (!res.ok) {
    const err = new Error(data.detail || `Request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return data;
}

const postJSON = (path, body) => api(path, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

async function startSession() {
  const { session_id } = await api("session", { method: "POST" });
  state.sessionId = session_id;
}

async function checkHealth() {
  try {
    const h = await api("health");
    els.status.className = "status ok";
    els.statusText.textContent = h.blip ? "Online · BLIP on" : "Online";
    els.status.title = h.blip
      ? "Object detection + BLIP question answering"
      : "Object detection only. Start the backend with --blip for open questions.";
    return true;
  } catch (e) {
    els.status.className = "status bad";
    els.statusText.textContent = "Backend offline";
    els.status.title = e.message;
    return false;
  }
}

// ------------------------------------------------------------ chat bubbles
function addMessage(role, text, { image } = {}) {
  const row = document.createElement("div");
  row.className = `msg ${role}`;
  if (role === "bot" || role === "user") {
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "bot" ? "N" : "You";
    row.appendChild(avatar);
  }
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (image) {
    const img = document.createElement("img");
    img.className = "thumb";
    img.src = image;
    img.alt = "Uploaded image";
    bubble.appendChild(img);
  } else {
    bubble.textContent = text;
  }
  row.appendChild(bubble);
  els.messages.appendChild(row);
  els.messages.scrollTop = els.messages.scrollHeight;
  return row;
}

function showTyping() {
  const row = document.createElement("div");
  row.className = "msg bot typing";
  row.innerHTML = '<div class="avatar">N</div><div class="bubble"><i></i><i></i><i></i></div>';
  els.messages.appendChild(row);
  els.messages.scrollTop = els.messages.scrollHeight;
  return row;
}

function setBusy(busy) {
  state.busy = busy;
  els.send.disabled = busy;
  els.suggestions.querySelectorAll("button").forEach((b) => { b.disabled = busy; });
}

// ------------------------------------------------------------ suggestions
function renderSuggestions() {
  els.suggestions.replaceChildren();
  let items;
  if (!state.detections.length && !state.imageUrl) {
    items = ["What can you do?", "Who are you?"];
  } else {
    const labels = [...new Set(state.detections.map((d) => d.label))];
    items = [...GENERAL_SUGGESTIONS];
    labels.slice(0, 3).reverse().forEach((l) => {
      items.splice(1, 0, `What colour is the ${l}?`, `Where is the ${l}?`);
    });
    items = [...new Set(items)].slice(0, 9);
  }
  for (const text of items) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = text;
    b.addEventListener("click", () => send(text));
    els.suggestions.appendChild(b);
  }
}

// ------------------------------------------------------------ boxes + list
function colorFor(label) {
  if (!state.colors[label]) {
    state.colors[label] = PALETTE[Object.keys(state.colors).length % PALETTE.length];
  }
  return state.colors[label];
}

function renderDetections(width, height) {
  els.boxes.replaceChildren();
  els.objects.replaceChildren();
  state.colors = {};

  for (const d of state.detections) {
    const [x1, y1, x2, y2] = d.box;
    const box = document.createElement("div");
    box.className = "box";
    box.dataset.label = d.label;
    box.style.setProperty("--c", colorFor(d.label));
    box.style.left = `${(x1 / width) * 100}%`;
    box.style.top = `${(y1 / height) * 100}%`;
    box.style.width = `${((x2 - x1) / width) * 100}%`;
    box.style.height = `${((y2 - y1) / height) * 100}%`;
    if (y1 / height < 0.05) box.classList.add("inside");
    if (x1 / width > 0.75) box.classList.add("right"); // keep the label on the image
    const tag = document.createElement("span");
    tag.textContent = `${d.label} ${Math.round(d.confidence * 100)}%`;
    box.appendChild(tag);
    els.boxes.appendChild(box);
  }

  // One chip per kind of object: "3 × person · 89%"
  const groups = new Map();
  for (const d of state.detections) {
    const g = groups.get(d.label) || { n: 0, best: 0 };
    g.n += 1;
    g.best = Math.max(g.best, d.confidence);
    groups.set(d.label, g);
  }
  if (!groups.size) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No familiar objects found. You can still ask questions.";
    els.objects.appendChild(li);
  }
  for (const [label, g] of groups) {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button";
    b.style.setProperty("--c", colorFor(label));
    b.title = `Ask about the ${label}`;
    const sw = document.createElement("span"); sw.className = "swatch";
    const cnt = document.createElement("span"); cnt.className = "count";
    cnt.textContent = g.n > 1 ? `${g.n} ×` : "";
    const name = document.createElement("span"); name.textContent = label;
    const conf = document.createElement("span"); conf.className = "conf";
    conf.textContent = `${Math.round(g.best * 100)}%`;
    b.append(sw, cnt, name, conf);
    b.addEventListener("mouseenter", () => highlight([label]));
    b.addEventListener("mouseleave", () => highlight([]));
    b.addEventListener("focus", () => highlight([label]));
    b.addEventListener("blur", () => highlight([]));
    b.addEventListener("click", () => send(`Tell me about the ${label}`));
    li.appendChild(b);
    els.objects.appendChild(li);
  }
  els.boxToggleWrap.hidden = !state.detections.length;
}

// Dim every box except the ones being talked about.
let highlightTimer = null;
function highlight(labels, { linger = 0 } = {}) {
  clearTimeout(highlightTimer);
  const set = new Set(labels);
  els.boxes.classList.toggle("focusing", set.size > 0);
  els.boxes.querySelectorAll(".box").forEach((b) => {
    b.classList.toggle("hot", set.has(b.dataset.label));
  });
  if (linger && set.size) highlightTimer = setTimeout(() => highlight([]), linger);
}

// ------------------------------------------------------------ image upload
async function uploadImage(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    addMessage("error", "Please choose an image file (JPG or PNG).");
    return;
  }
  if (state.busy) return;
  setBusy(true);

  if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
  state.imageUrl = URL.createObjectURL(file);
  state.detections = [];
  els.preview.src = state.imageUrl;
  els.boxes.replaceChildren();
  els.objects.replaceChildren();
  els.dropzone.hidden = true;
  els.stage.hidden = false;
  els.scanning.hidden = false;
  addMessage("user", "", { image: state.imageUrl });
  const typing = showTyping();

  const form = new FormData();
  form.append("session_id", state.sessionId);
  form.append("file", file);
  try {
    const data = await withSession(() => api("image", { method: "POST", body: form }));
    state.detections = data.detections;
    renderDetections(data.width, data.height);
    typing.remove();
    addMessage("bot", data.message);
  } catch (e) {
    typing.remove();
    addMessage("error", e.message);
    els.stage.hidden = true;
    els.dropzone.hidden = false;
  } finally {
    els.scanning.hidden = true;
    setBusy(false);
    renderSuggestions();
    els.fileInput.value = "";
  }
}

// If the backend restarted, our session is gone: start a new one and retry
// once (for chat), or ask the user to upload again (for images).
async function withSession(call) {
  try {
    return await call();
  } catch (e) {
    if (e.status !== 404) throw e;
    await startSession();
    throw new Error("The backend was restarted, so I lost the image. Please upload it again.");
  }
}

// ------------------------------------------------------------ chatting
async function send(text) {
  text = (text || "").trim();
  if (!text || state.busy) return;
  els.input.value = "";
  addMessage("user", text);
  setBusy(true);
  const typing = showTyping();
  try {
    const data = await withSession(() =>
      postJSON("chat", { session_id: state.sessionId, message: text }));
    typing.remove();
    addMessage("bot", data.reply);
    if (data.focus && data.focus.length) highlight(data.focus, { linger: 4000 });
  } catch (e) {
    typing.remove();
    addMessage("error", e.message);
    if (e.message.includes("upload it again")) clearImage();
  } finally {
    setBusy(false);
    els.input.focus();
  }
}

function clearImage() {
  state.detections = [];
  els.stage.hidden = true;
  els.dropzone.hidden = false;
  els.boxToggleWrap.hidden = true;
  if (state.imageUrl) URL.revokeObjectURL(state.imageUrl);
  state.imageUrl = null;
  renderSuggestions();
}

async function resetAll() {
  if (state.busy) return;
  try {
    if (state.sessionId) await postJSON("reset", { session_id: state.sessionId });
  } catch {
    await startSession().catch(() => {});
  }
  els.messages.replaceChildren();
  clearImage();
  greet();
}

function greet() {
  addMessage("bot",
    "Hello! I am NetraChat. Upload an image on the left, then ask me anything " +
    "about it: what is in it, how many objects there are, their colours, and " +
    "where they are.");
}

// ------------------------------------------------------------ wiring
els.composer.addEventListener("submit", (e) => { e.preventDefault(); send(els.input.value); });
els.fileInput.addEventListener("change", () => uploadImage(els.fileInput.files[0]));
els.changeImage.addEventListener("click", () => els.fileInput.click());
els.reset.addEventListener("click", resetAll);
els.boxToggle.addEventListener("change", () => {
  els.boxes.classList.toggle("off", !els.boxToggle.checked);
});

["dragenter", "dragover"].forEach((ev) => document.addEventListener(ev, (e) => {
  e.preventDefault();
  els.dropzone.classList.add("drag");
}));
["dragleave", "drop"].forEach((ev) => document.addEventListener(ev, (e) => {
  e.preventDefault();
  if (ev === "drop" || e.target === els.dropzone) els.dropzone.classList.remove("drag");
}));
document.addEventListener("drop", (e) => uploadImage(e.dataTransfer?.files?.[0]));
document.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) uploadImage(item.getAsFile());
});

(async function init() {
  greet();
  renderSuggestions();
  setBusy(true);
  const online = await checkHealth();
  if (online) {
    try {
      await startSession();
      setBusy(false);
    } catch (e) {
      addMessage("error", e.message);
    }
  } else {
    addMessage("error",
      "I cannot reach the NetraChat backend. Start it in another terminal with: " +
      "python api.py — I will connect automatically once it is running.");
  }
  // Keep checking; if the backend comes online later, connect automatically.
  setInterval(async () => {
    if ((await checkHealth()) && !state.sessionId) {
      try {
        await startSession();
        setBusy(false);
        addMessage("note", "Connected to the backend. You can upload an image now.");
      } catch { /* try again on the next check */ }
    }
  }, 10000);
})();
