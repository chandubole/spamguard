/* =========================================================
   Smart Spam Detector - frontend logic
   Predictions ALWAYS come from the Flask backend.
   ========================================================= */

// ---- Configuration: change the backend URL here only ----
const API_URL = "/predict";
const REQUEST_TIMEOUT_MS = 15000;
const MAX_HISTORY = 10;
const PREVIEW_LENGTH = 40; // privacy: only this many characters are stored

// ---- Model results: copy these from your notebook's test-set evaluation ----
// Use 0-1 (e.g. 0.978) or 0-100 (e.g. 97.8). Leave null to show a dash.
const MODEL_INFO = {
  accuracy: null,
  precision: null,
  recall: null,
  f1: null,
  dataset: "",   // e.g. "SMS Spam Collection (5,572 messages), 80/20 train-test split"
};

// ---- Example messages for the demo buttons (inputs only, not predictions) ----
const SAMPLES = {
  spam: "URGENT! You have won a FREE prize of Rs 50,000. Click here now to claim your reward before it expires: http://bit.ly/claim-now",
  ham: "Hey, I will be 10 minutes late. Can you save me a seat and send me yesterday's lecture notes?",
};
const KEYS = { stats: "ssd_stats", history: "ssd_history" };

// ---- Element shortcuts ----
const $ = (id) => document.getElementById(id);
const el = {
  form: $("form"), message: $("message"), count: $("count"), error: $("error"),
  detectBtn: $("detectBtn"), detectLabel: $("detectLabel"), clearBtn: $("clearBtn"),
  resultCard: $("resultCard"), resultIcon: $("resultIcon"), resultTitle: $("resultTitle"),
  resultText: $("resultText"), confidence: $("confidence"),
  triggerCard: $("triggerCard"), chips: $("chips"), noWords: $("noWords"), highlighted: $("highlighted"),
  statTotal: $("statTotal"), statSpam: $("statSpam"), statHam: $("statHam"),
  historyList: $("historyList"), historyEmpty: $("historyEmpty"), clearHistory: $("clearHistory"),
  menuBtn: $("menuBtn"), navMenu: $("navMenu"),
};

let isLoading = false;

/* ---------- localStorage helpers (safe if storage is blocked) ---------- */
function load(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
}
function save(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* ignore */ }
}
let stats = load(KEYS.stats, { total: 0, spam: 0, ham: 0 });
let history = load(KEYS.history, []);

/* ---------- Backend request ---------- */
async function requestPrediction(message) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let response;
  try {
    response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
      signal: controller.signal,
    });
  } catch {
    throw new Error("Unable to connect to the prediction server. Please try again.");
  } finally {
    clearTimeout(timer);
  }

  if (!response.ok) {
    throw new Error(`The server returned an error (HTTP ${response.status}). Please try again.`);
  }
  let data;
  try { data = await response.json(); }
  catch { throw new Error("The server sent a response that could not be read."); }
  return normalizeResponse(data);
}

// Check the backend JSON and turn it into a predictable object
function normalizeResponse(data) {
  const label = String(data?.prediction ?? "").toLowerCase().trim();
  let isSpam;
  if (label === "spam") isSpam = true;
  else if (["ham", "not spam", "not_spam", "legitimate"].includes(label)) isSpam = false;
  else throw new Error("The server returned an unexpected prediction value.");

  const words = Array.isArray(data.trigger_words)
    ? data.trigger_words.filter((w) => typeof w === "string" && w.trim()).map((w) => w.trim())
    : [];

  // Confidence is optional: use it only if it is a valid number
  let confidence = null;
  const c = Number(data.confidence);
  if (data.confidence !== undefined && data.confidence !== null && Number.isFinite(c) && c >= 0 && c <= 100) {
    confidence = c <= 1 ? c * 100 : c;
  }
  return { isSpam, words, confidence };
}

/* ---------- UI helpers ---------- */
function setLoading(on) {
  isLoading = on;
  el.detectBtn.disabled = on;
  el.detectBtn.classList.toggle("is-loading", on);
  el.detectLabel.textContent = on ? "Analyzing..." : "Detect Spam";
  $("scanWrap").classList.toggle("scanning", on);   // scan beam over the text box
  el.message.readOnly = on;
}
function showError(text) {
  el.error.textContent = text;
  el.error.hidden = false;
  el.message.classList.add("invalid");
}
function hideError() {
  el.error.hidden = true;
  el.message.classList.remove("invalid");
}
function hideResults() {
  el.resultCard.hidden = true;
  el.triggerCard.hidden = true;
}

const ICON_WARN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.300 3.900L1.800 18a2 2 0 001.700 3h17a2 2 0 001.700-3L13.700 3.900a2 2 0 00-3.400 0z"/><path d="M12 9v4M12 17h.01"/></svg>';
const ICON_OK = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M8 12.500l2.800 2.800L16 9.500"/></svg>';

function renderResult(result, message) {
  const { isSpam, words, confidence } = result;
  el.resultCard.className = "card result " + (isSpam ? "result--spam" : "result--ham");
  el.resultIcon.innerHTML = isSpam ? ICON_WARN : ICON_OK;
  el.resultTitle.textContent = isSpam ? "SPAM DETECTED" : "NOT SPAM";
  el.resultText.textContent = isSpam
    ? "The model classified this message as spam. Be careful with links, prizes or requests for personal details."
    : "The model classified this message as legitimate. Stay careful with any message you did not expect.";
  if (confidence !== null) {
    el.confidence.textContent = `Model confidence: ${confidence.toFixed(1)}%`;
    el.confidence.hidden = false;
  } else {
    el.confidence.hidden = true;
  }
  el.resultCard.hidden = false;

  // Trigger words: only what the backend returned
  el.chips.replaceChildren(...words.map((w) => {
    const li = document.createElement("li");
    li.textContent = w;
    return li;
  }));
  el.noWords.hidden = words.length > 0;
  renderHighlighted(message, words);
  el.triggerCard.hidden = false;
}

// Build the message with <mark> around backend trigger words (safe: no innerHTML)
function renderHighlighted(message, words) {
  el.highlighted.replaceChildren();
  if (!words.length) { el.highlighted.textContent = message; return; }
  const escaped = words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp(`(${escaped.join("|")})`, "gi");
  message.split(re).forEach((part, i) => {
    if (i % 2 === 1) {
      const mark = document.createElement("mark");
      mark.textContent = part;
      el.highlighted.appendChild(mark);
    } else if (part) {
      el.highlighted.appendChild(document.createTextNode(part));
    }
  });
}

/* ---------- Stats and history ---------- */
// Count a number up (or down) smoothly
function animateNumber(node, to, decimals = 0, suffix = "") {
  const from = Number(node.dataset.v ?? 0);
  node.dataset.v = to;
  if (matchMedia("(prefers-reduced-motion: reduce)").matches || from === to) {
    node.textContent = to.toFixed(decimals) + suffix; return;
  }
  const t0 = performance.now(), dur = 900;
  (function step(t) {
    const p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 3);
    node.textContent = (from + (to - from) * e).toFixed(decimals) + suffix;
    if (p < 1) requestAnimationFrame(step);
  })(t0);
}
function renderStats() {
  animateNumber(el.statTotal, stats.total);
  animateNumber(el.statSpam, stats.spam);
  animateNumber(el.statHam, stats.ham);
}

/* ---------- Model performance section ---------- */
function renderModel() {
  document.querySelectorAll(".metric").forEach((node) => {
    let v = MODEL_INFO[node.dataset.key];
    if (typeof v !== "number" || !Number.isFinite(v)) return;   // keep the dash
    if (v <= 1) v *= 100;
    animateNumber(node, v, 1, "%");
    node.closest(".metric-card").querySelector(".bar i").style.width = Math.min(100, v) + "%";
  });
  if (MODEL_INFO.dataset) $("datasetNote").textContent = "Dataset: " + MODEL_INFO.dataset;
}
function renderHistory() {
  el.historyList.replaceChildren(...history.map((h) => {
    const li = document.createElement("li");
    const tag = document.createElement("span");
    tag.className = "tag " + (h.isSpam ? "tag-spam" : "tag-ham");
    tag.textContent = h.isSpam ? "SPAM" : "NOT SPAM";
    const preview = document.createElement("span");
    preview.className = "preview";
    preview.textContent = h.preview;
    const time = document.createElement("time");
    time.dateTime = h.time;
    time.textContent = new Date(h.time).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
    li.append(tag, preview, time);
    return li;
  }));
  el.historyEmpty.hidden = history.length > 0;
}
function recordAnalysis(isSpam, message) {
  stats.total++;
  isSpam ? stats.spam++ : stats.ham++;
  const preview = message.length > PREVIEW_LENGTH ? message.slice(0, PREVIEW_LENGTH) + "..." : message;
  history.unshift({ isSpam, preview, time: new Date().toISOString() });
  history = history.slice(0, MAX_HISTORY);
  save(KEYS.stats, stats);
  save(KEYS.history, history);
  renderStats();
  renderHistory();
}

/* ---------- Event handlers ---------- */
el.message.addEventListener("input", () => {
  el.count.textContent = `${el.message.value.length} / ${el.message.maxLength} characters`;
  if (el.message.value.trim()) hideError();
});

el.clearBtn.addEventListener("click", () => {
  el.message.value = "";
  el.count.textContent = `0 / ${el.message.maxLength} characters`;
  hideError();
  hideResults();
  el.message.focus();
});

el.form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (isLoading) return; // block repeated submissions

  const message = el.message.value.trim();
  if (!message) {
    showError("Please enter a message before checking.");
    el.message.focus();
    return;
  }
  hideError();
  hideResults();
  setLoading(true);
  try {
    const result = await requestPrediction(message);
    renderResult(result, message);
    recordAnalysis(result.isSpam, message);
    el.resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
});

el.clearHistory.addEventListener("click", () => {
  history = [];
  stats = { total: 0, spam: 0, ham: 0 };
  save(KEYS.history, history);
  save(KEYS.stats, stats);
  renderStats();
  renderHistory();
});

/* ---------- Navigation: mobile menu + active link ---------- */
function setMenu(open) {
  el.navMenu.classList.toggle("open", open);
  el.menuBtn.setAttribute("aria-expanded", String(open));
}
el.menuBtn.addEventListener("click", () => setMenu(!el.navMenu.classList.contains("open")));
el.navMenu.querySelectorAll("a").forEach((a) => a.addEventListener("click", () => setMenu(false)));
document.addEventListener("keydown", (e) => { if (e.key === "Escape") setMenu(false); });

const links = [...document.querySelectorAll(".nav-link")];
const observer = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      links.forEach((l) => l.classList.toggle("active", l.getAttribute("href") === "#" + entry.target.id));
    }
  });
}, { rootMargin: "-45% 0px -50% 0px" });
["home", "how", "model", "about"].forEach((id) => observer.observe($(id)));

/* ---------- Init ---------- */
$("year").textContent = new Date().getFullYear();
renderStats();
renderHistory();

/* ---------- Visual effects: particle background + hero parallax ---------- */
(() => {
  const canvas = $("bg"), ctx = canvas.getContext("2d");
  const mouse = { x: -999, y: -999 };
  let w, h, pts = [];
  function resize() {
    w = canvas.width = innerWidth; h = canvas.height = innerHeight;
    pts = Array.from({ length: Math.min(70, Math.floor(w / 18)) }, () => ({
      x: Math.random() * w, y: Math.random() * h,
      vx: (Math.random() - .5) * .35, vy: (Math.random() - .5) * .35, r: Math.random() * 1.8 + .6,
    }));
  }
  addEventListener("resize", resize); resize();
  addEventListener("mousemove", (e) => { mouse.x = e.clientX; mouse.y = e.clientY; });
  const art = $("art");
  addEventListener("mousemove", (e) => {
    art.style.transform = `translate(${(e.clientX / innerWidth - .5) * 24}px, ${(e.clientY / innerHeight - .5) * 24}px)`;
  });
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  (function draw() {
    ctx.clearRect(0, 0, w, h);
    pts.forEach((p, i) => {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > w) p.vx *= -1;
      if (p.y < 0 || p.y > h) p.vy *= -1;
      ctx.fillStyle = "rgba(255,120,200,.7)";
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 7); ctx.fill();
      for (let j = i + 1; j < pts.length; j++) {
        const q = pts[j], d = Math.hypot(p.x - q.x, p.y - q.y);
        if (d < 120) { ctx.strokeStyle = `rgba(155,60,240,${.25 * (1 - d / 120)})`; ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke(); }
      }
      const dm = Math.hypot(p.x - mouse.x, p.y - mouse.y);
      if (dm < 150) { ctx.strokeStyle = `rgba(255,47,168,${.5 * (1 - dm / 150)})`; ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(mouse.x, mouse.y); ctx.stroke(); }
    });
    requestAnimationFrame(draw);
  })();
})();


/* ---------- Example buttons + Ctrl+Enter ---------- */
document.querySelectorAll("[data-sample]").forEach((btn) => {
  btn.addEventListener("click", () => {
    el.message.value = SAMPLES[btn.dataset.sample];
    el.message.dispatchEvent(new Event("input"));   // update counter, clear errors
    hideResults();
    el.message.focus();
  });
});
el.message.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); el.form.requestSubmit(); }
});

/* ---------- Scroll reveal + model metrics ---------- */
(() => {
  const targets = document.querySelectorAll(".section h2, .stat, .steps li, .tech .card, .metric-card, .history, #about p");
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!reduce && "IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((en) => { if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); } });
    }, { threshold: .15 });
    targets.forEach((t, i) => { t.classList.add("reveal"); t.style.transitionDelay = (i % 4) * 80 + "ms"; io.observe(t); });
    // run the metric count-up when its section scrolls into view
    const mo = new IntersectionObserver((en) => { if (en[0].isIntersecting) { renderModel(); mo.disconnect(); } }, { threshold: .3 });
    mo.observe($("model"));
  } else {
    renderModel();
  }
})();
