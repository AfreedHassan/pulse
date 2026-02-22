import "./style.css";
import { createState } from "./state";
import type { AppPanel, WritingMode, TranscriptEntry } from "./types";
import * as api from "./api";

const state = createState();
let timerInterval: ReturnType<typeof setInterval> | null = null;

const COPY_ICON_SVG = `<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>`;
const CHECK_ICON_SVG = `<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M20 6L9 17l-5-5"/></svg>`;

// DOM refs
const providerDropdown = document.getElementById("provider-dropdown") as HTMLDivElement;
const providerTrigger = document.getElementById("provider-trigger") as HTMLButtonElement;
const modelDropdown = document.getElementById("model-dropdown") as HTMLDivElement;
const modelTrigger = document.getElementById("model-trigger") as HTMLButtonElement;
const emptyState = document.getElementById("empty-state") as HTMLDivElement;
const historyContainer = document.getElementById("history-container") as HTMLDivElement;
const liveContainer = document.getElementById("live-container") as HTMLDivElement;
const liveText = document.getElementById("live-text") as HTMLParagraphElement;
const loadingOverlay = document.getElementById("loading-overlay") as HTMLDivElement;
const loadingText = document.getElementById("loading-text") as HTMLParagraphElement;
const levelBar = document.getElementById("level-bar") as HTMLDivElement;
const statusDot = document.getElementById("status-dot") as HTMLSpanElement;
const statusText = document.getElementById("status-text") as HTMLSpanElement;
const timerText = document.getElementById("timer-text") as HTMLSpanElement;
const recordBtn = document.getElementById("record-btn") as HTMLButtonElement;
const writingModeGrid = document.getElementById("writing-mode-grid") as HTMLDivElement;
const pasteModeCb = document.getElementById("paste-mode-cb") as HTMLInputElement;
const hotkeyCb = document.getElementById("hotkey-cb") as HTMLInputElement;
const llmUrlInput = document.getElementById("llm-url") as HTMLInputElement;
const llmKeyInput = document.getElementById("llm-key") as HTMLInputElement;
const llmModelInput = document.getElementById("llm-model") as HTMLInputElement;
const promoCard = document.getElementById("promo-card") as HTMLDivElement;
const statSessions = document.getElementById("stat-sessions") as HTMLSpanElement;
const statWords = document.getElementById("stat-words") as HTMLSpanElement;
const statTime = document.getElementById("stat-time") as HTMLSpanElement;

// Panels
const transcriptPanel = document.getElementById("transcript-panel") as HTMLDivElement;
const settingsPanel = document.getElementById("settings-panel") as HTMLDivElement;
const stylePanel = document.getElementById("style-panel") as HTMLDivElement;
const helpPanel = document.getElementById("help-panel") as HTMLDivElement;

const allPanels = [transcriptPanel, settingsPanel, stylePanel, helpPanel];
const panelMap: Record<string, HTMLDivElement> = {
  transcript: transcriptPanel,
  settings: settingsPanel,
  style: stylePanel,
  help: helpPanel,
};

// Sidebar nav
const navItems = document.querySelectorAll<HTMLButtonElement>(".nav-item[data-nav]");

function showPanel(panel: string) {
  state.activePanel = panel as AppPanel;
  allPanels.forEach((p) => p.classList.add("hidden"));
  const target = panelMap[panel];
  if (target) target.classList.remove("hidden");

  navItems.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.nav === panel);
  });
}

navItems.forEach((btn) => {
  btn.addEventListener("click", () => {
    const nav = btn.dataset.nav;
    if (nav) showPanel(nav);
  });
});

// Custom select dropdown logic
function setupCustomSelect(
  dropdown: HTMLDivElement,
  trigger: HTMLButtonElement,
  onSelect: (value: string, label: string) => void,
) {
  const menu = dropdown.querySelector(".custom-select-menu") as HTMLDivElement;

  trigger.addEventListener("click", (e) => {
    e.stopPropagation();
    // Close any other open dropdowns
    document.querySelectorAll(".custom-select.open").forEach((el) => {
      if (el !== dropdown) el.classList.remove("open");
    });
    dropdown.classList.toggle("open");
  });

  menu.addEventListener("click", (e) => {
    const option = (e.target as HTMLElement).closest(".custom-select-option") as HTMLElement | null;
    if (!option) return;
    const value = option.dataset.value!;
    const label = option.textContent!;

    menu.querySelectorAll(".custom-select-option").forEach((o) => o.classList.remove("selected"));
    option.classList.add("selected");
    trigger.querySelector(".custom-select-value")!.textContent = label;
    dropdown.classList.remove("open");
    onSelect(value, label);
  });
}

// Close dropdowns on outside click
document.addEventListener("click", () => {
  document.querySelectorAll(".custom-select.open").forEach((el) => el.classList.remove("open"));
});

// Provider / Model selection
setupCustomSelect(providerDropdown, providerTrigger, (value) => {
  state.provider = value as "Whisper" | "Moonshine" | "CoreML";
  modelDropdown.classList.toggle("hidden", state.provider !== "Whisper");
  triggerModelLoad();
});

setupCustomSelect(modelDropdown, modelTrigger, (value) => {
  state.whisperModel = value;
  triggerModelLoad();
});

function triggerModelLoad() {
  state.modelReady = false;
  recordBtn.disabled = true;
  api.loadModel(state.provider, state.whisperModel).catch((err: unknown) => {
    setStatus("error", `Failed to load model: ${err}`);
  });
}

// Writing mode buttons
writingModeGrid.addEventListener("click", (e) => {
  const target = (e.target as HTMLElement).closest("[data-mode]") as HTMLElement | null;
  if (!target) return;
  const mode = target.dataset.mode as WritingMode;
  state.writingMode = mode;
  writingModeGrid.querySelectorAll(".writing-mode-btn").forEach((btn) => {
    const isActive = (btn as HTMLElement).dataset.mode === mode;
    btn.classList.toggle("active", isActive);
  });
});

// Settings inputs
pasteModeCb.addEventListener("change", () => { state.pasteMode = pasteModeCb.checked; });
hotkeyCb.addEventListener("change", () => { state.hotkeyEnabled = hotkeyCb.checked; });
llmUrlInput.addEventListener("input", () => { state.llmBaseUrl = llmUrlInput.value; });
llmKeyInput.addEventListener("input", () => { state.llmApiKey = llmKeyInput.value; });
llmModelInput.addEventListener("input", () => { state.llmModel = llmModelInput.value; });

// Record button
recordBtn.addEventListener("click", () => {
  state.recording ? doStopRecording() : doStartRecording();
});

function doStartRecording() {
  api.startRecording({
    writing_mode: state.writingMode,
    paste_mode: state.pasteMode,
    llm_base_url: state.llmBaseUrl,
    llm_api_key: state.llmApiKey,
    llm_model: state.llmModel,
  }).catch((err: unknown) => {
    setStatus("error", `Recording failed: ${err}`);
  });
}

function doStopRecording() {
  api.stopRecording().catch((err: unknown) => {
    setStatus("error", `Stop failed: ${err}`);
  });
}

// Timer
function startTimer() {
  state.recordStart = Date.now();
  timerText.classList.remove("hidden");
  updateTimer();
  timerInterval = setInterval(updateTimer, 100);
}

function stopTimer() {
  if (timerInterval) {
    clearInterval(timerInterval);
    timerInterval = null;
  }
  timerText.classList.add("hidden");
  state.recordStart = null;
}

function updateTimer() {
  if (!state.recordStart) return;
  const elapsed = Date.now() - state.recordStart;
  const mins = Math.floor(elapsed / 60000);
  const secs = Math.floor((elapsed % 60000) / 1000);
  const tenths = Math.floor((elapsed % 1000) / 100);
  timerText.textContent = `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${tenths}`;
}

// Status helper
function setStatus(type: "idle" | "ready" | "recording" | "error", text: string) {
  statusText.textContent = text;
  state.statusText = text;
  statusDot.className = "status-dot";
  switch (type) {
    case "ready":
      statusDot.classList.add("ready");
      break;
    case "recording":
      statusDot.classList.add("recording");
      break;
    case "error":
      statusDot.classList.add("error");
      break;
  }
}

// Stats
function updateStats() {
  const totalSessions = state.history.length;
  let totalWords = 0;
  let totalDuration = 0;
  for (const entry of state.history) {
    totalWords += entry.text.split(/\s+/).filter(Boolean).length;
    totalDuration += entry.duration_secs;
  }
  statSessions.textContent = String(totalSessions);
  statWords.textContent = String(totalWords);
  statTime.textContent = totalDuration < 60
    ? `${Math.round(totalDuration)}s`
    : `${Math.round(totalDuration / 60)}m`;
}

function getDateLabel(): string {
  const today = new Date()
    .toLocaleDateString("en-US", { month: "long", day: "numeric", year: "numeric" })
    .toUpperCase();
  return `TODAY — ${today}`;
}

function renderHistory() {
  historyContainer.innerHTML = "";

  if (state.history.length === 0) {
    emptyState.classList.remove("hidden");
    historyContainer.classList.add("hidden");
    return;
  }

  emptyState.classList.add("hidden");
  historyContainer.classList.remove("hidden");
  promoCard.classList.add("hidden");

  // Group by date label
  const groups = new Map<string, TranscriptEntry[]>();
  for (const entry of state.history) {
    const label = getDateLabel();
    if (!groups.has(label)) groups.set(label, []);
    groups.get(label)!.push(entry);
  }

  for (const [dateLabel, entries] of groups) {
    const groupEl = document.createElement("div");
    groupEl.className = "history-date-group";

    const header = document.createElement("div");
    header.className = "history-date-header";
    header.textContent = dateLabel;
    groupEl.appendChild(header);

    for (const entry of entries) {
      const row = document.createElement("div");
      row.className = "history-entry";

      const timeEl = document.createElement("span");
      timeEl.className = "history-entry-time";
      timeEl.textContent = entry.timestamp;

      const textEl = document.createElement("p");
      textEl.className = "history-entry-text";
      textEl.textContent = entry.text;

      const durationEl = document.createElement("span");
      durationEl.className = "history-entry-duration";
      durationEl.textContent = `${entry.duration_secs.toFixed(1)}s`;

      const actionsEl = document.createElement("div");
      actionsEl.className = "history-entry-actions";

      const copyBtn = document.createElement("button");
      copyBtn.className = "copy-btn";
      copyBtn.title = "Copy";
      copyBtn.innerHTML = COPY_ICON_SVG;
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(entry.text);
        copyBtn.innerHTML = CHECK_ICON_SVG;
        setTimeout(() => {
          copyBtn.innerHTML = COPY_ICON_SVG;
        }, 1500);
      });

      actionsEl.appendChild(copyBtn);
      row.appendChild(timeEl);
      row.appendChild(textEl);
      row.appendChild(durationEl);
      row.appendChild(actionsEl);
      groupEl.appendChild(row);
    }

    historyContainer.appendChild(groupEl);
  }

  historyContainer.scrollTop = historyContainer.scrollHeight;
}

function enterRecordingUI() {
  state.recording = true;
  recordBtn.textContent = "Stop";
  recordBtn.classList.add("is-recording");
  liveContainer.classList.remove("hidden");
  liveText.textContent = "";
  setStatus("recording", "Recording...");
  startTimer();
}

function exitRecordingUI() {
  state.recording = false;
  recordBtn.textContent = "Record";
  recordBtn.classList.remove("is-recording");
  liveContainer.classList.add("hidden");
  levelBar.style.width = "0%";
  setStatus("ready", "Ready");
  stopTimer();
}

// Event subscriptions — store unlisten handles for cleanup
const unlisteners: Array<() => void> = [];

async function setupEvents() {
  unlisteners.push(await api.onModelLoading((msg) => {
    loadingOverlay.classList.remove("hidden");
    loadingText.textContent = msg;
    setStatus("idle", msg);
  }));

  unlisteners.push(await api.onModelReady(() => {
    loadingOverlay.classList.add("hidden");
    state.modelReady = true;
    recordBtn.disabled = false;
    setStatus("ready", "Ready");
  }));

  unlisteners.push(await api.onRecordingStarted(enterRecordingUI));

  unlisteners.push(await api.onAudioLevel((level) => {
    state.audioLevel = level;
    levelBar.style.width = `${Math.min(level * 100, 100)}%`;
  }));

  unlisteners.push(await api.onChunk((text) => {
    state.liveText += text;
    liveText.textContent = state.liveText;
  }));

  unlisteners.push(await api.onDone((data) => {
    const entry: TranscriptEntry = {
      text: data.text,
      timestamp: new Date().toLocaleTimeString(),
      duration_secs: data.duration_secs,
    };
    state.history.push(entry);
    renderHistory();
    updateStats();
    state.liveText = "";
    exitRecordingUI();
  }));

  unlisteners.push(await api.onError((msg) => {
    setStatus("error", msg);
    if (state.recording) {
      exitRecordingUI();
    }
  }));

  unlisteners.push(await api.onHotkeyStart(() => {
    if (state.modelReady && !state.recording) {
      doStartRecording();
    }
  }));

  unlisteners.push(await api.onHotkeyStop(() => {
    if (state.recording) {
      doStopRecording();
    }
  }));
}

window.addEventListener("beforeunload", () => {
  unlisteners.forEach((fn) => fn());
});

// Init
async function init() {
  try {
    const defaults = await api.getDefaultSettings();
    state.llmBaseUrl = defaults.llm_base_url;
    state.llmModel = defaults.llm_model;
    state.pasteMode = defaults.paste_mode;
    state.hotkeyEnabled = defaults.hotkey_enabled;

    llmUrlInput.value = defaults.llm_base_url;
    llmModelInput.value = defaults.llm_model;
    pasteModeCb.checked = defaults.paste_mode;
    hotkeyCb.checked = defaults.hotkey_enabled;
    if (defaults.llm_api_key_set) {
      llmKeyInput.placeholder = "API Key (set via environment)";
    }
  } catch {
    // Backend not available yet — use defaults
  }

  // Sync model dropdown visibility with initial provider state.
  modelDropdown.classList.toggle("hidden", state.provider !== "Whisper");

  updateStats();
  await setupEvents();
}

init();
