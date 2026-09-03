import { createPlayground } from "https://cdn.jsdelivr.net/npm/livecodes@0.14.1/+esm";
import {
  AutoProcessor,
  env,
  Gemma4ForConditionalGeneration,
  TextStreamer,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/+esm";
import * as prettier from "https://cdn.jsdelivr.net/npm/prettier@3.5.3/standalone.mjs";
import * as prettierBabel from "https://cdn.jsdelivr.net/npm/prettier@3.5.3/plugins/babel.mjs";
import * as prettierEstree from "https://cdn.jsdelivr.net/npm/prettier@3.5.3/plugins/estree.mjs";
import * as prettierHtml from "https://cdn.jsdelivr.net/npm/prettier@3.5.3/plugins/html.mjs";
import * as prettierPostcss from "https://cdn.jsdelivr.net/npm/prettier@3.5.3/plugins/postcss.mjs";

const ui = {
  status: document.getElementById("status"),
  modelLoad: document.getElementById("model-load"),
  modelLoadFill: document.getElementById("model-load-fill"),
  modelLoadText: document.getElementById("model-load-text"),
  thinking: document.getElementById("thinking"),
  messages: document.getElementById("messages"),
  form: document.getElementById("chat-form"),
  prompt: document.getElementById("prompt"),
  send: document.getElementById("send"),
  init: document.getElementById("init-ai"),
  provider: document.getElementById("provider"),
  modelField: document.getElementById("model-field"),
  modelPreset: document.getElementById("model-preset"),
};

const state = {
  provider: null,
  generator: null,
  chromeSession: null,
  modelId: null,
  modelDevice: null,
  playground: null,
  persistentCacheReady: false,
  conversationTurns: [],
  cacheSourceHint: "",
};

env.allowLocalModels = false;
env.useBrowserCache = true;
env.useWasmCache = true;

const CACHE_PROBE_FILES = ["config.json", "tokenizer.json", "onnx/decoder_model_merged_q4f16.onnx_data"];
const MODEL_INIT_TIMEOUT_MS = 90000;

function buildHfResolveUrl(modelId, fileName) {
  return `https://huggingface.co/${modelId}/resolve/main/${fileName}`;
}

async function detectModelCacheSource(modelId) {
  if (typeof caches === "undefined") return "";

  try {
    const cache = await caches.open(env.cacheKey || "transformers-cache");
    let hits = 0;

    for (const fileName of CACHE_PROBE_FILES) {
      const url = buildHfResolveUrl(modelId, fileName);
      const match = await cache.match(url);
      if (match) hits += 1;
    }

    if (hits === 0) return "from network";
    if (hits === CACHE_PROBE_FILES.length) return "from cache";
    return "cache + network";
  } catch {
    return "";
  }
}

async function configureTransformersCache(modelId) {
  env.useCustomCache = false;
  env.customCache = null;
  env.useBrowserCache = true;
  env.useWasmCache = true;
  state.persistentCacheReady = true;
  state.cacheSourceHint = await detectModelCacheSource(modelId);
}

async function withTimeout(promise, timeoutMs, label) {
  let timer = null;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`${label} timed out after ${Math.round(timeoutMs / 1000)}s`)), timeoutMs);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

const MODEL_PRESETS = {
  gemma4e2bitonnx: {
    label: "Gemma 4 E2B IT (ONNX)",
    modelId: "onnx-community/gemma-4-E2B-it-ONNX",
  },
  gemma4e4bitonnx: {
    label: "Gemma 4 E4B IT (ONNX)",
    modelId: "onnx-community/gemma-4-E4B-it-ONNX",
  },
};

const TRANSFORMERS_MAX_PROMPT_CHARS = 2500;
const TRANSFORMERS_RETRY_PROMPT_CHARS = 1200;
const TRANSFORMERS_MAX_NEW_TOKENS = 700;
const TRANSFORMERS_RETRY_MAX_NEW_TOKENS = 350;
const MAX_CONVERSATION_TURNS = 8;

const PLAN_SHAPE_TEXT =
  '{"action":"read|write|answer","target":"html|css|js|all","summary":"string","information":"string","changes":{"html":"string","css":"string","js":"string"}}';

const PLAN_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["action", "target", "summary", "information", "changes"],
  properties: {
    action: { type: "string", enum: ["read", "write", "answer"] },
    target: { type: "string", enum: ["html", "css", "js", "all"] },
    summary: { type: "string" },
    information: { type: "string" },
    changes: {
      type: "object",
      additionalProperties: false,
      required: ["html", "css", "js"],
      properties: {
        html: { type: "string" },
        css: { type: "string" },
        js: { type: "string" },
      },
    },
  },
};

function setStatus(text) {
  ui.status.textContent = text;
}

function setModelLoading(visible, text = "", percent = null) {
  if (!ui.modelLoad) return;

  ui.modelLoad.hidden = !visible;
  if (ui.modelLoadText) {
    ui.modelLoadText.textContent = text;
  }

  if (ui.modelLoadFill) {
    ui.modelLoadFill.style.width = Number.isFinite(percent) ? `${Math.max(0, Math.min(100, percent))}%` : "0%";
  }
}

function updateModelLoading(progress) {
  if (!ui.modelLoad) return;

  let percent = null;
  if (typeof progress?.progress === "number") {
    percent = progress.progress <= 1 ? progress.progress * 100 : progress.progress;
  } else if (typeof progress?.loaded === "number" && typeof progress?.total === "number" && progress.total > 0) {
    percent = (progress.loaded / progress.total) * 100;
  }

  const parts = [];
  if (progress?.status) parts.push(String(progress.status));
  if (progress?.file) parts.push(String(progress.file).split("/").pop());
  if (Number.isFinite(percent)) parts.push(`${Math.round(percent)}%`);
  const sourceLabel = state.cacheSourceHint;
  if (sourceLabel) parts.push(sourceLabel);

  setModelLoading(true, parts.join(" • ") || "Loading model...", percent);
}

function setThinking(isThinking) {
  ui.thinking.hidden = !isThinking;
  ui.send.textContent = isThinking ? "Thinking..." : "Send";
}

function getSelectedProvider() {
  return ui.provider?.value === "transformers" ? "transformers" : "chrome";
}

function getProviderLabel(provider) {
  return provider === "transformers" ? "Transformers.js" : "Chrome built-in AI";
}

function isReady(provider = getSelectedProvider()) {
  if (provider === "transformers") return state.provider === "transformers" && !!state.generator;
  return state.provider === "chrome" && !!state.chromeSession;
}

function syncReadyUi() {
  const ready = isReady(getSelectedProvider());
  ui.prompt.disabled = !ready;
  ui.send.disabled = !ready;
}

function syncInitButtonUi() {
  ui.init.hidden = isReady(getSelectedProvider());
}

function syncProviderUi() {
  const isTransformers = getSelectedProvider() === "transformers";
  if (ui.modelField) {
    ui.modelField.hidden = !isTransformers;
  }
}

function scrollMessagesToBottom() {
  ui.messages.scrollTop = ui.messages.scrollHeight;
}

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = text;
  ui.messages.appendChild(div);
  scrollMessagesToBottom();
  return div;
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function formatJsCode(code) {
  try {
    return await prettier.format(code, {
      parser: "babel",
      plugins: [prettierBabel, prettierEstree],
    });
  } catch {
    return code;
  }
}

async function formatCssCode(code) {
  try {
    return await prettier.format(code, {
      parser: "css",
      plugins: [prettierPostcss],
    });
  } catch {
    return code;
  }
}

async function formatHtmlCode(code) {
  try {
    return await prettier.format(code, {
      parser: "html",
      plugins: [prettierHtml],
    });
  } catch {
    return code;
  }
}

async function addAgentMessage(text) {
  const div = addMessage("agent", "");
  div.classList.add("typing");

  const started = Date.now();
  const chunkSize = text.length > 1000 ? 8 : text.length > 400 ? 4 : 2;
  for (let i = 0; i < text.length; i += chunkSize) {
    div.textContent = text.slice(0, i + chunkSize);
    scrollMessagesToBottom();
    await wait(18);
  }

  const elapsed = Date.now() - started;
  const minVisibleMs = 350;
  if (elapsed < minVisibleMs) {
    await wait(minVisibleMs - elapsed);
  }

  div.textContent = text;
  div.classList.remove("typing");
  scrollMessagesToBottom();
}

function truncate(text, max = 1500) {
  if (text.length <= max) return text;
  return `${text.slice(0, max)}\n\n...truncated...`;
}

function truncateForModel(text, maxChars) {
  const value = typeof text === "string" ? text.trim() : "";
  if (value.length <= maxChars) return value;
  return `${value.slice(0, maxChars)}\n/* truncated for model context */`;
}

function rememberConversationTurn(userText, assistantText) {
  const user = typeof userText === "string" ? userText.trim() : "";
  const assistant = typeof assistantText === "string" ? assistantText.trim() : "";
  if (!user && !assistant) return;

  state.conversationTurns.push({ user, assistant });
  if (state.conversationTurns.length > MAX_CONVERSATION_TURNS) {
    state.conversationTurns.splice(0, state.conversationTurns.length - MAX_CONVERSATION_TURNS);
  }
}

function getConversationContextText() {
  if (!Array.isArray(state.conversationTurns) || state.conversationTurns.length === 0) return "";

  return state.conversationTurns
    .map((turn, index) => {
      const user = truncateForModel(turn.user || "", 180).replace(/\n/g, " ");
      const assistant = truncateForModel(turn.assistant || "", 220).replace(/\n/g, " ");
      return `${index + 1}. User: ${user || "(empty)"}\n   Agent: ${assistant || "(empty)"}`;
    })
    .join("\n");
}

function isWebGpuOutOfMemoryError(error) {
  const message = String(error?.message || error || "").toLowerCase();
  return (
    message.includes("e_outofmemory") ||
    message.includes("outofmemory") ||
    message.includes("failed to create a webgpu compute pipeline") ||
    message.includes("failed to call ortrun")
  );
}

function getModelCodeContext(current, maxChars = TRANSFORMERS_MAX_PROMPT_CHARS) {
  return {
    html: truncateForModel(current.html, maxChars),
    css: truncateForModel(current.css, maxChars),
    js: truncateForModel(current.js, maxChars),
  };
}

function extractBalancedJson(text) {
  let start = -1;
  let depth = 0;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (ch === "{") {
      if (depth === 0) start = i;
      depth += 1;
      continue;
    }
    if (ch === "}") {
      depth -= 1;
      if (depth === 0 && start !== -1) return text.slice(start, i + 1);
      if (depth < 0) break;
    }
  }
  return null;
}

function extractJson(text) {
  const unwrapped = text.replace(/```json/gi, "").replace(/```/g, "").trim();
  const xmlMatch = unwrapped.match(/<json>([\s\S]*?)<\/json>/i);
  const candidate = xmlMatch ? xmlMatch[1].trim() : extractBalancedJson(unwrapped);
  if (!candidate) return null;
  try {
    return JSON.parse(candidate);
  } catch {
    return null;
  }
}

function normalizePlan(plan) {
  if (!plan || typeof plan !== "object") return null;
  const action = typeof plan.action === "string" ? plan.action.toLowerCase() : "answer";
  const target = typeof plan.target === "string" ? plan.target.toLowerCase() : "all";
  const summary = typeof plan.summary === "string" ? plan.summary : "";
  const information = typeof plan.information === "string" ? plan.information : "";
  const rawChanges = plan.changes && typeof plan.changes === "object" ? plan.changes : {};
  const changes = {
    html: typeof rawChanges.html === "string" ? rawChanges.html : "",
    css: typeof rawChanges.css === "string" ? rawChanges.css : "",
    js: typeof rawChanges.js === "string" ? rawChanges.js : "",
  };

  const allowedActions = new Set(["read", "write", "answer"]);
  const allowedTargets = new Set(["html", "css", "js", "all"]);
  return {
    action: allowedActions.has(action) ? action : "answer",
    target: allowedTargets.has(target) ? target : "all",
    summary,
    information,
    changes,
  };
}

function isLikelyCodeSnippet(target, value) {
  if (typeof value !== "string") return false;
  const text = value.trim();
  if (!text) return false;
  if (target === "html") return /<\s*\w+[^>]*>/.test(text) || /<\/?\w+>/.test(text);
  if (target === "css") return /\{[\s\S]*\}/.test(text) || /[.#\w-]+\s*:\s*[^;]+;/.test(text);
  if (target === "js") return /[;{}()=>]|\b(function|const|let|var|if|for|while|document|window)\b/.test(text);
  return false;
}

function normalizeEscapedCodeText(value) {
  if (typeof value !== "string") return "";

  // Some model responses include literal escape sequences (e.g. "\\n") inside JSON fields.
  // Convert them to real newlines/tabs so LiveCodes receives valid source text.
  if (!value.includes("\\n") && !value.includes("\\r\\n") && !value.includes("\\t")) {
    return value;
  }

  return value.replace(/\\r\\n/g, "\n").replace(/\\n/g, "\n").replace(/\\t/g, "\t");
}

function getWritePayload(plan) {
  if (!plan || plan.action !== "write") return null;

  const changes = {
    html: normalizeEscapedCodeText(typeof plan.changes?.html === "string" ? plan.changes.html : ""),
    css: normalizeEscapedCodeText(typeof plan.changes?.css === "string" ? plan.changes.css : ""),
    js: normalizeEscapedCodeText(typeof plan.changes?.js === "string" ? plan.changes.js : ""),
  };

  const applied = {};
  if (isLikelyCodeSnippet("html", changes.html)) applied.html = changes.html;
  if (isLikelyCodeSnippet("css", changes.css)) applied.css = changes.css;
  if (isLikelyCodeSnippet("js", changes.js)) applied.js = changes.js;

  return Object.keys(applied).length > 0 ? applied : null;
}

function getCodeStats(text) {
  const value = text || "";
  return { chars: value.length, lines: value.length === 0 ? 0 : value.split("\n").length };
}

function formatStatsLine(label, text) {
  const stats = getCodeStats(text);
  return `${label}: ${stats.lines} lines, ${stats.chars} chars`;
}

function uniqueMatches(text, regex) {
  const values = new Set();
  const source = text || "";
  for (const match of source.matchAll(regex)) {
    if (match[1]) values.add(match[1].toLowerCase());
  }
  return values;
}

function getHtmlHints(beforeText, afterText) {
  const before = beforeText || "";
  const after = afterText || "";
  const hints = [];

  const beforeTags = Array.from(uniqueMatches(before, /<\s*([a-z0-9-]+)/gi));
  const afterTags = Array.from(uniqueMatches(after, /<\s*([a-z0-9-]+)/gi));
  const addedTags = afterTags.filter((tag) => !beforeTags.includes(tag));
  if (addedTags.length > 0) hints.push(`added ${addedTags.slice(0, 4).join(", ")} elements`);

  const headingBefore = before
    .match(/<h[1-3][^>]*>([\s\S]*?)<\/h[1-3]>/i)?.[1]
    ?.replace(/<[^>]+>/g, "")
    .trim();
  const headingAfter = after
    .match(/<h[1-3][^>]*>([\s\S]*?)<\/h[1-3]>/i)?.[1]
    ?.replace(/<[^>]+>/g, "")
    .trim();
  if (headingBefore !== headingAfter && headingAfter) {
    hints.push(`updated heading text to "${truncate(headingAfter, 60)}"`);
  }

  const buttonBefore = before
    .match(/<button[^>]*>([\s\S]*?)<\/button>/i)?.[1]
    ?.replace(/<[^>]+>/g, "")
    .trim();
  const buttonAfter = after
    .match(/<button[^>]*>([\s\S]*?)<\/button>/i)?.[1]
    ?.replace(/<[^>]+>/g, "")
    .trim();
  if (buttonBefore !== buttonAfter && buttonAfter) {
    hints.push(`changed button label to "${truncate(buttonAfter, 40)}"`);
  }
  return hints;
}

function getCssHints(beforeText, afterText) {
  const before = beforeText || "";
  const after = afterText || "";
  const hints = [];

  const propsBefore = uniqueMatches(before, /([a-z-]+)\s*:/gi);
  const propsAfter = uniqueMatches(after, /([a-z-]+)\s*:/gi);
  const addedProps = Array.from(propsAfter).filter((name) => !propsBefore.has(name));

  if (
    addedProps.some((p) =>
      ["display", "grid", "flex", "position", "gap", "align-items", "justify-content"].includes(p)
    )
  ) {
    hints.push("adjusted layout styling");
  }
  if (addedProps.some((p) => ["color", "background", "background-color", "border-color"].includes(p))) {
    hints.push("updated color styling");
  }
  if (
    addedProps.some((p) =>
      ["padding", "margin", "border-radius", "width", "height", "font-size"].includes(p)
    )
  ) {
    hints.push("tuned spacing/size styling");
  }

  const beforeCount = (before.match(/[^{}]+\{/g) || []).length;
  const afterCount = (after.match(/[^{}]+\{/g) || []).length;
  if (afterCount > beforeCount) hints.push("added new style rules");
  if (afterCount < beforeCount) hints.push("removed some style rules");

  return [...new Set(hints)];
}

function getJsHints(beforeText, afterText) {
  const before = beforeText || "";
  const after = afterText || "";
  const hints = [];

  const beforeEvents = uniqueMatches(before, /addEventListener\(\s*['"]([^'"]+)['"]/gi);
  const afterEvents = uniqueMatches(after, /addEventListener\(\s*['"]([^'"]+)['"]/gi);
  const newEvents = Array.from(afterEvents).filter((e) => !beforeEvents.has(e));
  if (newEvents.length > 0) hints.push(`added ${newEvents.slice(0, 3).join(", ")} event handlers`);

  const beforeFns = uniqueMatches(before, /function\s+([a-zA-Z0-9_]+)/g);
  const afterFns = uniqueMatches(after, /function\s+([a-zA-Z0-9_]+)/g);
  const newFns = Array.from(afterFns).filter((f) => !beforeFns.has(f));
  if (newFns.length > 0) hints.push("introduced new helper logic");

  if (
    (before.includes("console.log") || before.includes("console.error")) !==
    (after.includes("console.log") || after.includes("console.error"))
  ) {
    hints.push("changed logging/debug behavior");
  }

  if (hints.length === 0) hints.push("updated script behavior");
  return hints;
}

function summarizeTargetChange(targetLabel, beforeText, afterText) {
  if (targetLabel === "HTML") {
    const hints = getHtmlHints(beforeText, afterText);
    return hints.length > 0 ? `HTML: ${hints.join("; ")}.` : "HTML: updated page structure/content.";
  }
  if (targetLabel === "CSS") {
    const hints = getCssHints(beforeText, afterText);
    return hints.length > 0 ? `CSS: ${hints.join("; ")}.` : "CSS: updated styling rules.";
  }
  const hints = getJsHints(beforeText, afterText);
  return `JS: ${hints.join("; ")}.`;
}

function listChangedTargets(before, after) {
  const changed = [];
  if ((before.html || "").trim() !== (after.html || "").trim()) changed.push("HTML");
  if ((before.css || "").trim() !== (after.css || "").trim()) changed.push("CSS");
  if ((before.js || "").trim() !== (after.js || "").trim()) changed.push("JS");
  return changed;
}

function formatReadMessage(target, code) {
  const providerLabel = getProviderLabel(state.provider);
  if (target === "all") {
    return [
      `Read current playground code (provider: ${providerLabel}).`,
      "",
      formatStatsLine("HTML", code.html),
      formatStatsLine("CSS", code.css),
      formatStatsLine("JS", code.js),
      "",
      `HTML:\n${truncate(code.html, 500)}`,
      `CSS:\n${truncate(code.css, 500)}`,
      `JS:\n${truncate(code.js, 500)}`,
    ].join("\n\n");
  }

  const key = target === "html" ? "html" : target === "css" ? "css" : "js";
  return [
    `Read ${target.toUpperCase()} from the playground (provider: ${providerLabel}).`,
    "",
    formatStatsLine(target.toUpperCase(), code[key] || ""),
    "",
    `${target.toUpperCase()}:\n${truncate(code[key] || "", 1200)}`,
  ].join("\n");
}

function formatWriteMessage(before, after) {
  const providerLabel = getProviderLabel(state.provider);
  const changed = listChangedTargets(before, after);
  if (changed.length === 0) {
    return "Write action completed, but no code differences were detected.";
  }

  const sections = [];
  if (changed.includes("HTML")) {
    sections.push(summarizeTargetChange("HTML", before.html || "", after.html || ""));
  }
  if (changed.includes("CSS")) {
    sections.push(summarizeTargetChange("CSS", before.css || "", after.css || ""));
  }
  if (changed.includes("JS")) {
    sections.push(summarizeTargetChange("JS", before.js || "", after.js || ""));
  }

  return [
    `Applied write action using ${providerLabel} and re-ran the playground.`,
    `Updated targets: ${changed.join(", ")}.`,
    "",
    ...sections,
    "",
    "Summary: The requested edits were applied and the result panel was refreshed.",
  ].join("\n\n");
}

async function initPlayground() {
  const initialHtml = await formatHtmlCode(`
<main>
  <h1>Hello from LiveCodes</h1>
  <button id="cta">Click me</button>
</main>
`);
  const initialCss = await formatCssCode(`
body {
  font-family: system-ui;
  padding: 2rem;
}

#cta {
  padding: 0.6rem 1rem;
  border: 0;
  background: #1f7a5a;
  color: #fff;
}
`);
  const initialJs = await formatJsCode(`
document.getElementById('cta').addEventListener('click', () => console.log('clicked'));
`);

  state.playground = await createPlayground("#playground", {
    params: {
      html: initialHtml,
      css: initialCss,
      js: initialJs,
      console: "open",
    },
    loading: "eager",
  });
}

async function initChromeAi() {
  if (!globalThis.LanguageModel) {
    setStatus("Chrome Prompt API unavailable in this browser/profile.");
    return;
  }

  const expected = {
    expectedInputs: [{ type: "text", languages: ["en"] }],
    expectedOutputs: [{ type: "text", languages: ["en"] }],
  };

  setStatus("Checking Chrome built-in model availability...");
  const availability = await LanguageModel.availability(expected);
  if (availability === "unavailable") {
    setStatus("Chrome built-in model unavailable on this device/profile.");
    return;
  }

  state.chromeSession = await LanguageModel.create({
    ...expected,
    initialPrompts: [
      {
        role: "system",
        content:
          "You are an agent that controls a coding playground. Return valid JSON only with schema " +
          PLAN_SHAPE_TEXT +
          ". For action=write, place complete updated source in changes.html/changes.css/changes.js for any files you want to modify and leave unchanged ones empty.",
      },
    ],
  });

  state.provider = "chrome";
  state.generator = null;
  state.modelId = null;
  state.modelDevice = null;
  state.conversationTurns = [];

  setStatus("Chrome built-in AI ready.");
  syncReadyUi();
  syncInitButtonUi();
}

async function initTransformersAi() {
  const preset = MODEL_PRESETS[ui.modelPreset.value] || MODEL_PRESETS.gemma4e2bitonnx;
  const modelDevice = navigator.gpu ? "webgpu" : "wasm";

  state.cacheSourceHint = "";
  await configureTransformersCache(preset.modelId);

  if (!navigator.gpu) {
    setStatus("WebGPU unavailable. Loading Transformers.js with CPU/WASM fallback...");
  }

  const sourceLabel = state.cacheSourceHint;
  setStatus(`Loading Transformers.js model: ${preset.modelId}${sourceLabel ? ` (${sourceLabel})` : ""}`);
  setModelLoading(true, sourceLabel ? `Starting model load (${sourceLabel})...` : "Starting model load...", 0);

  try {
    const [processor, model] = await withTimeout(
      Promise.all([
        AutoProcessor.from_pretrained(preset.modelId, {
          progress_callback: updateModelLoading,
        }),
        Gemma4ForConditionalGeneration.from_pretrained(preset.modelId, {
          dtype: "q4f16",
          device: modelDevice,
          progress_callback: updateModelLoading,
        }),
      ]),
      MODEL_INIT_TIMEOUT_MS,
      "Transformers model initialization"
    );

    state.generator = { processor, model };
    state.chromeSession = null;
    state.provider = "transformers";
    state.modelId = preset.modelId;
    state.modelDevice = modelDevice;
    state.conversationTurns = [];

    setModelLoading(false);
    setStatus(
      `Transformers.js ready: ${preset.modelId} (${modelDevice.toUpperCase()})${
        sourceLabel ? `, loaded ${sourceLabel}` : ""
      }`
    );
    syncReadyUi();
    syncInitButtonUi();
  } catch (error) {
    state.generator = null;
    state.modelId = null;
    state.modelDevice = null;
    state.provider = null;
    setModelLoading(false);
    const message = String(error?.message || error || "unknown error");
    syncReadyUi();
    syncInitButtonUi();
    throw new Error(`Unable to load ${preset.label} in Transformers.js: ${message}`);
  }
}

function buildPlannerPrompt(input, current) {
  const context = state.provider === "transformers" ? getModelCodeContext(current) : current;
  const conversationContext = getConversationContextText();
  return [
    "You are an agent that controls a coding playground.",
    "Return only valid JSON. Do not include markdown.",
    "Use this schema exactly:",
    PLAN_SHAPE_TEXT,
    "Rules:",
    "- Use action=read when the user asks to inspect, show, or explain existing code.",
    "- Use action=write when the user asks to modify code.",
    "- Use action=answer for questions or guidance that should not change code.",
    "- Always populate information with your direct natural-language response to the user.",
    "- For action=write, put complete updated source for each changed file in changes.html, changes.css, or changes.js.",
    "- For files that should remain unchanged, set that changes field to an empty string.",
    "- Every non-empty changes field must be different from the current code for that file.",
    "- For action=read or answer, set all changes fields to empty strings.",
    "- Do not describe code edits without providing code in the relevant changes fields.",
    "- Output must start with { and end with }.",
    "",
    `User request: ${input}`,
    conversationContext ? "" : null,
    conversationContext ? `Recent conversation:\n${conversationContext}` : null,
    "",
    "Current code context:",
    `HTML:\n${context.html}`,
    `CSS:\n${context.css}`,
    `JS:\n${context.js}`,
  ]
    .filter((line) => line !== null)
    .join("\n");
}

async function promptChrome(promptText, useConstraint = false) {
  if (!useConstraint) return state.chromeSession.prompt(promptText);

  try {
    return await state.chromeSession.prompt(promptText, {
      responseConstraint: PLAN_SCHEMA,
      omitResponseConstraintInput: true,
    });
  } catch {
    return state.chromeSession.prompt(promptText);
  }
}

async function promptTransformers(promptText) {
  const runtime = state.generator;
  if (!runtime?.processor || !runtime?.model) {
    throw new Error("Transformers.js model is not initialized.");
  }

  const runPrompt = async (text, maxNewTokens) => {
    const messages = [{ role: "user", content: [{ type: "text", text }] }];
    const prompt = runtime.processor.apply_chat_template(messages, {
      enable_thinking: false,
      add_generation_prompt: true,
    });
    const inputs = await runtime.processor(prompt);

    let generatedText = "";
    const streamer = new TextStreamer(runtime.processor.tokenizer, {
      skip_prompt: true,
      callback_function: (token) => {
        generatedText += token;
      },
    });

    await runtime.model.generate({
      ...inputs,
      max_new_tokens: maxNewTokens,
      do_sample: false,
      streamer,
    });

    return generatedText.trim();
  };

  try {
    return await runPrompt(promptText, TRANSFORMERS_MAX_NEW_TOKENS);
  } catch (error) {
    if (!isWebGpuOutOfMemoryError(error)) throw error;

    setStatus("WebGPU ran out of memory. Retrying with a smaller prompt...");
    const retryPrompt = truncateForModel(promptText, TRANSFORMERS_RETRY_PROMPT_CHARS);
    return runPrompt(retryPrompt, TRANSFORMERS_RETRY_MAX_NEW_TOKENS);
  }
}

async function promptActiveProvider(promptText, useConstraint = false) {
  if (state.provider === "transformers") {
    return promptTransformers(promptText);
  }
  return promptChrome(promptText, useConstraint);
}

async function getPlanFromChrome(input, current) {
  const plannerPrompt = buildPlannerPrompt(input, current);
  const raw = await promptActiveProvider(plannerPrompt, true);
  return { raw, plan: normalizePlan(extractJson(raw)) };
}

async function getCurrentCode() {
  const code = await state.playground.getCode();
  return {
    html: code.markup.content,
    css: code.style.content,
    js: code.script.content,
  };
}

async function setPlaygroundCode(next) {
  await state.playground.setConfig({
    markup: {
      language: "html",
      content: next.html,
    },
    style: {
      language: "css",
      content: next.css,
    },
    script: {
      language: "javascript",
      content: next.js,
    },
  });

  await state.playground.run();
}

async function applyCodeUpdate(changes) {
  const current = await getCurrentCode();
  const next = { ...current };

  if (typeof changes?.html === "string") next.html = changes.html;
  if (typeof changes?.css === "string") next.css = changes.css;
  if (typeof changes?.js === "string") next.js = changes.js;

  next.html = await formatHtmlCode(next.html);
  next.css = await formatCssCode(next.css);
  next.js = await formatJsCode(next.js);

  await setPlaygroundCode(next);

  const applied = await getCurrentCode();
  return { before: current, after: applied };
}

async function handleUserPrompt(input) {
  if (!isReady(getSelectedProvider())) {
    await addAgentMessage(`Initialize ${getProviderLabel(getSelectedProvider())} before sending a message.`);
    return;
  }

  const current = await getCurrentCode();
  const firstAttempt = await getPlanFromChrome(input, current);

  let raw = firstAttempt.raw;
  let plan = firstAttempt.plan;

  if (!plan) {
    await addAgentMessage("Model response was not valid JSON. No code was changed.");
    rememberConversationTurn(input, "Response parse failed; no code changes applied.");
    return;
  }

  if (plan.action === "read") {
    const readMsg = formatReadMessage(plan.target, current);
    await addAgentMessage(readMsg);
    if (plan.information && plan.information.trim()) {
      await addAgentMessage(plan.information.trim());
    }
    rememberConversationTurn(input, plan.information || `Read action completed (${plan.target}).`);
    return;
  }

  if (plan.action === "write") {
    const payload = getWritePayload(plan);
    if (!payload) {
      await addAgentMessage(
        "Write action was detected, but the returned payload did not look like code. No code was changed."
      );
      if (plan.information && plan.information.trim()) {
        await addAgentMessage(plan.information.trim());
      }
      rememberConversationTurn(input, plan.information || "Write action rejected because payload did not contain code.");
      return;
    }

    try {
      const result = await applyCodeUpdate(payload);
      const writeMsg = formatWriteMessage(result.before, result.after);
      await addAgentMessage(writeMsg);
      if (plan.information && plan.information.trim()) {
        await addAgentMessage(plan.information.trim());
      }
      rememberConversationTurn(input, plan.summary || plan.information || "Write action applied.");
    } catch (error) {
      await addAgentMessage(`Failed to apply code update: ${error.message}`);
      rememberConversationTurn(input, `Write action failed: ${error.message}`);
    }
    return;
  }

  const info =
    (typeof plan.information === "string" && plan.information.trim()) ||
    (typeof plan.summary === "string" && plan.summary.trim()) ||
    "No code change was applied.";
  await addAgentMessage(info);
  rememberConversationTurn(input, info);
}

ui.init.addEventListener("click", async () => {
  ui.init.disabled = true;
  try {
    const provider = getSelectedProvider();
    if (provider === "transformers") {
      await initTransformersAi();
    } else {
      await initChromeAi();
    }
  } catch (error) {
    setStatus(`AI init failed: ${error.message}`);
  } finally {
    ui.init.disabled = false;
    syncReadyUi();
    syncInitButtonUi();
  }
});

ui.provider?.addEventListener("change", () => {
  syncProviderUi();
  syncReadyUi();
  syncInitButtonUi();

  const provider = getSelectedProvider();
  if (!isReady(provider)) {
    setStatus(`Selected provider: ${getProviderLabel(provider)}. Click Initialize AI.`);
  }
});

ui.form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const text = ui.prompt.value.trim();
  if (!text) return;

  addMessage("user", text);
  ui.prompt.value = "";

  ui.send.disabled = true;
  setThinking(true);
  try {
    await handleUserPrompt(text);
  } catch (error) {
    await addAgentMessage(`Error: ${error.message}`);
  } finally {
    setThinking(false);
    syncReadyUi();
  }
});

ui.prompt?.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (!ui.send.disabled) {
      ui.form.requestSubmit();
    }
  }
});

initPlayground().catch((error) => {
  setStatus(`Failed to initialize LiveCodes: ${error.message}`);
});

if (navigator.storage?.persist) {
  navigator.storage.persist().catch(() => {});
}

syncReadyUi();
syncInitButtonUi();
syncProviderUi();
setStatus("Selected provider: Chrome built-in AI. Click Initialize AI.");
