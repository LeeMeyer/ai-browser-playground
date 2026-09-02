import { createPlayground } from "https://cdn.jsdelivr.net/npm/livecodes@0.14.1/+esm";
import { env, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.0";

const ui = {
  status: document.getElementById("status"),
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
  generatorTask: null,
  chromeSession: null,
  modelId: null,
  playground: null,
};

env.allowLocalModels = false;
env.useBrowserCache = true;

const PLAN_SHAPE_TEXT =
  '{"action":"read|write|answer","target":"html|css|js|all","summary":"string","content":"string","code":{"html":"string","css":"string","js":"string"}}';

const PLAN_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["action", "target", "summary", "content", "code"],
  properties: {
    action: {
      type: "string",
      enum: ["read", "write", "answer"],
    },
    target: {
      type: "string",
      enum: ["html", "css", "js", "all"],
    },
    summary: { type: "string" },
    content: { type: "string" },
    code: {
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

const TRANSFORMERS_MODEL_CONFIG = {
  "Xenova/flan-t5-base": {
    task: "text2text-generation",
    pipelineOptionsList: [{}],
  },
  "onnx-community/Qwen2.5-Coder-0.5B-Instruct": {
    task: "text-generation",
    pipelineOptionsList: [{ dtype: "q4" }, { dtype: "q4f16" }, { dtype: "fp16" }, {}],
  },
  "onnx-community/Qwen2.5-Coder-1.5B-Instruct": {
    task: "text-generation",
    pipelineOptionsList: [{ dtype: "q4" }, { dtype: "q4f16" }, { dtype: "fp16" }, {}],
  },
  "onnx-community/deepseek-coder-1.3b-instruct-ONNX": {
    task: "text-generation",
    pipelineOptionsList: [{ dtype: "q4" }, { dtype: "q4f16" }, { dtype: "fp16" }, {}],
  },
  "onnx-community/Llama-3.1-8B-Instruct": {
    task: "text-generation",
    pipelineOptionsList: [{ dtype: "q4" }, { dtype: "q4f16" }, { dtype: "fp16" }, {}],
  },
};

function setStatus(text) {
  ui.status.textContent = text;
}

function setThinking(isThinking) {
  if (!ui.thinking) return;
  ui.thinking.hidden = !isThinking;
  ui.send.textContent = isThinking ? "Thinking..." : "Send";
}

function isProviderReady(provider) {
  if (provider === "chrome") {
    return state.provider === "chrome" && !!state.chromeSession;
  }
  if (provider === "transformers") {
    return state.provider === "transformers" && !!state.generator;
  }
  return false;
}

function syncReadyUi() {
  const ready = isProviderReady(ui.provider.value);
  ui.prompt.disabled = !ready;
  ui.send.disabled = !ready;
}

function syncInitButtonUi() {
  ui.init.hidden = isProviderReady(ui.provider.value);
}

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = text;
  ui.messages.appendChild(div);
  ui.messages.scrollTop = ui.messages.scrollHeight;
}

function truncate(text, max = 1500) {
  if (text.length <= max) return text;
  return `${text.slice(0, max)}\n\n...truncated...`;
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
      if (depth === 0 && start !== -1) {
        return text.slice(start, i + 1);
      }
      if (depth < 0) break;
    }
  }

  return null;
}

function extractJson(text) {
  const unwrapped = text
    .replace(/```json/gi, "")
    .replace(/```/g, "")
    .trim();
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
  const content = typeof plan.content === "string" ? plan.content : "";
  const summary = typeof plan.summary === "string" ? plan.summary : "";
  const rawCode = plan.code && typeof plan.code === "object" ? plan.code : {};
  const code = {
    html: typeof rawCode.html === "string" ? rawCode.html : "",
    css: typeof rawCode.css === "string" ? rawCode.css : "",
    js: typeof rawCode.js === "string" ? rawCode.js : "",
  };

  const allowedActions = new Set(["read", "write", "answer"]);
  const allowedTargets = new Set(["html", "css", "js", "all"]);

  return {
    action: allowedActions.has(action) ? action : "answer",
    target: allowedTargets.has(target) ? target : "all",
    content,
    summary,
    code,
  };
}

function heuristicPlanFromInput(input) {
  const lower = input.toLowerCase();
  const readHints = ["show", "read", "inspect", "what is", "current", "display", "print"];
  const writeHints = ["change", "update", "make", "set", "replace", "add", "remove", "edit", "modify"];

  const hasReadHint = readHints.some((w) => lower.includes(w));
  const hasWriteHint = writeHints.some((w) => lower.includes(w));

  let target = "all";
  if (lower.includes("html") || lower.includes("markup")) target = "html";
  if (lower.includes("css") || lower.includes("style")) target = "css";
  if (lower.includes("js") || lower.includes("javascript") || lower.includes("script")) target = "js";

  if (hasWriteHint && !hasReadHint) {
    return {
      action: "write",
      target,
      content: "",
      code: { html: "", css: "", js: "" },
      summary: "I understood this as a code update request, but I need a stricter model response. Try Initialize AI again or use a stronger model.",
    };
  }

  if (hasReadHint || !hasWriteHint) {
    return {
      action: "read",
      target,
      content: "",
      code: { html: "", css: "", js: "" },
      summary: "Falling back to a direct code read because the model output was not valid JSON.",
    };
  }

  return {
    action: "answer",
    target: "all",
    content: "",
    code: { html: "", css: "", js: "" },
    summary: "I could not determine an action from that request.",
  };
}

async function initPlayground() {
  state.playground = await createPlayground("#playground", {
    params: {
      html: "<main><h1>Hello from LiveCodes</h1><button id='cta'>Click me</button></main>",
      css: "body{font-family:system-ui;padding:2rem;}#cta{padding:0.6rem 1rem;border:0;background:#1f7a5a;color:#fff;}",
      js: "document.getElementById('cta').addEventListener('click', () => console.log('clicked'));",
      console: "open",
    },
    loading: "eager",
  });
}

async function initTransformersAi() {
  const modelId = ui.modelPreset.value;
  if (!modelId) {
    setStatus("Select a model first.");
    return;
  }

  const selectedConfig = TRANSFORMERS_MODEL_CONFIG[modelId];
  if (!selectedConfig) {
    setStatus("Unsupported model preset. Pick one of the listed options.");
    return;
  }

  const { task, pipelineOptionsList } = selectedConfig;

  let lastError = null;
  for (const options of pipelineOptionsList) {
    const dtypeLabel = options.dtype ? `dtype=${options.dtype}` : "dtype=default";
    setStatus(`Loading model ${modelId} (${task}, ${dtypeLabel}). First load may take a while...`);

    try {
      state.generator = await pipeline(task, modelId, {
        ...options,
        progress_callback: (event) => {
          if (!event || !event.status) return;
          const progress = typeof event.progress === "number" ? ` (${Math.round(event.progress)}%)` : "";
          setStatus(`${event.status}${progress}`);
        },
      });
      lastError = null;
      break;
    } catch (error) {
      lastError = error;
    }
  }

  if (!state.generator) {
    throw new Error(`Model load failed for all dtypes: ${lastError?.message || "unknown error"}`);
  }

  setStatus("Finalizing model setup...");
  state.provider = "transformers";
  state.generatorTask = task;
  state.chromeSession = null;
  state.modelId = modelId;
  setStatus(`Model loaded: ${modelId} (${task})`);
}

function extractGeneratedText(output) {
  if (!Array.isArray(output) || output.length === 0) return "";
  const generated = output[0]?.generated_text ?? output[0]?.summary_text ?? "";

  if (typeof generated === "string") {
    return generated;
  }

  if (Array.isArray(generated)) {
    const assistant = [...generated].reverse().find((entry) => entry?.role === "assistant");
    if (assistant && typeof assistant.content === "string") {
      return assistant.content;
    }

    const last = generated.at(-1);
    if (last && typeof last.content === "string") {
      return last.content;
    }
  }

  return `${generated}`;
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
          ". For target=all and action=write, put full updated code in code.html/code.css/code.js and keep content empty.",
      },
    ],
  });

  state.provider = "chrome";
  state.generator = null;
  setStatus("Chrome built-in AI ready.");
}

function buildPlannerPrompt(input, current) {
  return [
    "You are an agent that controls a coding playground.",
    "Return only valid JSON. Do not include markdown.",
    "Use this schema exactly:",
    PLAN_SHAPE_TEXT,
    "Rules:",
    "- Use action=read when the user asks to inspect, show, or explain existing code.",
    "- Use action=write when the user asks to modify code.",
    "- For action=write and target=all, fill code.html, code.css, and code.js with complete updated code.",
    "- For action=write and single target, use content only for that target and keep code fields empty strings.",
    "- For action=read or answer, keep content and all code fields empty strings.",
    "- Output must start with { and end with }.",
    "",
    `User request: ${input}`,
    "",
    "Current code context:",
    `HTML:\n${current.html}`,
    `CSS:\n${current.css}`,
    `JS:\n${current.js}`,
  ].join("\n");
}

function buildJsonRepairPrompt(raw) {
  return [
    "Convert the following text to strict JSON using this exact schema:",
    PLAN_SHAPE_TEXT,
    "Return only JSON and nothing else.",
    "",
    "TEXT:",
    raw,
  ].join("\n");
}

async function promptChrome(promptText, useConstraint = false) {
  if (!useConstraint) {
    return state.chromeSession.prompt(promptText);
  }

  try {
    return await state.chromeSession.prompt(promptText, {
      responseConstraint: PLAN_SCHEMA,
      omitResponseConstraintInput: true,
    });
  } catch {
    return state.chromeSession.prompt(promptText);
  }
}

async function getPlanFromModel(input, current) {
  const plannerPrompt = buildPlannerPrompt(input, current);
  const output =
    state.generatorTask === "text-generation"
      ? await state.generator(
          [
            {
              role: "user",
              content: plannerPrompt,
            },
          ],
          {
            max_new_tokens: 260,
            temperature: 0.1,
            do_sample: false,
          }
        )
      : await state.generator(plannerPrompt, {
          max_new_tokens: 260,
          temperature: 0.1,
          do_sample: false,
        });

  const raw = extractGeneratedText(output);
  return { raw, plan: normalizePlan(extractJson(raw)) };
}

async function getPlanFromChrome(input, current) {
  const plannerPrompt = buildPlannerPrompt(input, current);
  const raw = await promptChrome(plannerPrompt, true);
  return { raw, plan: normalizePlan(extractJson(raw)) };
}

async function repairPlanWithProvider(rawText) {
  const prompt = buildJsonRepairPrompt(rawText);

  if (state.provider === "chrome") {
    const raw = await promptChrome(prompt, true);
    return { raw, plan: normalizePlan(extractJson(raw)) };
  }

  const runOutput =
    state.generatorTask === "text-generation"
      ? await state.generator(
          [
            {
              role: "user",
              content: prompt,
            },
          ],
          {
            max_new_tokens: 220,
            temperature: 0.1,
            do_sample: false,
          }
        )
      : await state.generator(prompt, {
          max_new_tokens: 220,
          temperature: 0.1,
          do_sample: false,
        });
  const raw = extractGeneratedText(runOutput);
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

async function applyCodeUpdate(target, content, codeObject) {
  const current = await getCurrentCode();
  const next = { ...current };

  if (target === "html") next.html = content;
  if (target === "css") next.css = content;
  if (target === "js") next.js = content;

  // For target=all, prefer codeObject; fallback to parsing content for compatibility.
  if (target === "all") {
    try {
      let parsed = codeObject;
      if (!parsed || typeof parsed !== "object") {
        parsed = typeof content === "string" ? JSON.parse(content) : content;
      }
      next.html = typeof parsed.html === "string" ? parsed.html : next.html;
      next.css = typeof parsed.css === "string" ? parsed.css : next.css;
      next.js = typeof parsed.js === "string" ? parsed.js : next.js;
    } catch {
      throw new Error("Expected JSON content for target=all.");
    }
  }

  await state.playground.setConfig({
    markup: { content: next.html },
    style: { content: next.css },
    script: { content: next.js },
  });

  await state.playground.run();
  return next;
}

async function handleUserPrompt(input) {
  if (!isProviderReady(ui.provider.value)) {
    addMessage("agent", "Initialize the selected provider before sending a message.");
    return;
  }

  const current = await getCurrentCode();
  const firstAttempt =
    state.provider === "chrome"
      ? await getPlanFromChrome(input, current)
      : await getPlanFromModel(input, current);

  let raw = firstAttempt.raw;
  let plan = firstAttempt.plan;

  if (!plan) {
    const repaired = await repairPlanWithProvider(raw);
    raw = repaired.raw;
    plan = repaired.plan;
  }

  if (!plan) {
    plan = heuristicPlanFromInput(input);
  }

  if (!plan) {
    addMessage("agent", "I could not parse a JSON action from the model response.");
    addMessage("agent", truncate(raw));
    return;
  }

  if (plan.action === "read") {
    if (plan.target === "all") {
      addMessage(
        "agent",
        `${plan.summary || "Current code:"}\n\nHTML:\n${truncate(current.html, 500)}\n\nCSS:\n${truncate(current.css, 500)}\n\nJS:\n${truncate(current.js, 500)}`
      );
      return;
    }

    const value = current[plan.target] || "";
    addMessage("agent", `${plan.summary || "Current code:"}\n\n${plan.target.toUpperCase()}:\n${truncate(value, 1200)}`);
    return;
  }

  if (plan.action === "write") {
    const hasAllCode =
      plan.target === "all" &&
      plan.code &&
      (typeof plan.code.html === "string" || typeof plan.code.css === "string" || typeof plan.code.js === "string");

    if (!plan.content && !hasAllCode) {
      addMessage(
        "agent",
        "I understood this as a write request, but the model did not return updated code. Try a more explicit prompt or a stronger model."
      );
      return;
    }
    try {
      await applyCodeUpdate(plan.target, plan.content || "", plan.code);
      addMessage("agent", plan.summary || `Updated ${plan.target} and re-ran the playground.`);
    } catch (error) {
      addMessage("agent", `Failed to apply code update: ${error.message}`);
    }
    return;
  }

  addMessage("agent", plan.summary || truncate(raw));
}

function syncProviderUi() {
  const isTransformers = ui.provider.value === "transformers";
  ui.modelField.hidden = !isTransformers;
  if (ui.modelPreset) {
    ui.modelPreset.disabled = !isTransformers;
  }
  syncInitButtonUi();
}

ui.init.addEventListener("click", async () => {
  ui.init.disabled = true;
  try {
    if (ui.provider.value === "chrome") {
      await initChromeAi();
    } else {
      await initTransformersAi();
    }
  } catch (error) {
    setStatus(`AI init failed: ${error.message}`);
  } finally {
    ui.init.disabled = false;
    syncReadyUi();
    syncInitButtonUi();
  }
});

ui.provider.addEventListener("change", () => {
  syncProviderUi();
  syncReadyUi();
  syncInitButtonUi();
  if (ui.provider.value === "chrome") {
    setStatus("Selected provider: Chrome built-in AI. Click Initialize AI.");
  } else {
    setStatus("Selected provider: Transformers.js. Click Initialize AI.");
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
    addMessage("agent", `Error: ${error.message}`);
  } finally {
    setThinking(false);
    syncReadyUi();
  }
});

initPlayground().catch((error) => {
  setStatus(`Failed to initialize LiveCodes: ${error.message}`);
});

syncProviderUi();
syncReadyUi();
syncInitButtonUi();

if (ui.provider.value === "chrome") {
  setStatus("Selected provider: Chrome built-in AI. Click Initialize AI.");
} else {
  setStatus("Selected provider: Transformers.js. Click Initialize AI.");
}
