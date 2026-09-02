import { createPlayground } from "https://cdn.jsdelivr.net/npm/livecodes@0.14.1/+esm";

const ui = {
  status: document.getElementById("status"),
  thinking: document.getElementById("thinking"),
  messages: document.getElementById("messages"),
  form: document.getElementById("chat-form"),
  prompt: document.getElementById("prompt"),
  send: document.getElementById("send"),
  init: document.getElementById("init-ai"),
};

const state = {
  chromeSession: null,
  playground: null,
};

const PLAN_SHAPE_TEXT =
  '{"action":"read|write|answer","target":"html|css|js|all","summary":"string","content":"string","code":{"html":"string","css":"string","js":"string"}}';

const PLAN_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["action", "target", "summary", "content", "code"],
  properties: {
    action: { type: "string", enum: ["read", "write", "answer"] },
    target: { type: "string", enum: ["html", "css", "js", "all"] },
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

function setStatus(text) {
  ui.status.textContent = text;
}

function setThinking(isThinking) {
  ui.thinking.hidden = !isThinking;
  ui.send.textContent = isThinking ? "Thinking..." : "Send";
}

function isReady() {
  return !!state.chromeSession;
}

function syncReadyUi() {
  const ready = isReady();
  ui.prompt.disabled = !ready;
  ui.send.disabled = !ready;
}

function syncInitButtonUi() {
  ui.init.hidden = isReady();
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

function hasWritePayload(plan) {
  if (!plan || plan.action !== "write") return false;
  if (plan.target === "all") {
    return (
      (typeof plan.content === "string" && plan.content.trim().length > 0) ||
      typeof plan.code?.html === "string" ||
      typeof plan.code?.css === "string" ||
      typeof plan.code?.js === "string"
    );
  }
  return (
    (typeof plan.content === "string" && plan.content.trim().length > 0) ||
    (typeof plan.code?.[plan.target] === "string" && plan.code[plan.target].trim().length > 0)
  );
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

function getWritePayload(plan) {
  if (!plan || plan.action !== "write") return { target: null, content: "", codeObject: null };

  if (plan.target === "all") {
    let candidate = {
      html: typeof plan.code?.html === "string" ? plan.code.html : "",
      css: typeof plan.code?.css === "string" ? plan.code.css : "",
      js: typeof plan.code?.js === "string" ? plan.code.js : "",
    };

    // Fallback: some model variants return the full { html, css, js } object inside content.
    if (
      !isLikelyCodeSnippet("html", candidate.html) &&
      !isLikelyCodeSnippet("css", candidate.css) &&
      !isLikelyCodeSnippet("js", candidate.js)
    ) {
      try {
        const parsed = JSON.parse(plan.content || "");
        candidate = {
          html: typeof parsed?.html === "string" ? parsed.html : "",
          css: typeof parsed?.css === "string" ? parsed.css : "",
          js: typeof parsed?.js === "string" ? parsed.js : "",
        };
      } catch {
        // Ignore parse failures and keep evaluating existing candidate fields.
      }
    }

    const hasAnyCode =
      isLikelyCodeSnippet("html", candidate.html) ||
      isLikelyCodeSnippet("css", candidate.css) ||
      isLikelyCodeSnippet("js", candidate.js);
    return hasAnyCode
      ? { target: "all", content: "", codeObject: candidate }
      : { target: null, content: "", codeObject: null };
  }

  const contentCandidate = typeof plan.content === "string" ? plan.content : "";
  const codeCandidate = typeof plan.code?.[plan.target] === "string" ? plan.code[plan.target] : "";

  if (isLikelyCodeSnippet(plan.target, contentCandidate)) {
    return { target: plan.target, content: contentCandidate, codeObject: plan.code };
  }

  if (isLikelyCodeSnippet(plan.target, codeCandidate)) {
    return { target: plan.target, content: codeCandidate, codeObject: plan.code };
  }

  return { target: null, content: "", codeObject: null };
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
  if (target === "all") {
    return [
      "Read current playground code (provider: Chrome built-in AI).",
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
    `Read ${target.toUpperCase()} from the playground (provider: Chrome built-in AI).`,
    "",
    formatStatsLine(target.toUpperCase(), code[key] || ""),
    "",
    `${target.toUpperCase()}:\n${truncate(code[key] || "", 1200)}`,
  ].join("\n");
}

function formatWriteMessage(target, before, after) {
  const changed = listChangedTargets(before, after);
  if (changed.length === 0) {
    return `Write action completed for ${target.toUpperCase()}, but no code differences were detected.`;
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
    "Applied write action using Chrome built-in AI and re-ran the playground.",
    `Updated targets: ${changed.join(", ")}.`,
    "",
    ...sections,
    "",
    "Summary: The requested edits were applied and the result panel was refreshed.",
  ].join("\n\n");
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
    "- For action=write and target=all, fill code.html, code.css, code.js with complete updated code.",
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

function buildWriteRepairPrompt(input, current, priorRaw) {
  return [
    "The previous response indicated a write action but did not include actual updated code.",
    "Return valid JSON only with this schema:",
    PLAN_SHAPE_TEXT,
    "You must include updated code now:",
    "- If target is all: fill code.html, code.css, code.js with complete updated code.",
    "- If target is html/css/js: put full updated code for that target in content.",
    "- Do not leave write payload empty.",
    "",
    `User request: ${input}`,
    "",
    "Current code context:",
    `HTML:\n${current.html}`,
    `CSS:\n${current.css}`,
    `JS:\n${current.js}`,
    "",
    "Previous invalid/insufficient model output:",
    priorRaw,
  ].join("\n");
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

async function getPlanFromChrome(input, current) {
  const plannerPrompt = buildPlannerPrompt(input, current);
  const raw = await promptChrome(plannerPrompt, true);
  return { raw, plan: normalizePlan(extractJson(raw)) };
}

async function repairPlan(rawText) {
  const raw = await promptChrome(buildJsonRepairPrompt(rawText), true);
  return { raw, plan: normalizePlan(extractJson(raw)) };
}

async function requestConcreteWritePlan(input, current, priorRaw) {
  const raw = await promptChrome(buildWriteRepairPrompt(input, current, priorRaw), true);
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

  if (target === "all") {
    try {
      const parsed = codeObject && typeof codeObject === "object" ? codeObject : JSON.parse(content);
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
  return { before: current, after: next };
}

async function handleUserPrompt(input) {
  if (!isReady()) {
    await addAgentMessage("Initialize Chrome built-in AI before sending a message.");
    return;
  }

  const current = await getCurrentCode();
  const firstAttempt = await getPlanFromChrome(input, current);

  let raw = firstAttempt.raw;
  let plan = firstAttempt.plan;

  if (!plan) {
    const repaired = await repairPlan(raw);
    raw = repaired.raw;
    plan = repaired.plan;
  }

  if (plan && plan.action === "write" && !hasWritePayload(plan)) {
    const writeRepair = await requestConcreteWritePlan(input, current, raw);
    raw = writeRepair.raw;
    plan = writeRepair.plan || plan;
  }

  if (!plan) {
    await addAgentMessage("Model response was not valid JSON after retries. No code was changed.");
    await addAgentMessage(truncate(raw));
    return;
  }

  if (plan.action === "read") {
    await addAgentMessage(formatReadMessage(plan.target, current));
    return;
  }

  if (plan.action === "write") {
    const payload = getWritePayload(plan);
    if (!payload.target) {
      await addAgentMessage(
        "Write action was detected, but the returned payload did not look like code. No code was changed."
      );
      return;
    }

    try {
      const result = await applyCodeUpdate(payload.target, payload.content || "", payload.codeObject);
      await addAgentMessage(formatWriteMessage(payload.target, result.before, result.after));
    } catch (error) {
      await addAgentMessage(`Failed to apply code update: ${error.message}`);
    }
    return;
  }

  await addAgentMessage("No code change was applied. The model returned an informational answer.");
}

ui.init.addEventListener("click", async () => {
  ui.init.disabled = true;
  try {
    await initChromeAi();
  } catch (error) {
    setStatus(`AI init failed: ${error.message}`);
  } finally {
    ui.init.disabled = false;
    syncReadyUi();
    syncInitButtonUi();
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

initPlayground().catch((error) => {
  setStatus(`Failed to initialize LiveCodes: ${error.message}`);
});

syncReadyUi();
syncInitButtonUi();
setStatus("Chrome built-in AI not initialized. Click Initialize AI.");
