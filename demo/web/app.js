// Demo frontend: plain JavaScript, no build step, no network access beyond this server.
"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const state = { info: null, examples: [], chunks: [], numbers: [], runs: null, records: [], filter: "all", selected: null };

const VARIANT_LABELS = {
  C_verified: "Full system + answer check (default)",
  C_lean: "Without the answer check",
  C_full: "With router and GLiNER",
  B_dense_rag: "Plain dense RAG",
};
const STATUS_NOTES = {
  quoted: "copied from a passage",
  question: "given in the question",
  constant: "a constant",
  calculated: "a calculation code has redone",
  derived: "one step from checked numbers",
  corrected: "the SLM's arithmetic was wrong; code corrected it",
  unsupported: "no source in the passages",
  wrong_year: "taken from another year's column",
};
const CITATION = /[\[(]\s*(?:pages?|pp?\.?)\s*([0-9][0-9,\s\-–]*?)\s*[\])]/gi;

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

async function api(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json();
}

function fmtMs(ms) {
  if (ms == null) return "";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

function pct(value) {
  return value == null ? "n/a" : `${Math.round(value * 100)}%`;
}

// Section: tabs

$$(".tab").forEach((tab) =>
  tab.addEventListener("click", () => {
    $$(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${tab.dataset.tab}`));
    if (tab.dataset.tab === "eval" && !state.runs) loadRuns();
  })
);

// Section: start-up

async function init() {
  try {
    const info = await api("/api/info");
    state.info = info;
    $("#corpus").textContent = `${info.documents ?? "?"} filings, ${(info.chunks ?? 0).toLocaleString()} chunks, ${info.slm}`;
    const select = $("#variant");
    const order = ["C_verified", "C_lean", "C_full", "B_dense_rag"];
    for (const name of order.filter((v) => info.variants.includes(v))) {
      select.add(new Option(VARIANT_LABELS[name] || name, name, name === info.default_variant, name === info.default_variant));
    }
    state.examples = info.examples;
    showExamples();
  } catch (error) {
    $("#corpus").textContent = "Server not reachable";
  }
}

function showExamples() {
  const picks = [...state.examples].sort(() => Math.random() - 0.5).slice(0, 4);
  $("#examples").innerHTML = picks.length
    ? picks.map((e, i) => `<button type="button" class="example" data-i="${i}"><span class="grp">${escapeHtml((e.group || "").replace("-generated", "").replace("-relevant", ""))}</span><span class="q">${escapeHtml(e.question)}</span></button>`).join("")
    : '<span class="muted small">No example questions on this machine (the evaluation data is not shipped).</span>';
  $$(".example").forEach((button) =>
    button.addEventListener("click", () => {
      $("#question").value = picks[Number(button.dataset.i)].question;
      ask();
    })
  );
}
$("#shuffle").addEventListener("click", showExamples);

// Section: asking

const STAGE_KEYS = {
  understand: ["extract", "tag_section", "route"],
  search: ["retrieve", "fuse"],
  rerank: ["rerank", "gate"],
  write: ["generate"],
  verify: ["verify"],
};

function setStage(name, status, timings) {
  const item = $(`#stages li[data-stage="${name}"]`);
  item.classList.remove("active", "done", "fail", "skip");
  if (status) item.classList.add(status);
  if (timings) {
    const total = STAGE_KEYS[name].reduce((sum, key) => sum + (timings[key] || 0), 0);
    $("em", item).textContent = total ? fmtMs(total) : "";
  }
}

function resetStages() {
  $$("#stages li").forEach((item) => {
    item.className = "";
    $("em", item).textContent = "";
  });
}

let source = null;
$("#ask-form").addEventListener("submit", (event) => {
  event.preventDefault();
  ask();
});
$("#question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) ask();
});

function ask() {
  const question = $("#question").value.trim() || $("#question").placeholder;
  $("#question").value = question;
  if (source) source.close();
  resetStages();
  state.chunks = [];
  state.numbers = [];
  setStage("understand", "active");
  $("#ask-btn").disabled = true;
  $("#tags").innerHTML = '<span class="muted">Reading the question</span>';
  $("#evidence").innerHTML = '<span class="muted">Searching</span>';
  $("#evidence-meta").textContent = "";
  $("#answer-card").className = "answer-card";
  $("#answer-card").innerHTML = '<div class="answer-empty">Working. The first question also loads the models, which takes a little longer.</div>';
  const started = performance.now();
  const url = `/api/ask/stream?question=${encodeURIComponent(question)}&variant=${encodeURIComponent($("#variant").value)}`;
  source = new EventSource(url);

  source.addEventListener("status", (e) => {
    const { message } = JSON.parse(e.data);
    $("#answer-card").innerHTML = `<div class="answer-empty">${escapeHtml(message)}.</div>`;
  });
  source.addEventListener("analyzed", (e) => {
    const data = JSON.parse(e.data);
    renderTags(data.tags);
    setStage("understand", "done", data.timings_ms);
    setStage("search", "active");
  });
  source.addEventListener("retrieved", (e) => {
    const data = JSON.parse(e.data);
    state.chunks = data.chunks;
    renderEvidence();
    $("#evidence-meta").textContent = `from ${data.candidates} candidates${data.widened ? ", widened" : ""}`;
    setStage("search", "done", data.timings_ms);
    setStage("rerank", "done", data.timings_ms);
    setStage("write", "active");
    $("#answer-card").innerHTML = '<div class="answer-empty">The SLM is writing an answer from these five passages.</div>';
  });
  source.addEventListener("generated", (e) => {
    const data = JSON.parse(e.data);
    setStage("write", "done");
    setStage("verify", data.refused ? "skip" : "active");
    $("#answer-card").innerHTML = `<div class="answer-head"><span class="badge plain">Draft from the SLM, checking</span></div><div class="answer-text">${escapeHtml(data.text)}</div>`;
  });
  source.addEventListener("done", (e) => {
    const data = JSON.parse(e.data);
    source.close();
    $("#ask-btn").disabled = false;
    setStage("write", "done", data.timings_ms);
    if (data.verification) setStage("verify", data.verification.ok ? "done" : "fail", data.timings_ms);
    else setStage("verify", "skip");
    renderAnswer(data, performance.now() - started);
  });
  source.addEventListener("error", (e) => {
    let message = "The connection to the server was lost.";
    try { message = JSON.parse(e.data).message; } catch (_) { /* a network error has no data */ }
    source.close();
    $("#ask-btn").disabled = false;
    $$("#stages li.active").forEach((item) => item.classList.replace("active", "fail"));
    $("#answer-card").innerHTML = `<div class="answer-head"><span class="badge refused">Error</span></div><div class="answer-text">${escapeHtml(message)}</div>`;
  });
}

function renderTags(tags) {
  const items = [];
  const add = (name, value) => {
    if (!value) return;
    const [label, confidence] = value;
    items.push(`<div class="tag"><small>${name}</small><b>${escapeHtml(label)} <span class="conf">${Math.round(confidence * 100)}%</span></b></div>`);
  };
  add("company", tags.company);
  add("year", tags.year);
  add("section", tags.section);
  add("intent", tags.intent);
  for (const entity of tags.entities || []) items.push(`<div class="tag"><small>${escapeHtml(entity[1])}</small><b>${escapeHtml(entity[0])}</b></div>`);
  $("#tags").className = "tags";
  $("#tags").innerHTML = items.join("") || '<span class="muted">No confident tags: plain search over all filings.</span>';
}

function renderEvidence() {
  const quotedPages = new Set(state.numbers.filter((n) => n.status === "quoted").map((n) => n.note.replace("page ", "")));
  $("#evidence").className = "evidence";
  $("#evidence").innerHTML = state.chunks
    .map((c, i) => {
      const score = c.score ?? 0;
      const used = quotedPages.has(String(c.page));
      return `<button class="ev" data-i="${i}">
        <div class="ev-head"><b>${i + 1}. ${escapeHtml(c.company || c.doc_id)} ${escapeHtml(c.year || "")}, page ${c.page}</b><span class="muted">${score.toFixed(2)}</span></div>
        <div class="ev-meta"><span class="pill">${escapeHtml(c.type || "text")}</span>${c.section ? `<span class="pill">${escapeHtml(c.section)}</span>` : ""}${c.statement_slot ? '<span class="pill slot">statement slot</span>' : ""}${used ? '<span class="pill gold">numbers used</span>' : ""}</div>
        <div class="bar"><i style="width:${Math.max(3, score * 100)}%"></i></div>
        <div class="ev-snippet">${escapeHtml(c.text.slice(0, 220))}</div>
      </button>`;
    })
    .join("");
  $$("#evidence .ev").forEach((el) => el.addEventListener("click", () => openChunk(state.chunks[Number(el.dataset.i)])));
}

// Section: answer rendering

function highlight(text, numbers) {
  // Spans: verified numbers (marks) and page citations (buttons), in order, without overlaps.
  const spans = (numbers || []).filter((n) => n.end > n.start).map((n) => ({ ...n, kind: "number" }));
  for (const match of text.matchAll(CITATION)) {
    const start = match.index;
    const end = start + match[0].length;
    if (!spans.some((s) => s.start < end && start < s.end)) spans.push({ start, end, kind: "cite", pages: match[1] });
  }
  spans.sort((a, b) => a.start - b.start);
  let html = "";
  let at = 0;
  for (const span of spans) {
    if (span.start < at) continue;
    html += escapeHtml(text.slice(at, span.start));
    const piece = escapeHtml(text.slice(span.start, span.end));
    if (span.kind === "cite") {
      const page = parseInt(span.pages, 10);
      html += `<button class="cite" data-page="${page}" title="Open the passage">p. ${escapeHtml(span.pages.trim())}</button>`;
    } else {
      const note = `${STATUS_NOTES[span.status] || span.status}${span.note ? ": " + span.note : ""}`;
      html += `<mark class="n ${span.status}" data-note="${escapeHtml(note)}">${piece}</mark>`;
    }
    at = span.end;
  }
  return html + escapeHtml(text.slice(at));
}

function bindCitations(root) {
  $$(".cite", root).forEach((button) =>
    button.addEventListener("click", () => {
      const page = Number(button.dataset.page);
      const chunk = state.chunks.find((c) => c.page === page);
      if (chunk) openChunk(chunk);
    })
  );
}

function renderAnswer(data, elapsed) {
  const check = data.verification;
  state.numbers = check ? check.numbers : [];
  renderEvidence();
  let badge = '<span class="badge plain">Answered without the check</span>';
  if (check && check.ok && check.corrections.length) badge = '<span class="badge fixed">Verified after a correction</span>';
  else if (check && check.ok) badge = '<span class="badge ok">Verified: every number traced</span>';
  else if (check) badge = '<span class="badge refused">Refused: could not verify</span>';
  else if (data.refused) badge = '<span class="badge refused">Refused: not enough evidence</span>';

  let body;
  if (check && check.ok) body = `<div class="answer-text">${highlight(check.text, check.numbers)}</div>`;
  else body = `<div class="answer-text ${data.refused ? "refusal" : ""}">${data.refused ? escapeHtml(data.text) : highlight(data.text, [])}</div>`;

  let extra = "";
  if (check && check.corrections.length) {
    extra += `<div class="sub-block"><h4>Arithmetic corrected by code</h4><ul class="corrections">${check.corrections.map((c) => `<li>${escapeHtml(c)}</li>`).join("")}</ul>
      <details><summary>What the SLM wrote</summary><div class="slm-raw">${escapeHtml(data.unverified_answer || "")}</div></details></div>`;
  }
  if (check && !check.ok) {
    extra += `<div class="sub-block"><h4>Why it was not given</h4><ul class="reasons">${check.reasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>
      <details open><summary>What the SLM wrote (not shown to users)</summary><div class="slm-raw">${highlight(check.text, check.numbers)}</div></details></div>`;
  }
  if (check && check.numbers.length) {
    extra += `<div class="legend-inline"><span><mark class="n quoted">n</mark> from a passage</span><span><mark class="n calculated">n</mark> checked calculation</span><span><mark class="n corrected">n</mark> corrected</span><span><mark class="n unsupported">n</mark> no source</span><span>Hover a number for its source.</span></div>`;
  }
  const slm = data.slm ? ` | SLM ${data.slm.slm_output_tokens} tokens` : "";
  $("#answer-card").className = "answer-card";
  $("#answer-card").innerHTML = `<div class="answer-head">${badge}<span class="timing">${fmtMs(elapsed)} total${slm}</span></div>${body}${extra}`;
  bindCitations($("#answer-card"));
}

// Section: passage drawer

function openChunk(chunk) {
  const digits = state.numbers
    .filter((n) => n.status === "quoted" && n.note === `page ${chunk.page}`)
    .map((n) => (n.raw.match(/\d[\d,]*(?:\.\d+)?/) || [""])[0])
    .filter(Boolean);
  let text = escapeHtml(chunk.text);
  for (const value of [...new Set(digits)].sort((a, b) => b.length - a.length)) {
    const pattern = new RegExp(`(?<![\\d.,])${value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?![\\d]|[.,]\\d)`, "g");
    text = text.replace(pattern, (m) => `<mark>${m}</mark>`);
  }
  $("#drawer-body").innerHTML = `<h2>${escapeHtml(chunk.company || chunk.doc_id)} ${escapeHtml(chunk.year || "")}, page ${chunk.page}</h2>
    <p class="muted">${escapeHtml(chunk.doc_id)} | ${escapeHtml(chunk.type || "text")}${chunk.section ? " | " + escapeHtml(chunk.section) : ""}${chunk.score != null ? ` | reranker ${chunk.score.toFixed(2)}` : ""}${digits.length ? " | highlighted: numbers the answer copied" : ""}</p>
    <div class="chunk-text">${text}</div>`;
  $("#drawer").hidden = false;
}
$("#drawer-close").addEventListener("click", () => ($("#drawer").hidden = true));
$("#drawer").addEventListener("click", (e) => { if (e.target.id === "drawer") $("#drawer").hidden = true; });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#drawer").hidden = true; });

// Section: evaluation

async function loadRuns() {
  $("#compare").innerHTML = '<span class="muted">Loading saved runs</span>';
  try {
    state.runs = await api("/api/runs");
  } catch (error) {
    $("#compare").innerHTML = `<span class="muted">Could not load runs: ${escapeHtml(error.message)}</span>`;
    return;
  }
  const featured = state.runs.filter((r) => r.featured);
  const bestPrecision = Math.max(...featured.map((r) => r.precision ?? 0));
  $("#compare").innerHTML =
    `<div class="cmp-row head"><span>System</span><span class="stack-cell">Outcome of 100 questions</span><span class="cmp-num">Accuracy</span><span class="cmp-num">Precision</span><span class="cmp-num hide-small">Refused</span><span class="cmp-num hide-narrow">Wrong</span></div>` +
    featured
      .map((r) => {
        const wrong = r.hallucination_rate;
        const correct = r.accuracy;
        const refused = 1 - correct - wrong;
        return `<div class="cmp-row ${r.precision === bestPrecision ? "best" : ""}">
          <div class="cmp-name">${escapeHtml(r.label)}<small>${escapeHtml(r.name)}</small></div>
          <div class="stack-cell"><div class="stack" title="correct ${pct(correct)}, wrong ${pct(wrong)}, refused ${pct(refused)}"><i class="correct" style="width:${correct * 100}%"></i><i class="wrong" style="width:${wrong * 100}%"></i><i class="refused" style="width:${refused * 100}%"></i></div></div>
          <div class="cmp-num">${pct(correct)}<small>${pct(r.ci95[0])} to ${pct(r.ci95[1])}</small></div>
          <div class="cmp-num"><b>${pct(r.precision)}</b><small>when answered</small></div>
          <div class="cmp-num hide-small">${pct(r.refusal_rate)}</div>
          <div class="cmp-num hide-narrow">${pct(wrong)}</div>
        </div>`;
      })
      .join("");
  const select = $("#run-select");
  select.innerHTML = state.runs.map((r) => `<option value="${escapeHtml(r.name)}">${escapeHtml(r.label)} (${r.n})</option>`).join("");
  const preferred = ["C_verified_test_answer", "C_lean_test_answer_verified"].find((name) => state.runs.some((r) => r.name === name));
  if (preferred) select.value = preferred;
  loadRecords();
}

async function loadRecords() {
  $("#run-rows").innerHTML = '<div class="run-row muted">Loading</div>';
  state.records = await api(`/api/runs/${encodeURIComponent($("#run-select").value)}`);
  renderRows();
}

function renderRows() {
  const term = $("#run-search").value.trim().toLowerCase();
  const rows = state.records.filter((r) => (state.filter === "all" || r.outcome === state.filter) && (!term || r.question.toLowerCase().includes(term)));
  $("#run-rows").innerHTML =
    rows.map((r) => `<div class="run-row ${state.selected === r.id ? "sel" : ""}" data-id="${escapeHtml(r.id)}"><i class="sw ${r.outcome === "ungraded" ? "refused" : r.outcome}"></i><span class="q">${escapeHtml(r.question)}</span></div>`).join("") ||
    '<div class="run-row muted">No questions match.</div>';
  $$("#run-rows .run-row[data-id]").forEach((row) => row.addEventListener("click", () => showRecord(row.dataset.id)));
}

function showRecord(id) {
  state.selected = id;
  renderRows();
  const r = state.records.find((x) => x.id === id);
  const check = r.verification;
  const outcomeBadge = { correct: "ok", wrong: "refused", refused: "plain", ungraded: "plain" }[r.outcome];
  let answer = escapeHtml(r.prediction || "");
  if (check && check.ok && check.numbers) answer = highlight(check.text, check.numbers);
  let checkHtml = "";
  if (check) {
    checkHtml = `<dt>Check</dt><dd>${check.ok ? (check.corrections.length ? "passed after correcting: " + escapeHtml(check.corrections.join("; ")) : "passed") : "failed: " + escapeHtml(check.reasons.join("; "))}</dd>`;
    if (!check.ok) checkHtml += `<dt>SLM wrote</dt><dd>${highlight(check.text, check.numbers)}</dd>`;
    else if (r.unverified && check.corrections.length) checkHtml += `<dt>SLM wrote</dt><dd>${escapeHtml(r.unverified)}</dd>`;
  }
  $("#run-detail").innerHTML = `<div class="answer-head"><span class="badge ${outcomeBadge}">${r.outcome}</span><span class="muted small">${escapeHtml(r.group || "")} | graded by ${escapeHtml(r.method || "n/a")} | evidence in top 5: ${r.evidence_in_top5 ? "yes" : "no"}</span></div>
    <dl class="kv"><dt>Question</dt><dd>${escapeHtml(r.question)}</dd><dt>Gold</dt><dd>${escapeHtml(r.gold)}</dd><dt>Answer</dt><dd>${answer}</dd>${checkHtml}
    <dt>Gold pages</dt><dd>${r.evidence.map((e) => `${escapeHtml(e[0])} p. ${e[1]}`).join(", ") || "none"}</dd>
    <dt>Given pages</dt><dd>${r.chunks.map((c) => `${escapeHtml(c.doc_id || "")} p. ${c.page ?? "?"}`).join(", ")}</dd></dl>`;
}

$("#run-select").addEventListener("change", loadRecords);
$("#run-search").addEventListener("input", renderRows);
$$("#filters .chip").forEach((chip) =>
  chip.addEventListener("click", () => {
    state.filter = chip.dataset.filter;
    $$("#filters .chip").forEach((c) => c.classList.toggle("active", c === chip));
    renderRows();
  })
);

init();
