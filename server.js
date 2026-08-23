// server.js — OpenAI-compatible proxy for NVIDIA NIM
// Express 5 compatible.

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const { StringDecoder } = require('string_decoder');
const { timingSafeEqual } = require('crypto');

const app = express();
const PORT = process.env.PORT || 3000;

// ─── Configuration ───────────────────────────────────────────────────────────

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY;
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';
const SKIP_VALIDATION = process.env.SKIP_VALIDATION === 'true';
const DISCORD_WEBHOOK_URL = process.env.DISCORD_WEBHOOK_URL;

const MAX_TOKENS_LIMIT = 65536;
const REQUEST_TIMEOUT_MS = 180000;
const VALIDATION_TIMEOUT_MS = 15000;
const MAX_BUFFER_SIZE = 1024 * 1024; // 1MB

if (ENABLE_THINKING_MODE) console.log('[CONFIG] Thinking mode: ENABLED');

// ─── Config validation ──────────────────────────────────────────────────────

function validateConfig() {
  const fatal = (msg) => { console.error(`[FATAL] ${msg}`); process.exit(1); };
  if (!NIM_API_KEY) fatal('NIM_API_KEY is required. Get one at https://build.nvidia.com/');
  if (!CLIENT_AUTH_KEY) {
    console.warn('[WARN] CLIENT_AUTH_KEY not set. All requests will be rejected with 403.');
  }
}
validateConfig();

// ─── Model Mapping ─────────────────────────────────────────────────────────

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/nemotron-3-super-120b-a12b',
  'gpt-4': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-3.5': 'qwen/qwen3.5-397b-a17b',
  'gpt-4-turbo': 'moonshotai/kimi-k2.6',
  'gpt-4o': 'deepseek-ai/deepseek-v4-flash-0731',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'gemini-turbo': 'meta/llama-3.3-70b-instruct',
  'gemini-turbo?': 'abacusai/dracarys-llama-3.1-70b-instruct',
  'gpt-3.5o': 'nvidia/nemotron-mini-4b-instruct',
  'gpt-4-flash': 'deepseek-ai/deepseek-v4-flash-0731',
  'glm-5.2': 'z-ai/glm-5.2',
  'mistral': 'mistralai/mistral-large-3-675b-instruct-2512',
  'mistral-turbo': 'mistralai/mistral-medium-3.5-128b',
  'mistral-pro': 'mistralai/mistral-small-4-119b-2603',
  'mistral-nemo': 'mistralai/mistral-nemotron',
  'mistral-fast': 'mistralai/ministral-14b-instruct-2512',
  'google-light': 'google/gemma-4-31b-it',
  'google-lightest': 'google/gemma-2-2b-it',
  'google-lighter': 'google/gemma-3n-e4b-it',
  'm2.7': 'minimaxai/minimax-m2.7',
  'm3': 'minimaxai/minimax-m3',
  'step-3.5-flash': 'stepfun-ai/step-3.5-flash',
  'step-3.7-flash': 'stepfun-ai/step-3.7-flash'
};

// Default model used when an unrecognized alias is requested.
const DEFAULT_MODEL = 'nvidia/llama-3.3-nemotron-super-49b-v1.5';

const FALLBACK_MODELS = [
  'mistralai/mistral-medium-3.5-128b',
  'mistralai/mistral-small-4-119b-2603',
  'nvidia/llama-3.3-nemotron-super-49b-v1.5',
  'google/gemma-4-31b-it'
];

// ─── Reasoning subsystem ────────────────────────────────────────────────────
// Owns: which chat_template_kwargs/top-level fields each backend model needs
// to control thinking, and how to pull reasoning text back out of responses
// that embed it inline vs. as a structured field.

const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
if (SHOW_REASONING) console.log('[CONFIG] Reasoning display: ENABLED');

// ─── Reasoning subsystem notes ─────────────────────────────────────────────
// Reasoning/thinking parameters vary by backend model and aren't part of the
// OpenAI schema, so they can't just be forwarded as-is — getReasoningPayload()
// below maps each backend model to its own request shape.
//
// Everything getReasoningPayload() returns is spread directly into the
// top-level JSON body sent to NIM via axios. Do NOT wrap it in an
// `extra_body` key — that's an openai-SDK-only convention, unwrapped
// client-side by the official SDKs and merged into the outgoing JSON as
// top-level fields. This proxy talks to NIM's raw REST endpoint via axios
// directly, so a literal "extra_body" field is just silently ignored.
// Confirmed against NVIDIA's own curl docs, which send chat_template_kwargs
// directly at the top level.
//
// GLM models think by default. `reasoning_effort` only controls thinking
// *intensity* once thinking is already happening — it does NOT turn thinking
// off. The actual on/off switch is the top-level `thinking: { type: "enabled"
// | "disabled" }` field (per z.ai's docs). GLM-5.2 accepts only "high" or
// "max" for reasoning_effort, with "max" as the default.
//
// nemotron-3-ultra's `force_nonempty_content` flag is NOT a confirmed NVIDIA
// parameter — left in as opt-in/best-effort since unrecognized
// chat_template_kwargs are typically ignored by the backend rather than
// causing a hard failure.
//
// nemotron-3-super and nemotron-3-ultra also expose a `low_effort: true`
// chat_template_kwargs flag — a middle ground between full reasoning and off,
// but a fixed tier, not a self-deciding mode. Reachable by sending
// reasoning_effort: "low" on a request.
//
// MiniMax-M3 controls reasoning via chat_template_kwargs.thinking_mode:
// "enabled" | "disabled" | "adaptive" (confirmed against NVIDIA's own NIM
// API reference for this model). "adaptive" is the only genuinely
// self-deciding reasoning mode in this proxy — the model chooses whether to
// think per-turn — and is reachable by sending reasoning_effort: "adaptive".
// M3 also emits its reasoning inline in content wrapped in
// <mm:think>...</mm:think> — a different tag than the generic <think> used
// by qwen/nemotron-super — so it needs its own entry in
// CONTENT_DELIMITER_TAGS or the tags leak straight into content unparsed.
//
// google/gemma-4-31b-it needs TWO separate flags to actually see reasoning
// output: chat_template_kwargs.enable_thinking turns thinking on, but the
// `reasoning` field is only populated in the response if include_reasoning
// is ALSO sent as true at the top level (confirmed against NVIDIA's VLM
// NIM docs). Sending enable_thinking alone makes the model reason internally
// with nothing to show for it.
//
// DeepSeek V4 Flash 0731 uses NVIDIA's hosted API `reasoning_effort` field.
// The current API accepts "none", "high", and "max". NVIDIA translates this
// field into the appropriate model chat-template reasoning configuration.
//
// Reasoning output format: by default, reasoning is kept out of `content`
// and returned in a structured `reasoning`/`reasoning_content` field.
// Clients that expect legacy inline <thinking> tags baked into content can
// opt in by sending the `x-reasoning-format: inline` header.

// Backend models that embed reasoning inline in `content` via delimiter tags,
// rather than returning it as a separate structured field. Mapped to their
// specific tag pair so DelimiterParser knows what to look for.
const CONTENT_DELIMITER_TAGS = {
  'qwen/qwen3.5-397b-a17b': ['<think>', '</think>'],
  'nvidia/llama-3.3-nemotron-super-49b-v1.5': ['<think>', '</think>'],
  // MiniMax-M3 uses its own namespaced tag, not the generic <think> one.
  'minimaxai/minimax-m3': ['<mm:think>', '</mm:think>']
};

// Pure, stateful string parser for extracting reasoning blocks across chunks.
class DelimiterParser {
  constructor(openTag, closeTag) {
    this.openTag = openTag;
    this.closeTag = closeTag;
    this.inThinking = false;
    this.buffer = '';
  }

  processChunk(chunk) {
    this.buffer += chunk;
    let content = '';
    let reasoning = '';

    while (true) {
      const targetTag = this.inThinking ? this.closeTag : this.openTag;
      const tagIndex = this.buffer.indexOf(targetTag);

      if (tagIndex !== -1) {
        const textBefore = this.buffer.substring(0, tagIndex);
        if (this.inThinking) {
          reasoning += textBefore;
        } else {
          content += textBefore;
        }
        this.inThinking = !this.inThinking;
        this.buffer = this.buffer.substring(tagIndex + targetTag.length);
      } else {
        // Check for partial tag at the end
        let partialLen = 0;
        const maxLen = Math.min(this.buffer.length, targetTag.length - 1);
        for (let i = maxLen; i > 0; i--) {
          if (targetTag.startsWith(this.buffer.substring(this.buffer.length - i))) {
            partialLen = i;
            break;
          }
        }
        const textBefore = this.buffer.substring(0, this.buffer.length - partialLen);
        if (this.inThinking) {
          reasoning += textBefore;
        } else {
          content += textBefore;
        }
        this.buffer = this.buffer.substring(this.buffer.length - partialLen);
        break;
      }
    }

    return { content, reasoning };
  }

  flush() {
    let content = '';
    let reasoning = '';
    if (this.buffer) {
      if (this.inThinking) {
        reasoning += this.buffer;
      } else {
        content += this.buffer;
      }
      this.buffer = '';
    }
    return { content, reasoning };
  }
}

// Normalizes structured reasoning fields and extracts content delimiters.
class StreamNormalizer {
  constructor(model) {
    this.model = model;
    this.parser = null;
    // ONLY use content delimiters for models that embed reasoning in content
    const tags = CONTENT_DELIMITER_TAGS[model];
    if (tags) {
      this.parser = new DelimiterParser(tags[0], tags[1]);
    }
    // Models like Gemma 4, DeepSeek, GPT-OSS use structured fields and are NOT parsed here.
  }

  processDelta(delta) {
    const normalizedDelta = { ...delta };
    let reasoning = normalizedDelta.reasoning || normalizedDelta.reasoning_content || '';
    let content = normalizedDelta.content || '';

    // Priority: Structured reasoning > Content delimiters
    if (!reasoning && content && this.parser) {
      const parsed = this.parser.processChunk(content);
      reasoning = parsed.reasoning;
      content = parsed.content;
    }

    if (content) normalizedDelta.content = content;
    else delete normalizedDelta.content;

    if (reasoning) normalizedDelta.reasoning = reasoning;
    else delete normalizedDelta.reasoning;

    delete normalizedDelta.reasoning_content;
    return normalizedDelta;
  }

  flush() {
    if (!this.parser) return { content: '', reasoning: '' };
    return this.parser.flush();
  }
}

function normalizeNonStreamChoice(choice, model) {
  if (!choice) return choice;
  const message = choice.message || {};
  let reasoning = message.reasoning || message.reasoning_content || '';
  let content = message.content || '';

  if (!reasoning && content) {
    let parser = null;
    const tags = CONTENT_DELIMITER_TAGS[model];
    if (tags) {
      parser = new DelimiterParser(tags[0], tags[1]);
    }
    if (parser) {
      const parsed = parser.processChunk(content);
      const flushed = parser.flush();
      content = (parsed.content || '') + (flushed.content || '');
      reasoning = (parsed.reasoning || '') + (flushed.reasoning || '');
    }
  }

  const newMessage = { ...message };
  if (content) newMessage.content = content;
  if (reasoning) newMessage.reasoning = reasoning;
  delete newMessage.reasoning_content;

  return { ...choice, message: newMessage };
}

// Valid reasoning_effort values per backend model, where the backend enforces
// an enum. Anything outside this set is dropped rather than forwarded, so a
// bad client value fails fast in proxy logs instead of as an opaque upstream 400.
const REASONING_EFFORT_ENUMS = {
  'openai/gpt-oss-120b': ['low', 'medium', 'high'],
  'openai/gpt-oss-20b': ['low', 'medium', 'high'],
  'mistralai/mistral-medium-3.5-128b': ['high', 'none'],
  'mistralai/mistral-small-4-119b-2603': ['high', 'none'],
  'z-ai/glm-5.2': ['high', 'max'],

  // Current NVIDIA API values for DeepSeek V4 Flash 0731.
  'deepseek-ai/deepseek-v4-flash-0731': ['none', 'high', 'max'],

  // Not a true adaptive/effort scale — these two only expose a single extra
  // "low_effort" middle tier between full reasoning and off.
  'nvidia/nemotron-3-super-120b-a12b': ['low'],
  'nvidia/nemotron-3-ultra-550b-a55b': ['low'],

  // MiniMax-M3's only non-binary option: let the model decide per-turn.
  'minimaxai/minimax-m3': ['adaptive']
};

function validReasoningEffort(model, effort) {
  const allowed = REASONING_EFFORT_ENUMS[model];
  if (!allowed) return effort; // no enum enforced for this model, pass through
  if (allowed.includes(effort)) return effort;
  if (effort) {
    console.warn(`[REASONING] Dropping invalid reasoning_effort "${effort}" for ${model} (allowed: ${allowed.join(', ')})`);
  }
  return undefined;
}

// Pure function returning model-specific reasoning request payloads.
// IMPORTANT: everything returned here gets spread DIRECTLY into the top-level
// JSON body sent to NIM via axios. Do NOT wrap anything in an `extra_body` key —
// see the reasoning subsystem notes above.
//
// UNIVERSAL CLIENT OVERRIDE: reasoning_effort: "off" / "on" forces thinking
// off/on for that one request, regardless of the server's ENABLE_THINKING_MODE
// default. This exists because, before it was added, the on/off decision for
// 8 of the 10 reasoning models below (everything except mistral's accidental
// "none" and M3's "adaptive") was wired to the server env var ONLY — a
// client-side reasoning toggle in any chat UI had no field it could send that
// actually changed anything for those models. "off"/"on" are stripped before
// running the model-specific effort enum check below, so they never collide
// with a real per-model value like "high" or "adaptive".
//
// Caveat: gpt-oss models structurally always emit a reasoning channel — this
// can reduce them to their baseline default but can't literally eliminate
// reasoning tokens the way it can for every other model here.
function getReasoningPayload(model, enableThinking, clientReasoningEffort, hasTools) {
  if (clientReasoningEffort === 'off') enableThinking = false;
  else if (clientReasoningEffort === 'on') enableThinking = true;

  const rawEffort = (clientReasoningEffort === 'off' || clientReasoningEffort === 'on')
    ? undefined
    : clientReasoningEffort;
  const effort = validReasoningEffort(model, rawEffort);

  switch (model) {
    case 'nvidia/nemotron-3-super-120b-a12b': {
      if (!enableThinking) return {};
      const payload = { chat_template_kwargs: { enable_thinking: true } };
      if (effort === 'low') payload.chat_template_kwargs.low_effort = true;
      return payload;
    }

    case 'nvidia/nemotron-3-ultra-550b-a55b': {
      if (!enableThinking) return {};
      const payload = { chat_template_kwargs: { enable_thinking: true } };
      if (effort === 'low') payload.chat_template_kwargs.low_effort = true;
      // Unverified param — see header comment. Left as opt-in best-effort.
      if (hasTools) payload.chat_template_kwargs.force_nonempty_content = true;
      return payload;
    }

    case 'qwen/qwen3.5-397b-a17b': {
      // Model appears to default to thinking-on in its chat template. Only send
      // a field when the caller explicitly wants thinking OFF; otherwise let the
      // <think> delimiter parser handle whatever the model does natively.
      if (enableThinking) return {};
      return { chat_template_kwargs: { enable_thinking: false } };
    }

    case 'deepseek-ai/deepseek-v4-flash-0731': {
      // NVIDIA's current hosted API exposes reasoning_effort directly.
      // Allowed values are "none", "high", and "max".
      if (!enableThinking || effort === 'none') {
        return { reasoning_effort: 'none' };
      }

      return {
        reasoning_effort: effort || 'high'
      };
    }

    case 'openai/gpt-oss-120b':
    case 'openai/gpt-oss-20b': {
      if (effort) return { reasoning_effort: effort };
      if (enableThinking) return { reasoning_effort: 'high' };
      return {};
    }

    case 'mistralai/mistral-medium-3.5-128b':
    case 'mistralai/mistral-small-4-119b-2603': {
      if (effort) return { reasoning_effort: effort };
      if (enableThinking) return { reasoning_effort: 'high' };
      return {};
    }

    case 'z-ai/glm-5.2': {
      // GLM-5.2 thinks by default. `reasoning_effort` only controls
      // intensity (max vs high) once thinking is already happening — it does
      // NOT turn thinking off. The actual on/off switch is `thinking.type`.
      const payload = {
        thinking: { type: enableThinking ? 'enabled' : 'disabled' }
      };
      if (enableThinking && effort) payload.reasoning_effort = effort;
      return payload;
    }

    case 'google/gemma-4-31b-it': {
      if (!enableThinking) return {};
      // enable_thinking only makes the model reason internally — it does NOT
      // by itself put that reasoning in the response. NVIDIA's own VLM docs
      // require a separate top-level include_reasoning flag to actually
      // return the `reasoning` field; without it we may be paying the
      // latency/token cost of thinking and never seeing the output. Match it
      // to SHOW_REASONING so behavior is explicit instead of relying on
      // whatever include_reasoning defaults to upstream.
      return {
        chat_template_kwargs: { enable_thinking: true },
        include_reasoning: SHOW_REASONING
      };
    }

    case 'stepfun-ai/step-3.7-flash': {
      if (enableThinking) return {};
      return { chat_template_kwargs: { thinking: false } };
    }

    case 'minimaxai/minimax-m3': {
      // Per NVIDIA's own NIM API reference, MiniMax-M3 controls reasoning via
      // chat_template_kwargs.thinking_mode: "enabled" | "disabled" | "adaptive".
      // "adaptive" lets the model decide per-turn whether to think — the only
      // genuinely self-deciding reasoning mode across every model in this
      // proxy. Send reasoning_effort: "adaptive" on a request to use it;
      // otherwise this falls back to the standard on/off toggle like every
      // other model here.
      const thinkingMode = effort === 'adaptive'
        ? 'adaptive'
        : (enableThinking ? 'enabled' : 'disabled');
      return { chat_template_kwargs: { thinking_mode: thinkingMode } };
    }

    default:
      // Default reasoning models (Kimi, MiniMax, etc.) or non-reasoning models
      return {};
  }
}

// ─── Tool-call recovery subsystem ──────────────────────────────────────────
// Some backend models intermittently fail to convert their native tool-call
// output into NIM's structured `tool_calls` field and instead dump the raw
// Hermes-style tag straight into `content` as plain text:
//   <tool_call>
//   {"name": "...", "arguments": {...}}
//   </tool_call>
//
// This is a CONFIRMED upstream bug, not something wrong with this proxy's
// request shape:
//   - vllm-project/vllm#48095 reproduces this exact failure with GLM-5.2
//     (vLLM's own `glm47` tool-call parser fails and the call is written
//     unparsed into content), specifically under tool_choice: "required".
//   - zai-org/GLM-5#15 and two independent OpenCode bug reports show the
//     same GLM-5/5.2-via-NIM leak, intermittent and worse on long contexts
//     with many tool calls in a session.
//   - zed-industries/zed#55884 reproduces the identical failure mode with
//     nvidia/nemotron-3-super-120b-a12b via integrate.api.nvidia.com ("the
//     first few tool calls go correctly... later, the tool calling code is
//     seen in output directly") — so this isn't GLM-specific, it can happen
//     with any model in MODEL_MAPPING/FALLBACK_MODELS.
//
// Because it's an intermittent upstream parser failure rather than a model
// that flatly lacks tool-call support, it can't be fixed by routing around
// a specific model — it has to be caught and repaired wherever it happens.
// This recovery layer runs unconditionally (streaming and non-streaming) and
// is a no-op with negligible overhead when the tag never appears.

const TOOL_CALL_OPEN = '<tool_call>';
const TOOL_CALL_CLOSE = '</tool_call>';

function generateToolCallId() {
  return `call_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
}

// Non-streaming: extracts every <tool_call>{...}</tool_call> block from a
// complete content string. Returns { content, toolCalls } — content has the
// recovered blocks stripped out, toolCalls is an array of OpenAI-shaped
// tool_calls objects (possibly empty).
//
// Malformed JSON inside a tag (also a documented GLM-5.2 symptom — truncated
// args, missing closing brace, per the zai-org/GLM-5#15 reports) is left in
// place rather than silently dropped, so worst case the raw tag still reaches
// the client instead of vanishing without a trace.
function extractLeakedToolCalls(content) {
  if (!content || !content.includes(TOOL_CALL_OPEN)) {
    return { content, toolCalls: [] };
  }

  const toolCalls = [];
  let result = '';
  let cursor = 0;

  while (true) {
    const openIdx = content.indexOf(TOOL_CALL_OPEN, cursor);
    if (openIdx === -1) {
      result += content.slice(cursor);
      break;
    }
    const closeIdx = content.indexOf(TOOL_CALL_CLOSE, openIdx);
    if (closeIdx === -1) {
      // Unterminated tag (e.g. truncated at max_tokens) — leave it as-is
      // rather than guess at a repair.
      result += content.slice(cursor);
      break;
    }

    result += content.slice(cursor, openIdx);
    const inner = content.slice(openIdx + TOOL_CALL_OPEN.length, closeIdx).trim();

    try {
      const parsed = JSON.parse(inner);
      if (parsed && typeof parsed.name === 'string') {
        toolCalls.push({
          id: generateToolCallId(),
          type: 'function',
          function: {
            name: parsed.name,
            arguments: JSON.stringify(parsed.arguments ?? {})
          }
        });
      } else {
        // Well-formed JSON but not the shape we expect — keep it visible.
        result += content.slice(openIdx, closeIdx + TOOL_CALL_CLOSE.length);
      }
    } catch {
      console.warn('[TOOL_CALL_RECOVERY] Malformed JSON inside <tool_call> tag, leaving raw:', inner.slice(0, 200));
      result += content.slice(openIdx, closeIdx + TOOL_CALL_CLOSE.length);
    }

    cursor = closeIdx + TOOL_CALL_CLOSE.length;
  }

  return { content: result, toolCalls };
}

// Streaming: stateful, cross-chunk recovery for the same leak. Built on top
// of DelimiterParser — the same battle-tested cross-chunk tag-splitting
// logic already used above for <think>/<mm:think> — rather than a second
// bespoke parser, since a <tool_call> tag can just as easily be split across
// SSE chunk boundaries as a <think> tag can.
class ToolCallStreamRecovery {
  constructor() {
    this.parser = new DelimiterParser(TOOL_CALL_OPEN, TOOL_CALL_CLOSE);
    this.pending = '';
    this.toolCallIndex = 0;
  }

  // text: already-normalized content for this chunk (post reasoning-split).
  // Returns { content, toolCallDelta } — content is safe to forward to the
  // client as-is, toolCallDelta is a ready OpenAI tool_calls delta or null.
  process(text) {
    const { content, reasoning } = this.parser.processChunk(text);
    let outContent = content;

    if (reasoning) this.pending += reasoning;

    // pending non-empty + parser no longer inside the tag == a close tag was
    // just resolved in this call (pending is always cleared immediately
    // below, so this state can only be freshly true).
    if (this.pending && !this.parser.inThinking) {
      const inner = this.pending.trim();
      this.pending = '';
      try {
        const parsed = JSON.parse(inner);
        if (parsed && typeof parsed.name === 'string') {
          return {
            content: outContent,
            toolCallDelta: {
              index: this.toolCallIndex++,
              id: generateToolCallId(),
              type: 'function',
              function: { name: parsed.name, arguments: JSON.stringify(parsed.arguments ?? {}) }
            }
          };
        }
        outContent = TOOL_CALL_OPEN + inner + TOOL_CALL_CLOSE + outContent;
      } catch {
        console.warn('[TOOL_CALL_RECOVERY] Malformed JSON inside streamed <tool_call> tag, leaving raw:', inner.slice(0, 200));
        outContent = TOOL_CALL_OPEN + inner + TOOL_CALL_CLOSE + outContent;
      }
    }

    return { content: outContent, toolCallDelta: null };
  }

  // Stream ended while still mid-tag (cut off before the closing tag
  // arrived, e.g. hit max_tokens) — return the raw partial buffer as text
  // instead of silently dropping it.
  flush() {
    const flushed = this.parser.flush();
    let leftover = flushed.content || '';
    const tailInner = this.pending + (flushed.reasoning || '');
    if (tailInner) {
      leftover += TOOL_CALL_OPEN + tailInner;
    }
    this.pending = '';
    return leftover;
  }
}

// ─── Middleware ─────────────────────────────────────────────────────────────

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Catch malformed JSON bodies so clients get a clean OpenAI-style error
// instead of Express's default HTML error page. Without this, a broken
// request body never reaches the route handler's try/catch at all — Express
// throws during body parsing, before routing happens.
app.use((err, req, res, next) => {
  if (err && err.type === 'entity.parse.failed') {
    return res.status(400).json({
      error: {
        message: 'Invalid JSON in request body',
        type: 'invalid_request_error',
        code: 400
      }
    });
  }
  next(err);
});

// Extract token AFTER "Bearer " prefix, compare only the token
function extractBearerToken(authHeader) {
  if (!authHeader || typeof authHeader !== 'string') return null;
  const parts = authHeader.trim().split(' ');
  if (parts.length !== 2 || parts[0] !== 'Bearer') return null;
  return parts[1];
}

function safeTimingEqual(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  try {
    return timingSafeEqual(Buffer.from(a), Buffer.from(b));
  } catch {
    return false;
  }
}

app.use((req, res, next) => {
  if (req.path === '/health' || req.path === '/v1/models') {
    return next();
  }

  const token = extractBearerToken(req.headers.authorization);
  if (!token || !CLIENT_AUTH_KEY) {
    return res.status(403).json({
      error: {
        message: 'Forbidden: Invalid or missing authentication',
        type: 'authentication_error',
        code: 403
      }
    });
  }

  if (!safeTimingEqual(token, CLIENT_AUTH_KEY)) {
    return res.status(403).json({
      error: {
        message: 'Forbidden: Invalid authentication credentials',
        type: 'authentication_error',
        code: 403
      }
    });
  }

  next();
});

// ─── Validation ─────────────────────────────────────────────────────────────

async function validateModels() {
  if (SKIP_VALIDATION) {
    console.log('[VALIDATION] Skipped (SKIP_VALIDATION=true)');
    return;
  }

  console.log('[VALIDATION] Checking model availability via /v1/models...');
  try {
    const response = await axios.get(`${NIM_API_BASE}/models`, {
      headers: {
        Authorization: `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      timeout: VALIDATION_TIMEOUT_MS
    });

    const availableModels = new Set(
      (response.data.data || []).map(m => m.id)
    );

    const invalid = [];
    for (const [alias, nimId] of Object.entries(MODEL_MAPPING)) {
      if (availableModels.has(nimId)) {
        console.log(`[VALIDATION] ✓ ${alias} → ${nimId}`);
      } else {
        console.warn(`[VALIDATION] ✗ ${alias} → ${nimId} (not in catalog)`);
        invalid.push({ alias, nimId, error: 'Model not found in NIM catalog' });
      }
    }

    if (invalid.length > 0) {
      await sendDiscordAlert(invalid);
    } else {
      console.log('[VALIDATION] All models valid.');
    }
  } catch (err) {
    console.warn(`[VALIDATION] /v1/models endpoint failed: ${err.message}. Skipping validation.`);
    console.warn('[VALIDATION] Consider setting SKIP_VALIDATION=true if your NIM provider lacks a model listing endpoint.');
  }
}

async function sendDiscordAlert(invalidModels) {
  if (!DISCORD_WEBHOOK_URL) return;

  const embed = {
    title: '⚠️ NIM Proxy: Model Validation Failed',
    description: `${invalidModels.length} model(s) failed validation. Check NIM catalog for deprecations.`,
    color: 0xff4444,
    timestamp: new Date().toISOString(),
    fields: invalidModels.map(m => ({
      name: `\`${m.alias}\``,
      value: `Backend: \`${m.nimId}\`\nError: \`${m.error}\``,
      inline: true
    }))
  };

  try {
    await axios.post(DISCORD_WEBHOOK_URL, {
      embeds: [embed],
      username: 'NIM Proxy Monitor'
    }, { timeout: 5000 });
    console.log('[DISCORD] Alert sent.');
  } catch (err) {
    console.error('[DISCORD] Failed to send alert:', err.message);
  }
}

// ─── Helper: Safe Stream Writing ───────────────────────────────────────────

function safeWrite(res, data) {
  try {
    if (!res.writableEnded && !res.destroyed && res.writable) {
      res.write(data);
      return true;
    }
  } catch (err) {
    console.warn('[STREAM] Write failed:', err.message);
  }
  return false;
}

// ─── Helper: Fallback Chain ─────────────────────────────────────────────────

async function callWithFallback(baseRequest, models, enableThinking, clientReasoningEffort, hasTools) {
  let lastError = null;
  const debugReasoning = process.env.DEBUG_REASONING === 'true';

  for (const model of models) {
    try {
      const reasoningPayload = getReasoningPayload(model, enableThinking, clientReasoningEffort, hasTools);
      if (debugReasoning) {
        console.log(`[REASONING] Attempting ${model} with payload:`, JSON.stringify(reasoningPayload));
      }
      const res = await axios.post(
        `${NIM_API_BASE}/chat/completions`,
        { ...baseRequest, model, ...reasoningPayload },
        {
          headers: {
            Authorization: `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          responseType: baseRequest.stream ? 'stream' : 'json',
          timeout: REQUEST_TIMEOUT_MS
        }
      );
      return { response: res, model };
    } catch (err) {
      lastError = err;
      console.warn(
        `[FALLBACK] Model failed: ${model}`,
        err.response?.status,
        err.response?.data?.error?.message || err.message
      );
    }
  }

  throw lastError || new Error('All models failed');
}

// ─── Routes ────────────────────────────────────────────────────────────────

app.get('/health', (req, res) => {
  res.json({ status: 'ok', version: '2.4.0' });
});

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(id => ({
      id,
      object: 'model',
      // OpenAI's spec documents this field in Unix seconds, not milliseconds
      // — Date.now() alone is 1000x too large and inconsistent with the
      // correctly-converted timestamp used below in the chat completions
      // response.
      created: Math.floor(Date.now() / 1000),
      owned_by: 'nim-proxy'
    }))
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  let streamEndedCleanly = false;
  let upstreamStream = null;

  try {
    const {
      model,
      messages,
      temperature,
      max_tokens,
      stream,
      tools,
      tool_choice
    } = req.body;

    let primaryModel = MODEL_MAPPING[model];
    if (!primaryModel) {
      console.warn(`[PROXY] Unknown model alias "${model}", falling back to default: ${DEFAULT_MODEL}`);
      primaryModel = DEFAULT_MODEL;
    }

    // De-dupe in case the requested alias resolves to a model that's also in
    // the fallback chain — otherwise a failure retries the identical model
    // twice before actually diversifying.
    const modelChain = [...new Set([primaryModel, ...FALLBACK_MODELS])];

    const baseRequest = {
      messages,
      temperature: temperature ?? 0.7,
      max_tokens: Math.min(max_tokens ?? 2048, MAX_TOKENS_LIMIT),
      stream: stream || false,
      // Forward tool-calling fields as-is. Without this, clients using
      // function/tool calling silently get a plain chat completion back —
      // NIM never sees the tool definitions, so it never returns tool_calls.
      ...(tools && { tools }),
      ...(tool_choice && { tool_choice })
    };

    const { response, model: usedModel } = await callWithFallback(
      baseRequest,
      modelChain,
      ENABLE_THINKING_MODE,
      req.body.reasoning_effort,
      !!tools
    );

    upstreamStream = response.data;
    console.log('[PROXY] Model used:', usedModel);

    // Determine if the client wants legacy inline <thinking> tags in the content stream
    const inlineReasoning = req.headers['x-reasoning-format'] === 'inline';

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      const decoder = new StringDecoder('utf8');
      let buffer = '';
      let reasoningOpen = false;
      let doneSent = false;
      let cleanedUp = false;
      const normalizer = new StreamNormalizer(usedModel);
      // See "Tool-call recovery subsystem" above — catches models (GLM-5.2,
      // nemotron-3-super confirmed) that leak tool calls into content as a
      // raw <tool_call> tag instead of NIM's structured tool_calls field.
      const toolRecovery = new ToolCallStreamRecovery();

      const cleanup = () => {
        if (cleanedUp) return;
        cleanedUp = true;
        if (upstreamStream) {
          upstreamStream.removeAllListeners();
        }
        req.removeAllListeners('close');
      };

      const processLine = (line) => {
        if (!line.startsWith('data: ')) return;

        // Exact match only. .includes() would false-positive on any legitimate
        // model output that happens to contain the literal substring "[DONE]"
        // in its generated content (e.g. a coding/task-tracking response),
        // silently truncating the reply right there.
        if (line.trim() === 'data: [DONE]') {
          if (!doneSent) {
            safeWrite(res, 'data: [DONE]\n\n');
            doneSent = true;
          }
          streamEndedCleanly = true;
          return;
        }

        try {
          const data = JSON.parse(line.slice(6));
          const delta = data.choices?.[0]?.delta;

          if (delta) {
            const normalizedDelta = normalizer.processDelta(delta);
            let clientContent = '';

            if (SHOW_REASONING && inlineReasoning) {
              // Inline reasoning format: bake <thinking> tags into content
              // for clients that don't parse structured reasoning fields.
              if (normalizedDelta.reasoning && !reasoningOpen) {
                clientContent += `<thinking>\n${normalizedDelta.reasoning}`;
                reasoningOpen = true;
              } else if (normalizedDelta.reasoning) {
                clientContent += normalizedDelta.reasoning;
              }

              if (normalizedDelta.content && reasoningOpen) {
                clientContent += `\n</thinking>\n\n${normalizedDelta.content}`;
                reasoningOpen = false;
              } else if (normalizedDelta.content) {
                clientContent += normalizedDelta.content;
              }
            } else {
              // Default behavior: clean content, no inline tags
              clientContent = normalizedDelta.content || '';
            }

            // Recover tool calls the backend leaked into content as a raw
            // <tool_call> tag instead of a structured field (confirmed
            // upstream bug — see subsystem notes above). Must run on the
            // final clientContent (post reasoning-split, post inline-tag
            // formatting) since that's the actual text stream being built.
            const { content: recoveredContent, toolCallDelta } = toolRecovery.process(clientContent);
            clientContent = recoveredContent;
            if (toolCallDelta) {
              delta.tool_calls = [toolCallDelta];
              // Force the signal even if upstream's own finish_reason on this
              // chunk was null/'stop' — a recovered tool call means the
              // model's actual intent was a tool_calls turn, and
              // OpenAI-compatible clients key off this field to decide
              // whether to execute a tool vs. treat the turn as finished prose.
              if (data.choices[0]) data.choices[0].finish_reason = 'tool_calls';
            }

            delta.content = clientContent;

            // Keep a structured reasoning field alongside inline tags in
            // content. Some clients parse the inline <thinking> tags;
            // others (OpenRouter-style apps) look for a separate
            // `reasoning`/`reasoning_content` field to render their own
            // collapsible thinking UI. Send both so either style works.
            if (SHOW_REASONING && normalizedDelta.reasoning) {
              delta.reasoning = normalizedDelta.reasoning;
              delta.reasoning_content = normalizedDelta.reasoning;
            } else {
              delete delta.reasoning;
              delete delta.reasoning_content;
            }
          }

          safeWrite(res, `data: ${JSON.stringify(data)}\n\n`);
        } catch {
          console.warn('[STREAM] Invalid JSON line:', line.slice(0, 100));
          safeWrite(res, `data: ${JSON.stringify({
            error: {
              message: 'Upstream sent malformed chunk',
              type: 'stream_parse_error',
              details: line.slice(0, 100)
            }
          })}\n\n`);
        }
      };

      upstreamStream.on('data', chunk => {
        buffer += decoder.write(chunk);

        if (buffer.length > MAX_BUFFER_SIZE) {
          console.error('[STREAM] Buffer overflow, destroying connection');
          safeWrite(res, `data: ${JSON.stringify({
            error: {
              message: 'Stream buffer overflow',
              type: 'stream_error'
            }
          })}\n\n`);
          safeWrite(res, 'data: [DONE]\n\n');
          res.end();
          upstreamStream.destroy();
          cleanup();
          return;
        }

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          processLine(line);
        }
      });

      upstreamStream.on('end', () => {
        buffer += decoder.end();
        if (buffer.trim()) {
          for (const line of buffer.split('\n')) {
            processLine(line);
          }
        }

        const flushedDelta = normalizer.flush();

        // Stream ended while still mid <tool_call> tag (e.g. cut off by
        // max_tokens before the closing tag arrived). Surface the raw
        // partial tag as text rather than silently dropping it.
        const toolRecoveryLeftover = toolRecovery.flush();
        if (toolRecoveryLeftover) {
          console.warn('[TOOL_CALL_RECOVERY] Stream ended mid <tool_call> tag; flushing raw text instead of dropping it.');
          flushedDelta.content = (flushedDelta.content || '') + toolRecoveryLeftover;
        }

        if (flushedDelta.content || flushedDelta.reasoning) {
          let clientContent = '';

          if (SHOW_REASONING && inlineReasoning) {
            // Inline reasoning format: bake <thinking> tags into content.
            if (flushedDelta.reasoning && !reasoningOpen) {
              clientContent += `<thinking>\n${flushedDelta.reasoning}`;
              reasoningOpen = true;
            } else if (flushedDelta.reasoning) {
              clientContent += flushedDelta.reasoning;
            }

            if (flushedDelta.content && reasoningOpen) {
              clientContent += `\n</thinking>\n\n${flushedDelta.content}`;
              reasoningOpen = false;
            } else if (flushedDelta.content) {
              clientContent += flushedDelta.content;
            }
          } else {
            // Default behavior: clean content, no inline tags
            clientContent = flushedDelta.content || '';
          }

          const finalChunk = { choices: [{ delta: {} }] };
          if (clientContent) finalChunk.choices[0].delta.content = clientContent;

          // Mirror the per-chunk handling above: leftover reasoning text
          // must also reach structured-format clients, not just get folded
          // into inline tags. Previously this was dropped entirely whenever
          // SHOW_REASONING was on but inlineReasoning was off.
          if (SHOW_REASONING && !inlineReasoning && flushedDelta.reasoning) {
            finalChunk.choices[0].delta.reasoning = flushedDelta.reasoning;
            finalChunk.choices[0].delta.reasoning_content = flushedDelta.reasoning;
          }

          if (Object.keys(finalChunk.choices[0].delta).length > 0) {
            safeWrite(res, `data: ${JSON.stringify(finalChunk)}\n\n`);
          }
        }

        // A model can get cut off mid-reasoning (e.g. hits max_tokens while
        // still inside a <think> block, never emitting the closing tag).
        // If we opened an inline <thinking> tag earlier and nothing above
        // closed it, close it now so inline-format clients aren't left with
        // an unterminated tag.
        if (SHOW_REASONING && inlineReasoning && reasoningOpen) {
          safeWrite(res, `data: ${JSON.stringify({ choices: [{ delta: { content: '\n</thinking>\n' } }] })}\n\n`);
          reasoningOpen = false;
        }

        if (!doneSent) {
          safeWrite(res, 'data: [DONE]\n\n');
        }
        streamEndedCleanly = true;
        if (!res.writableEnded) {
          res.end();
        }
        cleanup();
      });

      upstreamStream.on('error', err => {
        console.error('[STREAM] Upstream error:', err.message);
        if (!res.writableEnded) {
          safeWrite(res, `data: ${JSON.stringify({
            error: {
              message: 'Stream interrupted by upstream error',
              type: 'stream_error'
            }
          })}\n\n`);
          safeWrite(res, 'data: [DONE]\n\n');
          res.end();
        }
        cleanup();
      });

      req.on('close', () => {
        const clientGone = req.destroyed || !res.writable;
        if (!streamEndedCleanly && clientGone) {
          console.warn('[STREAM] Client disconnected prematurely');
        }
        if (upstreamStream && !upstreamStream.destroyed && !streamEndedCleanly) {
          upstreamStream.destroy();
        }
        cleanup();
      });
    } else {
      // Non-streaming response
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        // Report the model that actually answered (which may differ from the
        // requested alias if a fallback kicked in), not the raw client input.
        model: usedModel,
        created: Math.floor(Date.now() / 1000),
        choices: (response.data.choices || []).map((choice, i) => {
          const normalizedChoice = normalizeNonStreamChoice(choice, usedModel);
          let content = normalizedChoice.message?.content || '';
          const reasoning = normalizedChoice.message?.reasoning || '';

          // Recover tool calls the backend leaked into content as a raw
          // <tool_call> tag instead of NIM's structured tool_calls field
          // (confirmed upstream bug — see subsystem notes above).
          const { content: cleanedContent, toolCalls: recoveredToolCalls } = extractLeakedToolCalls(content);
          content = cleanedContent;

          if (SHOW_REASONING && inlineReasoning && reasoning) {
            // Inline reasoning format: bake <thinking> tags into content.
            content = `<thinking>\n${reasoning}\n</thinking>\n\n${content}`;
          }

          const finalMessage = { ...normalizedChoice.message, content };

          if (recoveredToolCalls.length > 0) {
            finalMessage.tool_calls = [
              ...(normalizedChoice.message?.tool_calls || []),
              ...recoveredToolCalls
            ];
            // A present-but-whitespace-only content string alongside
            // tool_calls trips up some client-side agent loops that expect
            // null content on a tool-call turn (mirrors real OpenAI
            // tool-call responses, which always send content: null).
            if (!finalMessage.content || !finalMessage.content.trim()) {
              finalMessage.content = null;
            }
          }

          // Same as the streaming path: keep the structured field alongside
          // the inline tags so structured-reasoning clients can render their
          // own UI.
          if (SHOW_REASONING && reasoning) {
            finalMessage.reasoning = reasoning;
            finalMessage.reasoning_content = reasoning;
          } else {
            delete finalMessage.reasoning;
            delete finalMessage.reasoning_content;
          }

          const finalChoice = {
            ...normalizedChoice,
            index: i,
            message: finalMessage,
            ...(recoveredToolCalls.length > 0 && { finish_reason: 'tool_calls' })
          };
          return finalChoice;
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };

      res.json(openaiResponse);
    }
  } catch (error) {
    console.error('[PROXY] Fatal error:', error.message);
    console.error('[PROXY] NIM response:', error.response?.data);

    if (!res.headersSent) {
      // If this fires after the streaming branch already called
      // res.setHeader('Content-Type', 'text/event-stream') but before any
      // actual write, res.json() below won't override it — Express's
      // res.json() only sets Content-Type when it isn't already set. Force
      // it back to JSON explicitly so the error body's declared type
      // actually matches its content.
      res.set('Content-Type', 'application/json');
      res.status(error.response?.status || 500).json({
        error: {
          message: error.message,
          type: 'invalid_request_error',
          code: error.response?.status || 500
        }
      });
    } else if (!res.writableEnded) {
      safeWrite(res, `data: ${JSON.stringify({
        error: {
          message: error.message,
          type: 'proxy_error'
        }
      })}\n\n`);
      safeWrite(res, 'data: [DONE]\n\n');
      res.end();
    }

    if (upstreamStream && !upstreamStream.destroyed) {
      upstreamStream.destroy();
    }
  }
});

app.use((req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.method} ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

// ─── Startup ───────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`[PROXY] Hybrid proxy running on port ${PORT}`);
  console.log(`[PROXY] Max tokens limit: ${MAX_TOKENS_LIMIT}`);
  validateModels().catch(err => {
    console.error('[VALIDATION] Startup check failed:', err.message);
  });
});
