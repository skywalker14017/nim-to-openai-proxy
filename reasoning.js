// reasoning.js
// Owns everything related to backend "thinking" control: which
// chat_template_kwargs/top-level fields each model needs to turn reasoning
// on/off, and how to pull reasoning text back out of responses that embed
// it inline in content vs. as a structured field.

// ─── Reasoning subsystem ────────────────────────────────────────────────────
// Owns: which chat_template_kwargs/top-level fields each backend model needs
// to control thinking, and how to pull reasoning text back out of responses
// that embed it inline vs. as a structured field.

const SHOW_REASONING = process.env.SHOW_REASONING === 'true';
if (SHOW_REASONING) console.log('[CONFIG] Reasoning display: ENABLED');

// Reasoning/thinking parameters vary by backend model and aren't part of the
// OpenAI schema, so they can't just be forwarded as-is — getReasoningPayload()
// below maps each backend model to its own request shape (see the comments
// on each case for that model's specific quirks).
//
// IMPORTANT: everything getReasoningPayload() returns is spread directly into
// the top-level JSON body sent to NIM via axios. Do NOT wrap it in an
// `extra_body` key — that's an openai-SDK-only convention that the official
// SDKs unwrap client-side into top-level fields before sending. This proxy
// posts to NIM's REST endpoint directly via axios, so a literal "extra_body"
// key in the body is just silently ignored by the backend.
//
// Reasoning output format: by default, reasoning is kept out of `content`
// and returned in a structured `reasoning`/`reasoning_content` field.
// Clients that expect legacy inline <thinking> tags baked into content can
// opt in by sending an `x-reasoning-format: inline` header.

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

  // reasoning_effort lives inside chat_template_kwargs for these two (see
  // getReasoningPayload below); thinking on/off is a separate flag. Pro and
  // Flash share the same V4 chat template, so they share the same enum.
  'deepseek-ai/deepseek-v4-flash-0731': ['low', 'high', 'max'],
  'deepseek-ai/deepseek-v4-pro-0813': ['low', 'high', 'max'],

  // Not a true adaptive/effort scale — these two only expose a single extra
  // "low_effort" middle tier between full reasoning and off.
  'nvidia/nemotron-3-super-120b-a12b': ['low'],
  'nvidia/nemotron-3-ultra-550b-a55b': ['low'],

  // MiniMax-M3's only non-binary option: let the model decide per-turn.
  'minimaxai/minimax-m3': ['adaptive'],

  // K3 can never fully disable reasoning (see the kimi-k3 case in
  // getReasoningPayload below) — this enum only governs how hard it thinks,
  // never whether it thinks at all.
  'moonshotai/kimi-k3': ['low', 'high', 'max']
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

// Resolves the client-facing reasoning_effort "off"/"on" override into an
// effective enableThinking boolean. Shared between getReasoningPayload()
// (to build the right payload) and callWithFallback() (to pick a sane
// per-request timeout) so the two can't drift out of sync on what "thinking
// is actually on for this request" means.
function resolveEffectiveThinking(enableThinking, clientReasoningEffort) {
  if (clientReasoningEffort === 'off') return false;
  if (clientReasoningEffort === 'on') return true;
  return enableThinking;
}

// Pure function returning model-specific reasoning request payloads. See the
// "Reasoning subsystem" note above regarding extra_body.
//
// Sending reasoning_effort: "off" / "on" forces thinking off/on for that one
// request, overriding the server's ENABLE_THINKING_MODE default — this is
// what lets a client-side reasoning toggle actually control every model
// below, not just the ones with a real effort enum. "off"/"on" are stripped
// before the model-specific effort check runs, so they never collide with a
// real per-model value like "high" or "adaptive".
//
// Caveat: gpt-oss models structurally always emit a reasoning channel, so
// this can only reduce them to their baseline default, not eliminate
// reasoning tokens entirely the way it can for other models here.
function getReasoningPayload(model, enableThinking, clientReasoningEffort, hasTools) {
  enableThinking = resolveEffectiveThinking(enableThinking, clientReasoningEffort);

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

    case 'deepseek-ai/deepseek-v4-flash-0731':
    case 'deepseek-ai/deepseek-v4-pro-0813': {
      // Both V4 models control reasoning via chat_template_kwargs — NOT a
      // bare top-level reasoning_effort field.
      if (!enableThinking) {
        return { chat_template_kwargs: { thinking: false } };
      }

      return {
        chat_template_kwargs: {
          thinking: true,
          reasoning_effort: effort || 'high'
        }
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

    case 'google/gemma-4-31b-it': {
      if (!enableThinking) return {};
      // enable_thinking alone makes the model reason internally but doesn't
      // put that reasoning in the response — a separate top-level
      // include_reasoning flag is required to get the `reasoning` field
      // back. Tie it to SHOW_REASONING so it's explicit either way.
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
      // "adaptive" lets the model decide per-turn whether to think — the
      // only self-deciding reasoning mode in this proxy. Send
      // reasoning_effort: "adaptive" to use it; otherwise this behaves like
      // a standard on/off toggle.
      const thinkingMode = effort === 'adaptive'
        ? 'adaptive'
        : (enableThinking ? 'enabled' : 'disabled');
      return { chat_template_kwargs: { thinking_mode: thinkingMode } };
    }

    case 'moonshotai/kimi-k3': {
      // K3 has no off-switch: the API always runs a thinking pass before
      // answering, so unlike every other model above there's no
      // chat_template_kwargs.thinking:false (or equivalent) to send. The
      // only lever is how much it thinks, via a top-level reasoning_effort.
      //
      // When thinking is "off" for this request, ask for the lowest tier
      // instead of returning {} — omitting the field lets it fall through
      // to Kimi's own default of 'max', the slowest and most expensive
      // setting, which would silently ignore the caller's intent to keep
      // this request cheap.
      if (effort) return { reasoning_effort: effort };
      return { reasoning_effort: enableThinking ? 'high' : 'low' };
    }

    default:
      // Default reasoning models (Kimi, MiniMax, etc.) or non-reasoning models
      return {};
  }
}

module.exports = {
  SHOW_REASONING,
  getReasoningPayload,
  resolveEffectiveThinking,
  StreamNormalizer,
  normalizeNonStreamChoice
};
