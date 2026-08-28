// tools.js
// Tool-call leak recovery. Some backend models intermittently fail to
// convert their native tool-call output into NIM's structured `tool_calls`
// field and instead dump the raw Hermes-style tag straight into `content`
// as plain text:
//   <tool_call>
//   {"name": "...", "arguments": {...}}
//   </tool_call>
//
// This is a known upstream parser bug rather than a request-shape problem
// on the proxy's end — see vllm-project/vllm#48095, zai-org/GLM-5#15, and
// zed-industries/zed#55884 for reproductions across different models and
// clients (GLM-5.2 and nemotron-3-super are the ones confirmed here, but it
// isn't limited to those two — any model in server.js's MODEL_MAPPING or
// FALLBACK_MODELS could hit it).
//
// Since it's an intermittent upstream failure rather than a model that
// flatly lacks tool-call support, it can't be fixed by avoiding a specific
// model — it's caught and repaired wherever it happens. This module runs
// unconditionally (streaming and non-streaming) and is a no-op when the tag
// never appears.

// ─── Tool-call recovery subsystem ──────────────────────────────────────────
// Some backend models intermittently fail to convert their native tool-call
// output into NIM's structured `tool_calls` field and instead dump the raw
// Hermes-style tag straight into `content` as plain text:
//   <tool_call>
//   {"name": "...", "arguments": {...}}
//   </tool_call>
//
// This is a known upstream parser bug rather than a request-shape problem on
// this proxy's end — see vllm-project/vllm#48095, zai-org/GLM-5#15, and
// zed-industries/zed#55884 for reproductions across different models and
// clients (GLM-5.2 and nemotron-3-super are the ones confirmed here, but it
// isn't limited to those two — any model in MODEL_MAPPING/FALLBACK_MODELS
// could hit it).
//
// Since it's an intermittent upstream failure rather than a model that
// flatly lacks tool-call support, it can't be fixed by avoiding a specific
// model — it's caught and repaired wherever it happens. This layer runs
// unconditionally (streaming and non-streaming) and is a no-op when the tag
// never appears.

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
// Malformed JSON inside a tag (truncated args, missing closing brace) is left
// in place rather than silently dropped, so worst case the raw tag still
// reaches the client instead of vanishing without a trace.
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

// Returns how many trailing characters of `str` could be the start of `tag`
// split across a chunk boundary - e.g. str ending in "<tool_c" against tag
// "<tool_call>" returns 7. Used to hold back a possibly-split tag instead of
// emitting a tag fragment as visible content.
function partialTagSuffixLength(str, tag) {
  const maxLen = Math.min(str.length, tag.length - 1);
  for (let i = maxLen; i > 0; i--) {
    if (tag.startsWith(str.slice(str.length - i))) return i;
  }
  return 0;
}

// Streaming, cross-chunk recovery for tool calls the backend leaks into
// content as raw <tool_call> tags.
//
// Deliberately NOT built on DelimiterParser (used above for <think>
// extraction): DelimiterParser concatenates every "inside tag" segment in a
// chunk into one string, which is correct for a single continuous <think>
// block but wrong here — if two separate <tool_call>...</tool_call> pairs
// land in the same chunk (which happens in bursts), their JSON bodies would
// get concatenated, fail to parse, and both calls would be lost. This is a
// purpose-built cursor scanner instead, mirroring extractLeakedToolCalls()
// above, so each pair is extracted independently no matter how many land in
// one chunk.
class ToolCallStreamRecovery {
  constructor() {
    this.buffer = '';
    this.toolCallIndex = 0;
  }

  // text: already-normalized content for this chunk (post reasoning-split).
  // Returns { content, toolCallDeltas } - content is safe to forward to the
  // client as-is, toolCallDeltas is an array (possibly empty, possibly more
  // than one) of ready OpenAI tool_calls delta objects.
  process(text) {
    this.buffer += text;

    const toolCallDeltas = [];
    let outContent = '';
    let cursor = 0;

    while (true) {
      const openIdx = this.buffer.indexOf(TOOL_CALL_OPEN, cursor);

      if (openIdx === -1) {
        // No complete open tag left. The tail of what remains could still be
        // the start of one split across a chunk boundary - hold that part
        // back instead of emitting a tag fragment as content.
        const remaining = this.buffer.slice(cursor);
        const partialLen = partialTagSuffixLength(remaining, TOOL_CALL_OPEN);
        const safeLen = remaining.length - partialLen;
        outContent += remaining.slice(0, safeLen);
        this.buffer = remaining.slice(safeLen);
        break;
      }

      const closeIdx = this.buffer.indexOf(TOOL_CALL_CLOSE, openIdx);
      if (closeIdx === -1) {
        // Found an open tag but its close hasn't arrived yet (could still be
        // coming in a later chunk). Flush the plain text before it, buffer
        // the rest for next time.
        outContent += this.buffer.slice(cursor, openIdx);
        this.buffer = this.buffer.slice(openIdx);
        break;
      }

      outContent += this.buffer.slice(cursor, openIdx);
      const inner = this.buffer.slice(openIdx + TOOL_CALL_OPEN.length, closeIdx).trim();

      try {
        const parsed = JSON.parse(inner);
        if (parsed && typeof parsed.name === 'string') {
          toolCallDeltas.push({
            index: this.toolCallIndex++,
            id: generateToolCallId(),
            type: 'function',
            function: { name: parsed.name, arguments: JSON.stringify(parsed.arguments ?? {}) }
          });
        } else {
          // Well-formed JSON but not the shape we expect - keep it visible.
          outContent += this.buffer.slice(openIdx, closeIdx + TOOL_CALL_CLOSE.length);
        }
      } catch {
        console.warn('[TOOL_CALL_RECOVERY] Malformed JSON inside streamed <tool_call> tag, leaving raw:', inner.slice(0, 200));
        outContent += this.buffer.slice(openIdx, closeIdx + TOOL_CALL_CLOSE.length);
      }

      cursor = closeIdx + TOOL_CALL_CLOSE.length;
    }

    return { content: outContent, toolCallDeltas };
  }

  // Stream ended while still mid-tag (cut off before the closing tag
  // arrived, e.g. hit max_tokens) - return the raw partial buffer as text
  // instead of silently dropping it.
  flush() {
    const leftover = this.buffer;
    this.buffer = '';
    return leftover;
  }
}

module.exports = { extractLeakedToolCalls, ToolCallStreamRecovery };
