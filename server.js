// server.js — OpenAI-compatible proxy for NVIDIA NIM
// Reasoning payload logic: reasoning.js. Tool-call leak recovery: tools.js.

const express = require('express');
const cors = require('cors');
const axios = require('axios');
const https = require('https');
const { StringDecoder } = require('string_decoder');
const { timingSafeEqual } = require('crypto');
const { SHOW_REASONING, getReasoningPayload, resolveEffectiveThinking, StreamNormalizer, normalizeNonStreamChoice } = require('./reasoning');
const { extractLeakedToolCalls, ToolCallStreamRecovery } = require('./tools');

const app = express();
const PORT = process.env.PORT || 3000;

// ─── Configuration ──────────────────────────────────────────────────────

const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;
const CLIENT_AUTH_KEY = process.env.CLIENT_AUTH_KEY;
const ENABLE_THINKING_MODE = process.env.ENABLE_THINKING_MODE === 'true';
const SKIP_VALIDATION = process.env.SKIP_VALIDATION === 'true';
const DISCORD_WEBHOOK_URL = process.env.DISCORD_WEBHOOK_URL;

// Verbose logging: reasoning payload per fallback attempt, full request/error bodies on failure.
const DEBUG_MODE = process.env.DEBUG_MODE === 'true';

const MAX_TOKENS_LIMIT = 65536;

// Shared keep-alive agent for connection reuse on long-lived deployments.
const keepAliveAgent = new https.Agent({
  keepAlive: true,
  maxSockets: 50,
  maxFreeSockets: 10
});

const nim = axios.create({
  baseURL: NIM_API_BASE,
  httpsAgent: keepAliveAgent
});

// Per-attempt timeout before falling back to the next model. Reasoning
// models get a longer window since thinking delays first-token latency.
// Check these against your platform's own request duration limit.
const REQUEST_TIMEOUT_MS = Number(process.env.REQUEST_TIMEOUT_MS) || 180000;
const REASONING_REQUEST_TIMEOUT_MS = Number(process.env.REASONING_REQUEST_TIMEOUT_MS) || 480000;
const VALIDATION_TIMEOUT_MS = 15000;
const MAX_BUFFER_SIZE = 1024 * 1024; // 1MB

if (ENABLE_THINKING_MODE) console.log('[CONFIG] Thinking mode: ENABLED');
if (DEBUG_MODE) console.log('[CONFIG] Debug mode: ENABLED (verbose reasoning + fallback logging)');

// ─── Config validation ──────────────────────────────────────────────────

function validateConfig() {
  const fatal = (msg) => { console.error(`[FATAL] ${msg}`); process.exit(1); };
  if (!NIM_API_KEY) fatal('NIM_API_KEY is required. Get one at https://build.nvidia.com/');
  if (!CLIENT_AUTH_KEY) {
    console.warn('[WARN] CLIENT_AUTH_KEY not set. All requests will be rejected with 403.');
  }
}
validateConfig();

// ─── Model Mapping ───────────────────────────────────────────────────────

// Aliases are periodically re-checked against NIM's live catalog (see
// validateModels() and GET /v1/models?live=true). Comments note the prior
// backend model ID where an entry was swapped out for a dead catalog entry.
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/nemotron-3-super-120b-a12b',
  'gpt-4': 'nvidia/nemotron-3-ultra-550b-a55b',
  'gpt-3.5': 'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning', // was qwen/qwen3.5-397b-a17b
  'gpt-4-turbo': 'moonshotai/kimi-k3',
  'claude-3-opus': 'google/diffusiongemma-26b-a4b-it',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'nvidia/llama-3.1-nemotron-70b-instruct', // was nvidia/llama-3.3-nemotron-super-49b-v1.5
  'gemini-turbo': 'nvidia/llama3-chatqa-1.5-70b', // was meta/llama-3.3-70b-instruct
  'gpt-3.5o': 'nvidia/nemotron-3.5-lightning-30b-a3b', // was google/gemma-2b
  'gpt-4-flash': 'deepseek-ai/deepseek-v4-flash-0731',
  'gpt-4o': 'deepseek-ai/deepseek-v4-pro-0813',
  'mistral': 'mistralai/mistral-large-2-instruct', // was mistralai/mistral-large-3-675b-instruct-2512
  'mistral-turbo': 'nv-mistralai/mistral-nemo-12b-instruct', // was mistralai/mistral-medium-3.5-128b
  'mistral-pro': 'mistralai/mistral-7b-instruct-v0.3', // was mistralai/mistral-small-4-119b-2603
  'mistral-nemo': 'mistralai/mistral-nemotron',
  'mistral-fast': 'nvidia/mistral-nemo-minitron-8b-8k-instruct', // was mistralai/ministral-14b-instruct-2512
  'google-light': 'google/gemma-4-31b-it',
  'google-lightest': 'meta/muse-glimmer-30b', // was google/gemma-2b
  'google-lighter': 'poolside/laguna-xs-2.1', // was google/gemma-3-4b-it
  'm3': 'minimaxai/minimax-m3'
};

// Used when an unrecognized alias is requested. Must point at a live model.
const DEFAULT_MODEL = 'google/gemma-4-31b-it';

// Ordered by observed reliability/speed — an early failing model delays every fallback behind it.
const FALLBACK_MODELS = [
  'google/diffusiongemma-26b-a4b-it',
  'google/gemma-4-31b-it',
  'mistralai/mistral-nemotron',
  'nvidia/nemotron-3-super-120b-a12b'
];

// ─── Middleware ─────────────────────────────────────────────────────────

app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Malformed JSON body -> clean OpenAI-style error instead of Express's default HTML error page.
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

function extractBearerToken(authHeader) {
  if (!authHeader || typeof authHeader !== 'string') return null;
  const trimmed = authHeader.trim();
  if (!trimmed.startsWith('Bearer ')) return null;
  const token = trimmed.slice('Bearer '.length).trim();
  return token || null;
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
  if (req.path === '/health' || req.path === '/v1/models' || req.path === '/') {
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

// ─── Validation ─────────────────────────────────────────────────────────

// Shared by startup validateModels() and GET /v1/models?live=true.
async function fetchLiveModelIds() {
  const response = await nim.get('/models', {
    headers: {
      Authorization: `Bearer ${NIM_API_KEY}`,
      'Content-Type': 'application/json'
    },
    timeout: VALIDATION_TIMEOUT_MS
  });

  return new Set((response.data.data || []).map(m => m.id));
}

async function validateModels() {
  if (SKIP_VALIDATION) {
    console.log('[VALIDATION] Skipped (SKIP_VALIDATION=true)');
    return;
  }

  console.log('[VALIDATION] Checking model availability via /v1/models...');
  try {
    const availableModels = await fetchLiveModelIds();

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
    console.warn('[VALIDATION] Set SKIP_VALIDATION=true if your NIM provider lacks a model listing endpoint.');
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

// ─── Helper: Safe Stream Writing ─────────────────────────────────────────

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

// ─── Helper: Fallback Chain ──────────────────────────────────────────────

// Per-model cooldown after a 403/429, in-memory (resets on restart, not shared across instances).
const modelCooldowns = new Map(); // model -> timestamp (ms) until which to skip it

const RATE_LIMIT_COOLDOWN_MS = Number(process.env.RATE_LIMIT_COOLDOWN_MS) || 30000;
const ACCESS_DENIED_COOLDOWN_MS = Number(process.env.ACCESS_DENIED_COOLDOWN_MS) || 300000;

function isInCooldown(model) {
  const until = modelCooldowns.get(model);
  return typeof until === 'number' && Date.now() < until;
}

function setCooldown(model, ms) {
  modelCooldowns.set(model, Date.now() + ms);
}

async function callWithFallback(baseRequest, models, enableThinking, clientReasoningEffort, hasTools) {
  let lastError = null;
  const timeoutMs = resolveEffectiveThinking(enableThinking, clientReasoningEffort)
    ? REASONING_REQUEST_TIMEOUT_MS
    : REQUEST_TIMEOUT_MS;

  // Skip cooling-down models; fall back to the full list if that empties the chain.
  const activeModels = models.filter(m => !isInCooldown(m));
  const attemptOrder = activeModels.length > 0 ? activeModels : models;

  for (const model of attemptOrder) {
    const reasoningPayload = getReasoningPayload(model, enableThinking, clientReasoningEffort, hasTools);
    const fullRequest = { ...baseRequest, model, ...reasoningPayload };

    if (DEBUG_MODE) {
      console.log(`[DEBUG] Attempting ${model} with reasoning payload:`, JSON.stringify(reasoningPayload), `(timeout: ${timeoutMs}ms)`);
    }

    try {
      const res = await nim.post(
        '/chat/completions',
        fullRequest,
        {
          headers: {
            Authorization: `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          responseType: baseRequest.stream ? 'stream' : 'json',
          timeout: timeoutMs
        }
      );
      return { response: res, model };
    } catch (err) {
      lastError = err;
      const status = err.response?.status;

      console.warn(
        `[FALLBACK] Model failed: ${model}`,
        status,
        err.response?.data?.error?.message || err.message
      );
      if (DEBUG_MODE) {
        console.log(`[DEBUG] Full request body sent to ${model}:`, JSON.stringify(fullRequest));
        console.log(`[DEBUG] Full upstream error response:`, JSON.stringify(err.response?.data || null));
      }

      // Same key for every attempt: a 401 means every remaining model would fail identically.
      if (status === 401) {
        throw err;
      }

      // 403: likely an access-tier issue, not a dead key — cooldown just this model.
      if (status === 403) {
        setCooldown(model, ACCESS_DENIED_COOLDOWN_MS);
      }

      // 429: rate limited — short cooldown.
      if (status === 429) {
        setCooldown(model, RATE_LIMIT_COOLDOWN_MS);
      }
    }
  }

  throw lastError || new Error('All models failed');
}

// ─── Routes ───────────────────────────────────────────────────────────────

// Prevents a bare 403 (from the auth middleware) on root URL hits from a browser.
app.get('/', (req, res) => {
  res.type('html').send(`<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nim-to-openai-proxy</title>
<style>
  body {
    margin: 0;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #0b0f14;
    color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    text-align: center;
    padding: 24px;
  }
  .card { max-width: 480px; }
  h1 { font-size: 1.3rem; margin: 0 0 0.75rem; }
  p { color: #9aa7b2; line-height: 1.55; margin: 0.5rem 0; }
  code { background: #161b22; padding: 2px 6px; border-radius: 4px; color: #7ee787; }
</style>
</head>
<body>
  <div class="card">
    <svg width="44" height="44" viewBox="0 0 24 24" fill="none" style="margin: 0 auto 14px; display: block;">
      <circle cx="12" cy="12" r="11" stroke="#7ee787" stroke-width="1.5"/>
      <path d="M7 12.5l3 3 6-6.5" stroke="#7ee787" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
    </svg>
    <h1>it's up.</h1>
    <p>this proxies OpenAI-format chat requests to NVIDIA NIM. point any OpenAI-compatible client at it, pick a model with a plain alias (<code>gpt-4</code>, <code>mistral</code>, etc), and it handles model fallback, streaming, and each backend's own reasoning/thinking quirks for you.</p>
    <p>it's an API, not a website: nothing lives at this root path.</p>
    <p>send requests to <code>/v1/chat/completions</code> with an <code>Authorization: Bearer &lt;token&gt;</code> header.</p>
    <p>status check, no token needed: <code>/health</code></p>
    <p>bugs / questions: <a href="https://github.com/skywalker14017/nim-to-openai-proxy" style="color:#7ee787;">open an issue on GitHub</a> (docs are there too), or hit me up on Discord (i'll be faster there): <code>skywalker_1401</code></p>
  </div>
</body>
</html>`);
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok', version: '2.6.0' });
});

app.get('/v1/models', async (req, res) => {
  if (req.query.live !== 'true') {
    return res.json({
      object: 'list',
      data: Object.keys(MODEL_MAPPING).map(id => ({
        id,
        object: 'model',
        created: Math.floor(Date.now() / 1000), // OpenAI spec expects Unix seconds
        owned_by: 'nim-proxy'
      }))
    });
  }

  // Cross-checks every alias against NIM's live catalog on demand.
  try {
    const availableModels = await fetchLiveModelIds();
    const data = Object.entries(MODEL_MAPPING).map(([id, backend]) => ({
      id,
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'nim-proxy',
      backend,
      available: availableModels.has(backend)
    }));

    res.json({
      object: 'list',
      data,
      live_check: {
        checked_at: new Date().toISOString(),
        unavailable_aliases: data.filter(m => !m.available).map(m => m.id)
      }
    });
  } catch (err) {
    console.warn(`[MODELS] Live check failed: ${err.message}`);
    res.status(502).json({
      error: {
        message: `Live model check against NIM failed: ${err.message}`,
        type: 'live_check_error',
        code: 502
      }
    });
  }
});

app.post('/v1/chat/completions', async (req, res) => {
  let streamEndedCleanly = false;
  let upstreamStream = null;

  try {
    const { model, max_tokens, temperature, stream, reasoning_effort } = req.body;

    let primaryModel = MODEL_MAPPING[model];
    if (!primaryModel) {
      console.warn(`[PROXY] Unknown model alias "${model}", falling back to default: ${DEFAULT_MODEL}`);
      primaryModel = DEFAULT_MODEL;
    }

    // De-dupe: avoids retrying the same model twice if it's also in FALLBACK_MODELS.
    const modelChain = [...new Set([primaryModel, ...FALLBACK_MODELS])];

    // Forward all client fields except model (replaced per-attempt) and
    // reasoning_effort (translated per-model by getReasoningPayload).
    const { model: _droppedModel, reasoning_effort: _droppedReasoningEffort, ...forwardedFields } = req.body;

    const baseRequest = {
      ...forwardedFields,
      temperature: temperature ?? 0.7,
      max_tokens: Math.min(max_tokens ?? 2048, MAX_TOKENS_LIMIT),
      stream: stream || false
    };

    const { response, model: usedModel } = await callWithFallback(
      baseRequest,
      modelChain,
      ENABLE_THINKING_MODE,
      reasoning_effort,
      !!req.body.tools
    );

    upstreamStream = response.data;
    console.log('[PROXY] Model used:', usedModel);

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

        // Exact match avoids false-positiving on model output that happens
        // to contain the literal substring "[DONE]".
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
              clientContent = normalizedDelta.content || '';
            }

            const { content: recoveredContent, toolCallDeltas } = toolRecovery.process(clientContent);
            clientContent = recoveredContent;
            if (toolCallDeltas.length > 0) {
              delta.tool_calls = toolCallDeltas;
              if (data.choices[0]) data.choices[0].finish_reason = 'tool_calls';
            }

            delta.content = clientContent;

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

        const toolRecoveryLeftover = toolRecovery.flush();
        if (toolRecoveryLeftover) {
          console.warn('[TOOL_CALL_RECOVERY] Stream ended mid <tool_call> tag; flushing raw text instead of dropping it.');
          flushedDelta.content = (flushedDelta.content || '') + toolRecoveryLeftover;
        }

        if (flushedDelta.content || flushedDelta.reasoning) {
          let clientContent = '';

          if (SHOW_REASONING && inlineReasoning) {
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
            clientContent = flushedDelta.content || '';
          }

          const finalChunk = { choices: [{ delta: {} }] };
          if (clientContent) finalChunk.choices[0].delta.content = clientContent;

          if (SHOW_REASONING && !inlineReasoning && flushedDelta.reasoning) {
            finalChunk.choices[0].delta.reasoning = flushedDelta.reasoning;
            finalChunk.choices[0].delta.reasoning_content = flushedDelta.reasoning;
          }

          if (Object.keys(finalChunk.choices[0].delta).length > 0) {
            safeWrite(res, `data: ${JSON.stringify(finalChunk)}\n\n`);
          }
        }

        // Close an inline <thinking> tag left open if the model was cut off mid-reasoning.
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
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        model: usedModel, // actual model that answered, may differ from the requested alias
        created: Math.floor(Date.now() / 1000),
        choices: (response.data.choices || []).map((choice, i) => {
          const normalizedChoice = normalizeNonStreamChoice(choice, usedModel);
          let content = normalizedChoice.message?.content || '';
          const reasoning = normalizedChoice.message?.reasoning || '';

          const { content: cleanedContent, toolCalls: recoveredToolCalls } = extractLeakedToolCalls(content);
          content = cleanedContent;

          if (SHOW_REASONING && inlineReasoning && reasoning) {
            content = `<thinking>\n${reasoning}\n</thinking>\n\n${content}`;
          }

          const finalMessage = { ...normalizedChoice.message, content };

          if (recoveredToolCalls.length > 0) {
            finalMessage.tool_calls = [
              ...(normalizedChoice.message?.tool_calls || []),
              ...recoveredToolCalls
            ];
            // null content on tool-call turns matches real OpenAI responses
            if (!finalMessage.content || !finalMessage.content.trim()) {
              finalMessage.content = null;
            }
          }

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
      // Express only sets Content-Type if unset; force JSON in case the
      // streaming branch already set text/event-stream before failing.
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

// ─── Startup ──────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`[PROXY] Hybrid proxy running on port ${PORT}`);
  console.log(`[PROXY] Max tokens limit: ${MAX_TOKENS_LIMIT}`);
  validateModels().catch(err => {
    console.error('[VALIDATION] Startup check failed:', err.message);
  });
});
