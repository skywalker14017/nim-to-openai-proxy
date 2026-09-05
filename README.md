### Announcement 
Due to personal health reasons, Jontte (original owner) will no longer be able to maintain this project. I'll try to maintain this for the time being!

### If you forked before June 7, 2026, please pull the latest version — previous versions had an auth bypass and startup DDoS vulnerability.

### Reasoning officially works! Every model (that supports reasoning) works. To use reasoning, use the [optional environmental variables](#optional-environment-variables) section!


### NVIDIA NIM to OpenAI Proxy
Hello, this is my first ever project on Github that I am making public. This is essentially just a translation layer between the API format that NVIDIA NIM uses to the format OpenAI uses. Jontte made this originally by building on a script from a Reddit guide. Over the time of a month he iterated on it, fixed problems, added auth, more models, and removed/replaced deprecated models.
These are the current available models for usage, and the use cases for all of them. 

### Why use this proxy?

Many frontends such as JanitorAI requires an OpenAI-compatible proxy to use NVIDIA NIM. SillyTavern can connect to NIM directly, but if you use **Lorebary** for prompts, lorebooks, or plugins, this proxy is necessary — Lorebary does not support NIM natively. Various other frontends that aren’t even related to RP may also not work with NIM.

### Legality

Yes, it's legal. It's just HTTP requests routed through your own proxy. You still need a valid NVIDIA API key and are subject to their rate limits. This is no different from using any other API gateway or reverse proxy.


### Requirements

Node.js 24+, a NVAPI/Nim API key, a deployment platform (though if you follow the guide below none of those should be a problem).

### Model Mapping

| Alias | Backend Model | Best For | Speed | Filters |
|---|---|---|---|---|
| `gpt-4-turbo` | `moonshotai/kimi-k3` | Deep, immersive RP & coding | Slow| Medium-High |
| `gpt-4` | `nvidia/nemotron-3-ultra-550b-a55b` | Immersive RP | Fast | Low |
| `gpt-4-flash` | `deepseek-ai/deepseek-v4-flash-0731` | Fast, non-edgy RP | Fast | High |
| `gpt-4o` | `deepseek-ai/deepseek-v4-pro-0813` | Coding | Slow | High |
| `gpt-3.5o` | `nvidia/nemotron-3.5-lightning-30b-a3b` | General chat, fast lightweight tasks | Very Fast | Low-Medium |
| `gemini-pro` | `nvidia/llama-3.1-nemotron-70b-instruct` | Daily driver, low latency | Fast | Low |
| `gemini-turbo` | `nvidia/llama3-chatqa-1.5-70b` | Fast general purpose | Fast | Low-Medium |
| `mistral` | `mistralai/mistral-large-2-instruct` | Best quality, unfiltered | Slow | Low |
| `mistral-turbo` | `nv-mistralai/mistral-nemo-12b-instruct` | Fast fallback | Very Fast | Low |
| `mistral-pro` | `mistralai/mistral-7b-instruct-v0.3` | Lightweight scenes | Very Fast | Low |
| `mistral-fast` | `nvidia/mistral-nemo-minitron-8b-8k-instruct` | Fast, compact Mistral | Very Fast | Low |
| `mistral-nemo` | `mistralai/mistral-nemotron` | Casual/anime RP | Fast | Low |
| `claude-3-opus` | `google/diffusiongemma-26b-a4b-it` | Alternative to Chinese models | Extremely fast | Low-Medium |
| `claude-3-sonnet` | `openai/gpt-oss-20b` | Fast, distinct voice | Fast | Low-Medium |
| `gpt-3.5-turbo` | `nvidia/nemotron-3-super-120b-a12b` | Lightweight tasks | Fast | Low |
| `gpt-3.5` | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | Nvidia nano fallback | Fast | Low |
| `google-light` | `google/gemma-4-31b-it` | Short scenes, fast | Fast | Low-Medium |
| `google-lighter` | `poolside/laguna-xs-2.1` | Coding | Fast | Unknown (to me) |
| `google-lightest` | `meta/muse-glimmer-30b` | Coding & Agentic work | Fast | Unknown (to me) |
| `m3` | `minimaxai/minimax-m3` | Experimental | Medium-High | Unknown (to me) |

### Filter Guide

| If your use-case involves... | Avoid | Use instead |
|---|---|---|
| Dark themes, violence, mature content | `gpt-4-flash`, `gpt-4-turbo` (They have high filters due to being based in China) | `mistral`, `gemini-pro`, `claude-3-opus` |
| Fast responses needed | `gpt-4o`, `gpt-4-turbo` | `gemini-pro`, `mistral-turbo`, `gpt-3.5o` |
| Long context / memory | Anything under 30B | `gpt-4-turbo`, `mistral`, `gpt-4` |
| Testing / very fast replies | — | `google-lightest`, `gpt-3.5o` |
| Coding / Long horizon work | — | `gpt-4-turbo`, `gpt-4o` |

### Fallback Chain

If your requested model fails, the proxy automatically tries:
1. Requested model
2. `google/gemma-4-31b-it`
3. `google/diffusiongemma-26b-a4b-it`
4. `mistralai/mistral-nemotron`
5. `nvidia/nemotron-3-super-120b-a12b`

All fallbacks are non-Chinese-hosted to avoid filter interruption mid-scene. These can be changed, but i found that these four work best as fallbacks.

### Auth Guide
I added auth middleware that wasn't present in the code I built upon. It uses an env var in your deployment. Use any secure string of 32+ characters, or generate one by hashing your NVAPI key. I recommend using an online hash tool or command to make a hash of your NVAPI key since the key is already complex as is, and a hash makes it more secure as it cannot be realistically reversed back to the NVAPI key. The first 32 characters of the hash are enough.
You can easily generate the hash with an online SHA-256 generator or any hash tool. Then make an env variable called "CLIENT_AUTH_KEY" and enter the first 32 characters of your hash into the variable (or any custom length over 16, or a custom key). Enter the hash into the API Key field in JanitorAI/SillyTavern.

### Proxy Setup Guide

Firstly head to https://build.nvidia.com/ and login/create an account. Then click your profile icon and navigate to "API keys". There you can generate an API key, and label it whatever you want. Save it immediately — you'll need to regenerate it if lost.

You *can* use basically any service that allows cloud deployments/VMs with a static IP, but I recommend Railway, Render, Vercel (which I personally use!).
and possibly Oracle if you are comfortable with SSH and value the freedom it gives, but Railway is the easiest to setup.
You need to login to Railway with your Github. **Fork the repo before deploying. I cannot see your env vars, but forking ensures your deployment is fully isolated!** This prevents me (or anyone) from seeing your deployment in Railway's dashboard or through github. I also recommend making sure deployments aren't visible on the frontpage.
After you have made a deployment, you need to wait around 3 minutes for it to finish deploying. Then go into the "variables" tab, and create an env var with the name "NIM_API_KEY", and enter your NVAPI key into the variable. Next in your deployment go to the settings page, and there the networking section. Generate a public URL for your deployment. This is necessary to access it. Now your proxy is ready.

### Important Information
You can check the status of your proxy with the "/health" endpoint, and a list of models with "/v1/models". These endpoints intentionally do not require the auth, so clients can verify connectivity before configuring auth.
Your actual chat endpoint is in "/v1/chat/completions", and is the one you use in Janitor AI/SillyTavern or whatever platform you use.
The client never sees your NVAPI key, which is why we don't use it as the auth, since the whole point of the auth configuration is so that your NVAPI key is not stored on your client.

### Optional Environment Variables

After deploying, you can set these in Railway's **Variables** tab (reasoning does not reliably work):

| Variable | Value | Effect |
|---|---|---|
| `SHOW_REASONING` | `true` | Shows model reasoning in `<thinking>` tags |
| `ENABLE_THINKING_MODE` | `true` | Sends thinking parameters to supported models |
| `DISCORD_WEBHOOK_URL` | Webhook URL | Alerts you when models fail validation |
| `SKIP_VALIDATION` | `true` | Disables startup model checks |
| `DEBUG_MODE` | `true` | Allows you to debug everything going on. |

Set to `false` or remove to disable. Changes apply without redeploying.

### Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| "All models failed" error | NIM API key invalid or expired | Regenerate key at build.nvidia.com |
| Very slow responses | Using `gpt-4o`, `gpt-4-turbo`, or other Chinese-hosted models during peak hours | Switch to `gemini-pro`, `mistral-turbo`, or `gpt-3.5o` |
| Filter interrupts RP | Using Chinese-hosted model for mature content | Use `mistral`, `gemini-pro`, or `claude-3-opus` |
| 404 on `/v1/chat/completions` | Auth mismatch | Verify `CLIENT_AUTH_KEY` matches between Railway and client |
| "Failed to fetch (unk)" / "A network error occurred" | JanitorAI cached old proxy config after changing URL or model | **Reload the page** — changes don't apply until refresh |


## Contributing

This is a personal project I am maintaining for my own use, but I'm happy if it helps others. If you spot a bug, want to suggest a model mapping, or have a small improvement, feel free to open an issue or PR. I’ll attempt to respond to you pretty fast, I can’t promise ultra-fast responses but I’ll do my best.

### What I'm open to
- Model mapping updates (NIM deprecates things constantly)
- Bug fixes
- Small feature additions that don't complicate the core flow
- Documentation improvements

### What I'm less likely to merge
- Major architectural changes.
- Features I don't personally use (harder for me to maintain)
- Anything that adds complexity without clear benefit

## Issues

Before opening an issue, check if it's already covered in the [Troubleshooting](#troubleshooting) section. If a model stopped working, it's probably deprecated by NVIDIA — check the [NIM catalog](https://build.nvidia.com/) first.

When reporting bugs, include:
- Which model alias you were using
- Whether streaming was enabled
- The error message (or "All models failed" if that's what you got)
- Your deployment platform (Railway, Render, etc.)

## Contact
Need to reach out faster? Add me on Discord, his username is - `Skywalker_1401`. I'll respond faster on Discord than Github.

## Disclaimer

I am not a professional developer.
