# Running Maki in production

A step-by-step guide to running `maki serve` for real users: what to prepare, how to configure every key, how to put it behind TLS, and how to check that it works. Read [API keys: which one is which](#api-keys-which-one-is-which) first; most setup mistakes come from mixing the two kinds of key up.

It runs the same way on Linux, macOS and Windows; where a command differs, both versions are shown. **Recommended setup: llama.cpp as the model server** (see [Which backend for production?](#which-backend-for-production)).

This guide covers one agent served over HTTP with `maki serve`. If you only use Maki as a library inside your own application, you need steps 1 and 2 only (the provider or model-server key, in the environment of your program).

## API keys: which one is which

A production deployment has up to three different secrets. They are unrelated, set in different places, and protect different things.

| | Server key (`MAKI_API_KEY`) | Provider key (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `VLLM_API_KEY`, `LLAMACPP_API_KEY`) | Client key |
|---|---|---|---|
| What it protects | Who may call **your** Maki server | Your account at the **model provider** | (same value as the server key) |
| Who invents it | **You**: generate any long random string | The provider: you create it in their dashboard (for vLLM and llama.cpp, you choose it when you start *their* server with `--api-key`) | |
| Where it is set | On the machine running `maki serve` | On the machine running `maki serve` | In the program that calls Maki: `AgentProxy(api_key=...)` or an `Authorization: Bearer ...` header |
| Needed when | Always, if the server is reachable by anyone but you | Only for hosted backends, or a vLLM/llama.cpp server started with `--api-key` | |
| llama.cpp (recommended) | | Optional but recommended: **you choose** the value and start `llama serve --api-key` with it; Maki sends it from `LLAMACPP_API_KEY` | |
| Ollama | | **Not used.** Maki's Ollama backend sends no credentials, so protect Ollama by keeping it on a private network or behind a proxy | |

Rules that follow from this:

- **The server key is not a provider key.** `MAKI_API_KEY` is never sent to OpenAI, Anthropic or anyone else, and a provider key is never accepted by your server.
- **The agent YAML has no key fields.** `agent.yaml` selects the backend and model; secrets are read from environment variables only, so they never end up in a file you commit.
- **Clients never need the provider key.** Callers only hold the server key. The provider key stays on the server.
- Without a server key the server accepts everyone. `maki serve` only warns about this (on stderr) when it listens on a non-loopback address; it still starts.

## Step 0: Pre-requirements

Check each item before you start.

| Requirement | Detail |
|---|---|
| Python | 3.10 or newer (`python3 --version`) |
| A host you can keep running | **Linux**, **macOS** or **Windows** (10/11, Server 2019+). Examples use `systemd` on Linux, `launchd` on macOS and a service wrapper on Windows; any process supervisor works. See [Platform notes](#platform-notes) |
| A model backend, one of: | a **llama.cpp** server (recommended), **Ollama**, a **vLLM** server, or a hosted provider (**OpenAI**, **Anthropic**, **OpenRouter**) with an account and a provider key |
| For llama.cpp | A GGUF model that fits your RAM/VRAM (the model file size plus working memory for the context), and the `llama` command installed from [llama.cpp](https://llama.app/docs/introduction) |
| A domain name and a TLS certificate | Needed if clients connect over a network. Skip for a same-machine setup |
| A reverse proxy (recommended) | nginx, Caddy, Traefik or a cloud load balancer. It terminates TLS and adds rate limiting, which `maki serve` does not have |
| A way to keep secrets | A root-owned env file (this guide), or your platform's secret store (systemd credentials, Docker/Kubernetes secrets, a cloud secret manager) |
| An unprivileged account | `maki serve` never needs root or Administrator rights |

Decide now, because it changes later steps:

1. **Which backend?** See below. Hosted backends need a provider key and outbound internet. Local backends (llama.cpp, Ollama, vLLM) need the model server running and reachable from the Maki host.
2. **Which plugins?** Every plugin you list lets the model act on behalf of whoever holds the server key. Start with none or read-only ones. Do not enable `allow_dangerous_tools` on a network-facing agent (see [Step 8](#step-8-review-what-the-agent-is-allowed-to-do)).
3. **Who connects?** Only one process on the same machine (bind to `127.0.0.1`, no TLS needed) or other machines (reverse proxy with TLS).

## Which backend for production?

**We suggest llama.cpp** when you run the model yourself. Reasons:

- **It can be locked down.** `llama serve --api-key` makes the model server demand a key, so only the Maki process can use it. Ollama has no equivalent in Maki: its backend sends no credentials, so anyone who can reach the Ollama port can use the model.
- **Your data stays on your infrastructure** and there is no per-token bill or provider outage to plan for.
- **One small binary and a model file**, with the same behavior on Linux, macOS and Windows, on CPU or GPU. Pin the model file and you pin the behavior.
- **Maki supports it fully**: chat, streaming, tool calling, reasoning capture, embeddings and reranking, and a `health()` check.

When something else fits better: use **vLLM** when many agents hit one GPU at once (continuous batching), a **hosted provider** when you need the largest models or no hardware, and **Ollama** for development and quick local experiments.

One llama.cpp server holds one model in single-model mode, so size it for your load. `maki serve` runs one task at a time per agent, so a single server is enough for one served agent; see `llama serve --help` for parallelism settings if several programs share the model.

## Step 1: Install

Create a dedicated user and an isolated virtual environment, and install only the extras you need. `distributed` is always required for `maki serve`.

```bash
sudo useradd --system --create-home --home-dir /opt/maki --shell /usr/sbin/nologin maki
sudo -u maki python3 -m venv /opt/maki/venv
sudo -u maki /opt/maki/venv/bin/pip install --upgrade pip
```

Pick the line that matches your backend:

```bash
# Ollama, llama.cpp or vLLM server (llamacpp and vllm extras install the openai SDK)
sudo -u maki /opt/maki/venv/bin/pip install "maki-framework[distributed]"            # Ollama
sudo -u maki /opt/maki/venv/bin/pip install "maki-framework[distributed,llamacpp]"   # llama.cpp
sudo -u maki /opt/maki/venv/bin/pip install "maki-framework[distributed,vllm]"       # vLLM

# Hosted providers
sudo -u maki /opt/maki/venv/bin/pip install "maki-framework[distributed,openai]"
sudo -u maki /opt/maki/venv/bin/pip install "maki-framework[distributed,anthropic]"
sudo -u maki /opt/maki/venv/bin/pip install "maki-framework[distributed,openrouter]"
```

Add extras for the plugins you enable (for example `web` for `web_search` and `web_to_md`); see [Installation](installation.md#extras). Pin the version you tested (for example `maki-framework==0.6.0`) so a redeploy does not upgrade you by surprise.

Check it:

```bash
/opt/maki/venv/bin/maki
```

Without arguments it prints the usage help.

**Windows (PowerShell).** Install Python 3.10+ from python.org (tick "Add python.exe to PATH"). Create `C:\maki` from an Administrator PowerShell and give the service account (Step 6) write access to it, then run the commands below:

```powershell
python -m venv C:\maki\venv
C:\maki\venv\Scripts\python -m pip install --upgrade pip
C:\maki\venv\Scripts\pip install "maki-framework[distributed,llamacpp]"
C:\maki\venv\Scripts\maki
```

Everywhere below, `/opt/maki/venv/bin/maki` becomes `C:\maki\venv\Scripts\maki.exe` and `/opt/maki` becomes `C:\maki`. On macOS use the Linux commands, skipping the `useradd` line: create a dedicated standard user in System Settings (or with `sysadminctl`), and use a folder owned by it, for example `/usr/local/maki`, in place of `/opt/maki`.

## Step 2: Prepare the model server and the key it needs

### Option A (recommended): llama.cpp

1. Install llama.cpp so the `llama` command works (see the [llama.cpp docs](https://llama.app/docs/introduction); Linux, macOS and Windows are supported, with CPU and GPU builds). Check with `llama --help`.
2. Generate a **llama.cpp key**. You choose it, so make it random, and different from the server key of Step 3:

   ```bash
   python3 -c "import secrets; print(secrets.token_urlsafe(48))"
   ```

3. Start the model server on loopback only, with that key. The model downloads on first start (`-hf`), or point to a file you already have (`-m`):

   ```bash
   LLAMACPP_API_KEY=paste-the-llamacpp-key llama serve \
       -hf ggml-org/gemma-4-e4b-it-GGUF:Q4_0 \
       --host 127.0.0.1 --port 8080 --api-key "$LLAMACPP_API_KEY"
   ```

   ```powershell
   $env:LLAMACPP_API_KEY = "paste-the-llamacpp-key"
   llama serve -hf ggml-org/gemma-4-e4b-it-GGUF:Q4_0 --host 127.0.0.1 --port 8080 --api-key $env:LLAMACPP_API_KEY
   ```

   For production, run it under the same supervisor as Maki (Step 6) rather than in a terminal. Run `llama serve --help` for context size, GPU offload and parallel-slot options; they depend on your hardware and model and are not set in Maki.
4. Check it before involving Maki. It answers `401` without the key and `200` with it:

   ```bash
   curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8080/v1/models
   curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $LLAMACPP_API_KEY" http://127.0.0.1:8080/v1/models
   ```

   (While the model is still loading the server typically answers with an error such as `503`: wait and retry. If the `401`/`200` pair does not appear, check `llama serve --help` for how your version enforces `--api-key`.)

Points that matter:

- **Keep llama.cpp on `127.0.0.1`** when it runs on the same host as Maki. If it runs on another machine, set `base_url` in `agent.yaml` (Step 4), bind it to the private interface only, keep `--api-key` on, and firewall the port so only the Maki host reaches it.
- `LLAMACPP_API_KEY` appears **twice with the same value**: once to start `llama serve` (`--api-key`), once in Maki's environment (Step 5) so Maki can authenticate. If they differ, every request fails.
- If you start `llama serve` **with a model** (single-model mode) the model name in `agent.yaml` is ignored by the server. In **router mode** (started without a model) `model:` in `agent.yaml` selects which one to load.
- llama.cpp's HTTP traffic has no TLS here; that is fine on loopback or a private network, not across the internet.

### Option B: a hosted provider

Skip to the table below. Ollama (not recommended for production) needs no key: go to Step 3.

### Provider keys (hosted backends only)

| Backend | Where the key comes from | Variable Maki reads |
|---|---|---|
| OpenAI | platform.openai.com → API keys | `OPENAI_API_KEY` |
| Anthropic | console.anthropic.com → API keys | `ANTHROPIC_API_KEY` |
| OpenRouter | openrouter.ai → Keys | `OPENROUTER_API_KEY` |
| vLLM | You: the value you passed to `vllm serve --api-key` | `VLLM_API_KEY` (only if the server requires it) |
| llama.cpp | You: the value you passed to `llama serve --api-key` | `LLAMACPP_API_KEY` (only if the server requires it) |

Practical points:

- Create **one key per deployment** (not your personal key) and, where the provider supports it, set a spending limit on it so a leak has a ceiling.
- If a hosted key is missing, the backend raises an error at startup (`OpenAI API key not found. Pass api_key= or set OPENAI_API_KEY.`), so `maki serve` will not start. This is a good first check.
- Optional OpenRouter attribution headers: `OPENROUTER_SITE_URL` and `OPENROUTER_APP_NAME`.

## Step 3: Generate the server key

This is the key your clients will use. Make it long and random, and generate it on the server:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

Keep the output; you need it in Step 5 (server) and Step 10 (clients). Use a different value for each environment (staging, production).

How Maki checks it: every endpoint except `GET /health` requires `Authorization: Bearer <key>`; a missing or wrong key gets `401 Unauthorized`. The comparison is constant-time. There is **one shared key**: no users, scopes or per-client keys. To give each client its own key or to revoke one without touching the others, do that in your reverse proxy.

## Step 4: Write the agent config

Create `/opt/maki/agent.yaml`. It contains no secrets.

```yaml
name: support-agent
role: customer support assistant
instructions: |
  Answer briefly and politely. If you do not know, say so.

# backend: ollama | llamacpp | vllm | openai | anthropic | openrouter   (default: ollama)
backend: llamacpp
# In single-model mode the server ignores `model`; in router mode it selects the model.
# model: ggml-org/gemma-4-e4b-it-GGUF:Q4_0

# Where llama.cpp listens. Default http://127.0.0.1:8080/v1, so omit it when it runs on this host.
# base_url is honoured for llamacpp and vllm only. For Ollama set MAKI_OLLAMA_BASE_URL
# in the environment instead (Step 5). Hosted backends ignore it.
# base_url: http://10.0.0.5:8080/v1

# Hosted alternative:
# backend: anthropic
# model: claude-sonnet-4-5

temperature: 0.2

tool_policy:
  max_rounds: 6      # model/tool round trips before an answer is forced
  retries: 2         # extra attempts for transient tool errors

# Backends tried in order if the main one fails (network error, timeout, 408/429).
# Each hosted fallback needs its own provider key in the environment.
# fallback:
#   - openai:gpt-4o

plugins: []          # start empty; add only what you need
```

Things that trip people up:

- `model: "openai:gpt-4o"` (without a `backend:` line) selects the provider **and** the model.
- A `fallback:` entry to a different provider needs *that* provider's key in the environment too, and the matching extra installed.
- Unknown plugin names, unknown `tool_policy` fields and unknown `judge:`/`loop:` keys stop the server at startup with a clear error. That is intended.
- For every YAML option see [Serving agents](serving.md) and the schema at the top of `maki/distributed/config_loader.py`.

## Step 5: Put the secrets in the environment

`maki serve` reads secrets from **environment variables**. Create an env file readable only by root (systemd reads it as root before it drops to the `maki` user):

```bash
sudo install -m 600 -o root -g root /dev/null /etc/maki.env
sudoedit /etc/maki.env
```

```bash
# /etc/maki.env  (no quotes, no "export", one KEY=value per line)

# 1. Server key from Step 3: what clients must send
MAKI_API_KEY=paste-the-generated-key-here

# 2. Key for the model server / provider from Step 2 (only the one(s) you use)
# llama.cpp: the SAME value you gave `llama serve --api-key`
LLAMACPP_API_KEY=paste-the-llamacpp-key
# Hosted providers:
# OPENAI_API_KEY=sk-...
# OPENROUTER_API_KEY=sk-or-...
# VLLM_API_KEY=...

# 3. Ollama only: where the Ollama server is (not a secret)
# MAKI_OLLAMA_BASE_URL=http://10.0.0.5:11434

# Optional tuning
# MAKI_REQUEST_TIMEOUT=120
# MAKI_LOG_LEVEL=INFO
```

**macOS:** use the same file and put the same lines in the launchd plist's `EnvironmentVariables` (Step 6), or a root-owned file that a small wrapper script loads with `set -a; . /usr/local/etc/maki.env; set +a` before `exec maki serve ...`.

**Windows:** there is no `EnvironmentFile`. Set the variables **for the service only**, not machine-wide: with NSSM (Step 6) run `nssm set maki AppEnvironmentExtra MAKI_API_KEY=... LLAMACPP_API_KEY=...`. Do not use `setx` for production secrets (it writes them to the user's registry hive, readable by that user's processes). Restrict who can see the service configuration to Administrators.

!!! warning "Do not rely on a `.env` file in the working directory"
    The `python-dotenv` convenience loader searches for a `.env` file **upward from where the maki package is installed**, not from the directory you start the server in. In a virtual environment at `/opt/maki/venv` that means `/opt/maki/.env` could be picked up by accident, while `./.env` next to your `agent.yaml` is not read. Variables already in the real environment always win over a `.env` file. For production, set real environment variables (systemd `EnvironmentFile`, launchd `EnvironmentVariables`, NSSM `AppEnvironmentExtra`, container secrets, your platform's secret store). A `.env` file is fine for development in a source checkout.

Also:

- Pass the server key as `MAKI_API_KEY`, **not** `--api-key`. A flag shows in the process list (`ps`, Task Manager) for every user on the machine and in shell history; Maki prints a warning when you use it. If both are present the flag wins.
- Never put keys in the YAML, in a Docker image layer, in a git repository or in a chat message. If a key is exposed, rotate it at the provider (or generate a new server key) and restart.

## Step 6: Run it as a service

Create `/etc/systemd/system/maki.service`:

```ini
[Unit]
Description=Maki agent server
After=network-online.target
Wants=network-online.target

[Service]
User=maki
Group=maki
WorkingDirectory=/opt/maki
EnvironmentFile=/etc/maki.env
ExecStart=/opt/maki/venv/bin/maki serve \
    --config /opt/maki/agent.yaml \
    --host 127.0.0.1 --port 8100 \
    --max-body-mb 2 \
    --max-concurrency 8 \
    --graceful-timeout 120
Restart=on-failure
RestartSec=5
# Basic hardening
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/maki

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now maki
sudo systemctl status maki
journalctl -u maki -f
```

Run llama.cpp the same way, as a second unit, so it starts first and restarts on failure. Give it the key through its own environment and keep it on loopback:

```ini
# /etc/systemd/system/llama.service
[Unit]
Description=llama.cpp model server
After=network-online.target

[Service]
User=maki
EnvironmentFile=/etc/maki.env
# systemd expands ${VAR} from the environment file
# a local file: download the GGUF first, or use -hf as in Step 2 (it caches under the service user's home)
ExecStart=/usr/local/bin/llama serve -m /opt/maki/models/model.gguf --host 127.0.0.1 --port 8080 --api-key ${LLAMACPP_API_KEY}
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

and add `After=llama.service` and `Wants=llama.service` to `maki.service`. (A key in `ExecStart` is visible to local users in the process list, like any command-line key: on a shared host, restrict who can log in, or use a wrapper script that reads the key from the environment.)

**macOS (launchd).** Save as `/Library/LaunchDaemons/com.maki.serve.plist` (use `~/Library/LaunchAgents/` for a per-user service), then `sudo launchctl bootstrap system /Library/LaunchDaemons/com.maki.serve.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.maki.serve</string>
  <key>UserName</key><string>maki</string>
  <key>WorkingDirectory</key><string>/usr/local/maki</string>
  <key>ProgramArguments</key><array>
    <string>/usr/local/maki/venv/bin/maki</string><string>serve</string>
    <string>--config</string><string>/usr/local/maki/agent.yaml</string>
    <string>--host</string><string>127.0.0.1</string><string>--port</string><string>8100</string>
  </array>
  <key>EnvironmentVariables</key><dict>
    <key>MAKI_API_KEY</key><string>paste-the-server-key</string>
    <key>LLAMACPP_API_KEY</key><string>paste-the-llamacpp-key</string>
  </dict>
  <key>KeepAlive</key><true/>
  <key>RunAtLoad</key><true/>
  <key>StandardErrorPath</key><string>/usr/local/maki/maki.log</string>
</dict></plist>
```

Make the plist `chmod 600`, owned by root, because it holds the keys. Create a second plist the same way for `llama serve`.

**Windows (service).** Maki is a normal console program, so wrap it with a service manager. [NSSM](https://nssm.cc) is the simplest; WinSW or Task Scheduler ("run whether user is logged on or not", trigger "At startup") also work. In an Administrator PowerShell:

```powershell
nssm install maki C:\maki\venv\Scripts\maki.exe serve --config C:\maki\agent.yaml --host 127.0.0.1 --port 8100 --max-concurrency 8 --graceful-timeout 120
nssm set maki AppDirectory C:\maki
nssm set maki AppEnvironmentExtra MAKI_API_KEY=paste-the-server-key LLAMACPP_API_KEY=paste-the-llamacpp-key
nssm set maki AppStdout C:\maki\maki.log
nssm set maki AppStderr C:\maki\maki.log
nssm set maki ObjectName ".\svc-maki" "service-account-password"
nssm set maki Start SERVICE_AUTO_START
nssm start maki

# llama.cpp the same way, then make maki wait for it:
nssm install llama (Get-Command llama).Source serve -m C:\maki\models\model.gguf --host 127.0.0.1 --port 8080 --api-key paste-the-llamacpp-key
nssm set maki DependOnService llama
```

Passing keys on the `nssm` command line leaves them in the PowerShell history: clear it afterwards (`Clear-History`, and delete `(Get-PSReadLineOption).HistorySavePath`) or set them through `nssm edit maki` instead. Create `svc-maki` as a standard (non-admin) local account. Windows may prompt to allow `maki.exe` through the firewall; allow **private** networks only, or none when everything is on loopback.

What the options mean:

| Option | Meaning | Guidance |
|---|---|---|
| `--host 127.0.0.1` | Only this machine can connect | Keep it, and let the reverse proxy in Step 7 face the network |
| `--max-body-mb` | Largest request body (413 beyond it; default 4) | Lower it if your prompts are small |
| `--max-concurrency` | Requests in flight before a `503` (default: unlimited) | The agent runs **one task at a time** and the rest wait. Set about 2–4× the tasks you are willing to queue |
| `--graceful-timeout` | Seconds in-flight requests get on shutdown (default 30) | At least as long as your slowest request |

`ReadWritePaths` matters if you use plugins, loops or memory that write under `/opt/maki` or `~/.maki`; adjust the paths to what your agent actually writes.

On startup the server prints what it is doing. Confirm these lines:

```text
Maki Agent Server — 'support-agent' (customer support assistant)
  Listening on  http://127.0.0.1:8100
  Auth          Bearer token
  TLS           disabled
```

`Auth  disabled (trusted network)` means `MAKI_API_KEY` did not reach the process. The usual cause is a typo in the env file or a missing `EnvironmentFile=` line.

## Step 7: Put TLS in front

The bearer token and every prompt and answer are readable on the network without TLS. Two options:

**A. Reverse proxy (recommended).** Example nginx site (certificate obtained with certbot or your provider):

```nginx
limit_req_zone $binary_remote_addr zone=agent:10m rate=5r/s;   # must sit in the http {} context, outside server {}

server {
    listen 443 ssl;
    server_name agent.example.com;
    ssl_certificate     /etc/letsencrypt/live/agent.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/agent.example.com/privkey.pem;

    client_max_body_size 2m;
    location / {
        limit_req zone=agent burst=10 nodelay;
        proxy_pass http://127.0.0.1:8100;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_http_version 1.1;
        proxy_buffering off;          # keeps /stream (server-sent events) flowing
        proxy_read_timeout 300s;      # longer than your slowest model call
    }
}
```

Open only port 443 in the firewall. Port 8100 stays closed to the outside.

On Windows use Caddy (`caddy reverse-proxy --from agent.example.com --to 127.0.0.1:8100`, automatic certificates, installed as a service with NSSM) or IIS with URL Rewrite and Application Request Routing. On macOS, Caddy or nginx from Homebrew work the same as on Linux. Rate limiting is done by the proxy of your choice.

**B. Maki's own TLS**, when there is no proxy. Add `--tls-cert FILE --tls-key FILE` and bind to the interface you need (for example `--host 0.0.0.0`). You then lose rate limiting and IP allow-lists, so use a firewall to limit who can reach the port.

## Step 8: Review what the agent is allowed to do

Anyone holding the server key can make the agent do whatever its plugins allow.

- Keep `plugins:` to what the use case needs.
- Leave `allow_dangerous_tools` off (the default). It lets the model call file-writing, FTP and trading methods.
- Plugins that browse the web or read files act on the **server's** network and files; run the service as the unprivileged user from Step 1 and give it access only to the folders it needs.
- `/docs`, `/redoc` and `/openapi.json` are switched off automatically when a server key is set.

## Step 9: Verify the deployment

Run these from your own machine (replace the host and use your real key). Each one proves a different thing.

```bash
export MAKI_URL=https://agent.example.com
export MAKI_KEY=paste-the-server-key-here

# 1. Server and proxy are up. /health needs no key. Expect {"status":"ok",...}
curl -s $MAKI_URL/health

# 2. Authentication is enforced. Expect 401
curl -s -o /dev/null -w "%{http_code}\n" $MAKI_URL/info

# 3. Your key works and the right model is configured. Expect backend and model
curl -s -H "Authorization: Bearer $MAKI_KEY" $MAKI_URL/info

# 4. An end-to-end call reaches the model provider. Expect {"result": "...", ...}
curl -s -X POST $MAKI_URL/execute \
  -H "Authorization: Bearer $MAKI_KEY" -H "Content-Type: application/json" \
  -d '{"task": "Reply with the single word: pong"}'
```

On Windows PowerShell, `curl` is an alias for another command. Use `curl.exe` and single quotes around the JSON, with the quotes inside it doubled, or use `Invoke-RestMethod`:

```powershell
$env:MAKI_URL = "https://agent.example.com"; $env:MAKI_KEY = "paste-the-server-key"
curl.exe -s $env:MAKI_URL/health
curl.exe -s -o NUL -w "%{http_code}`n" $env:MAKI_URL/info
curl.exe -s -H "Authorization: Bearer $env:MAKI_KEY" $env:MAKI_URL/info
Invoke-RestMethod -Method Post -Uri "$env:MAKI_URL/execute" -Headers @{Authorization="Bearer $env:MAKI_KEY"} -ContentType "application/json" -Body '{"task":"Reply with the single word: pong"}'
```

Reading the results:

| Symptom | Likely cause |
|---|---|
| Step 1 fails to connect | Service not running, firewall, or proxy misconfigured (`journalctl -u maki`) |
| Step 2 returns `200` instead of `401` | `MAKI_API_KEY` did not reach the process (startup banner said `Auth disabled`) |
| Step 3 returns `401` | The client key differs from the server key (trailing newline or space when copying is common) |
| Step 4 returns `502 Agent backend unavailable` | The model server is unreachable from the Maki host: llama.cpp not running or still loading the model, wrong `base_url`, firewall |
| Step 4 returns `400 Agent backend rejected the request` | The model server or provider refused the call: wrong or expired key (for llama.cpp: `LLAMACPP_API_KEY` differs from the `--api-key` it was started with), unknown model name, or no credit. The exact reason is in the server log, not in the response |
| Step 4 returns `504` | The model took longer than `MAKI_REQUEST_TIMEOUT`; raise it, and the proxy's read timeout |
| Step 4 returns `503` | `--max-concurrency` reached; retry, or raise the limit |

To see how the server behaves under load before real users arrive, use the load check described in [Serving agents](serving.md#running-in-production). It is a development tool and is **not included in the pip package**, so run it from a source checkout (`git clone` + `pip install -e .`) on any machine that can reach the server:

```bash
python -m maki.test.load_check --url $MAKI_URL --api-key $MAKI_KEY --duration 60 --concurrency 8
```

## Step 10: Configure the clients

The client holds the **server key** only.

From Python:

```python
import os
from maki.distributed.proxy import AgentProxy

agent = AgentProxy(
    "https://agent.example.com",      # the endpoint: the public URL of your proxy
    api_key=os.environ["MAKI_KEY"],   # the server key, not a provider key
    timeout=120,                      # per request, in seconds; default 60. Raise it for slow models
)
print(agent.execute_task("Hello"))
```

From any HTTP client, send `Authorization: Bearer <key>` (see Step 9). Store the key in the client's own secret store, not in its source code.

## Step 11: Operate it

- **Logs:** `journalctl -u maki` (Linux), the file from `StandardErrorPath` (macOS), the `AppStdout` file (Windows). Errors returned to callers are generic; the detail is only in these logs. Credential-like query parameters in URLs are masked.
- **Health checks:** point your load balancer or monitor at `GET /health` (no key needed, and it answers even while every worker is busy). It does not prove the model provider is reachable; use `/execute` with a tiny task for that.
- **Rotate the server key:** generate a new one (Step 3), update the secrets file or service settings from Step 5, restart (`sudo systemctl restart maki`, `sudo launchctl kickstart -k system/com.maki.serve`, or `nssm restart maki`), then update the clients. There is no overlap period, so schedule it.
- **Rotate a provider or llama.cpp key:** create the new key (for llama.cpp, restart `llama serve` with it as well, and do both within the same maintenance window), update the secrets, restart Maki, then revoke the old key.
- **Upgrade:** `pip install "maki-framework[...]==NEW"` in the venv, read the [changelog](https://github.com/BowlOfData/maki/blob/main/CHANGELOG.md), restart, repeat Step 9.
- **Limits to plan for:** one agent per process (run several processes, on different ports, for several agents); a conversation is not kept across restarts; there is no metrics endpoint; there is no built-in rate limiting.

## Platform notes

Maki needs **no code changes** for any of the three systems; it is the same Python package. What differs is the surrounding operations, covered above. Be aware of what has been tested:

- **Linux** is the primary platform: the full test suite runs there in CI. **macOS** is covered only by the file-moving tests in CI, but it shares the POSIX code paths with Linux.
- **Windows is supported on a best-effort basis.** The code has Windows paths where the operating systems differ (file locking for loops, process termination, file moving), but CI runs only part of the tests there, and it has not been verified that the server under production load behaves identically. After installing on Windows, run the Step 9 checks and the load check before relying on it.
- Paths in `agent.yaml` (skills folders, loop directories) can use Windows paths; in YAML put them in single quotes (`'C:\maki\skills'`) so backslashes are kept literally.
- Loop requirements that run a command (`argv`) run directly, not through a shell, on every system: give the program and its arguments as a list, not a shell string, and make sure the program exists on the `PATH` of the **service account** (which differs from yours).
- Services start with a minimal environment: tools you can run in your own terminal (`llama`, `git`) may not be on the service's `PATH`. Use full paths in service definitions.

## Checklist

- [ ] Python 3.10+ on Linux, macOS or Windows, dedicated unprivileged user, virtual environment, extras installed, version pinned
- [ ] llama.cpp running on loopback (or a private network) with `--api-key`, and the same value in `LLAMACPP_API_KEY` for Maki; or a provider key with a spending limit. Keys stored only in the environment
- [ ] Server key generated with `secrets.token_urlsafe`, stored as `MAKI_API_KEY` (not `--api-key`)
- [ ] `agent.yaml` has no secrets; plugin list minimal; `allow_dangerous_tools` off
- [ ] Secrets file (`/etc/maki.env`, plist, or service settings on Windows) readable only by administrators; no reliance on a working-directory `.env`
- [ ] Service bound to `127.0.0.1`, behind TLS and rate limiting; only port 443 open
- [ ] `--max-concurrency`, `--max-body-mb` and `--graceful-timeout` set deliberately
- [ ] Startup banner shows `Auth  Bearer token`
- [ ] Step 9 checks pass: health `ok`, no key gives `401`, key gives `200`, `/execute` answers
- [ ] Clients hold only the server key
