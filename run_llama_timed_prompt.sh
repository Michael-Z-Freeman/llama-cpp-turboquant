#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$SCRIPT_DIR/build/bin/llama-server}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3-14B-Q6_K.gguf}"
PORT="${PORT:-8081}"
HOST="127.0.0.1"
CTX_SIZE="${CTX_SIZE:-32768}"
CACHE_TYPE_K="${CACHE_TYPE_K:-turbo4}"
CACHE_TYPE_V="${CACHE_TYPE_V:-turbo4}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
SPLIT_MODE="${SPLIT_MODE:-row}"
FLASH_ATTN="${FLASH_ATTN:-on}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
THINK_TEMPERATURE="${THINK_TEMPERATURE:-0.6}"
THINK_TOP_P="${THINK_TOP_P:-0.95}"
NONTHINK_TEMPERATURE="${NONTHINK_TEMPERATURE:-0.7}"
NONTHINK_TOP_P="${NONTHINK_TOP_P:-0.8}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"
NO_CONTEXT_SHIFT="${NO_CONTEXT_SHIFT:-1}"
USE_YARN="${USE_YARN:-0}"
YARN_ORIG_CTX="${YARN_ORIG_CTX:-32768}"
ROPE_SCALE="${ROPE_SCALE:-4}"
THINK_MODE="${THINK_MODE:-off}"
CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-}"

if [[ ! -x "$LLAMA_SERVER_BIN" ]]; then
  echo "Error: llama-server not executable at: $LLAMA_SERVER_BIN" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Error: model not found at: $MODEL_PATH" >&2
  exit 1
fi

if [[ -n "$CHAT_TEMPLATE_FILE" ]] && [[ ! -f "$CHAT_TEMPLATE_FILE" ]]; then
  echo "Error: chat template file not found at: $CHAT_TEMPLATE_FILE" >&2
  exit 1
fi

LOG_FILE="$(mktemp "${TMPDIR:-/tmp}/llama-server-timed.XXXXXX.log")"
RESP_FILE="$(mktemp "${TMPDIR:-/tmp}/llama-response.XXXXXX.json")"
METRICS_FILE="$(mktemp "${TMPDIR:-/tmp}/llama-metrics.XXXXXX.txt")"
STARTED_SERVER=0
SERVER_PID=""

cleanup() {
  if [[ "$STARTED_SERVER" -eq 1 ]] && [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo
    echo "Shutting down llama-server (pid=$SERVER_PID) ..."
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -f "$RESP_FILE" "$METRICS_FILE"
}
trap cleanup EXIT

echo "Starting llama-server on http://$HOST:$PORT ..."

SERVER_ARGS=(
  -m "$MODEL_PATH"
  --jinja
  --gpu-layers "$N_GPU_LAYERS"
  --split-mode "$SPLIT_MODE"
  --flash-attn "$FLASH_ATTN"
  --ctx-size "$CTX_SIZE"
  --cache-type-k "$CACHE_TYPE_K"
  --cache-type-v "$CACHE_TYPE_V"
  --parallel 1
  --host "$HOST"
  --port "$PORT"
)

if [[ -n "$CHAT_TEMPLATE_FILE" ]]; then
  SERVER_ARGS+=(--chat-template-file "$CHAT_TEMPLATE_FILE")
fi

if [[ "$NO_CONTEXT_SHIFT" == "1" ]]; then
  SERVER_ARGS+=(--no-context-shift)
fi

if [[ "$USE_YARN" == "1" ]]; then
  SERVER_ARGS+=(--rope-scaling yarn --rope-scale "$ROPE_SCALE" --yarn-orig-ctx "$YARN_ORIG_CTX")
fi

"$LLAMA_SERVER_BIN" \
  "${SERVER_ARGS[@]}" \
  >"$LOG_FILE" 2>&1 &

SERVER_PID="$!"
STARTED_SERVER=1

READY=0
for _ in $(seq 1 120); do
  if curl -sSf "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 0.5
done

if [[ "$READY" -ne 1 ]]; then
  echo "Error: server did not become ready" >&2
  echo "--- server log tail ---" >&2
  tail -n 60 "$LOG_FILE" >&2 || true
  exit 1
fi

MODEL_NAME="$(basename "$MODEL_PATH")"
if [[ "$THINK_MODE" == "on" ]]; then
  THINK_PREFIX="/think "
else
  THINK_PREFIX="/no_think "
fi

echo "Ready. Enter prompts to test latency."
echo "Type /status to show server/GPU status hints."
echo "Type /think or /no_think to toggle reasoning mode."
echo "Type /exit to stop and shut down the server."
echo "thinking_mode=$THINK_MODE"
echo "server_flags: gpu_layers=$N_GPU_LAYERS split_mode=$SPLIT_MODE flash_attn=$FLASH_ATTN"
echo "sampling(thinking): temp=$THINK_TEMPERATURE top_k=$TOP_K top_p=$THINK_TOP_P min_p=$MIN_P presence_penalty=$PRESENCE_PENALTY"
echo "sampling(non_thinking): temp=$NONTHINK_TEMPERATURE top_k=$TOP_K top_p=$NONTHINK_TOP_P min_p=$MIN_P presence_penalty=$PRESENCE_PENALTY"
echo "max_tokens=$MAX_TOKENS"
echo "no_context_shift=$NO_CONTEXT_SHIFT"
echo "use_yarn=$USE_YARN rope_scale=$ROPE_SCALE yarn_orig_ctx=$YARN_ORIG_CTX"
if [[ -n "$CHAT_TEMPLATE_FILE" ]]; then
  echo "chat_template_file=$CHAT_TEMPLATE_FILE"
else
  echo "chat_template_file=(model default)"
fi
echo "server_log=$LOG_FILE"

show_status() {
  echo
  echo "status:"
  echo "  server_pid=$SERVER_PID"
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "  process_alive=yes"
  else
    echo "  process_alive=no"
  fi

  if curl -sSf "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
    echo "  api_health=ok"
  else
    echo "  api_health=unreachable"
  fi

  echo "  endpoint=http://$HOST:$PORT"
  echo "  model_path=$MODEL_PATH"
  echo "  thinking_mode=$THINK_MODE"
  echo "  server_flags: gpu_layers=$N_GPU_LAYERS split_mode=$SPLIT_MODE flash_attn=$FLASH_ATTN"
  echo "  sampling(thinking): temp=$THINK_TEMPERATURE top_k=$TOP_K top_p=$THINK_TOP_P min_p=$MIN_P presence_penalty=$PRESENCE_PENALTY"
  echo "  sampling(non_thinking): temp=$NONTHINK_TEMPERATURE top_k=$TOP_K top_p=$NONTHINK_TOP_P min_p=$MIN_P presence_penalty=$PRESENCE_PENALTY"
  echo "  max_tokens=$MAX_TOKENS"
  echo "  no_context_shift=$NO_CONTEXT_SHIFT"
  echo "  use_yarn=$USE_YARN rope_scale=$ROPE_SCALE yarn_orig_ctx=$YARN_ORIG_CTX"
  if [[ -n "$CHAT_TEMPLATE_FILE" ]]; then
    echo "  chat_template_file=$CHAT_TEMPLATE_FILE"
  else
    echo "  chat_template_file=(model default)"
  fi
  echo "  server_log=$LOG_FILE"
  echo
  echo "recent offload-related log lines:"
  if ! rg -n "offload|ROCm|HIP|CUDA|layers offloaded|memory|VMM|KV|cache" "$LOG_FILE" | tail -n 25; then
    tail -n 25 "$LOG_FILE" || true
  fi
  echo
}

ensure_server_alive() {
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Error: llama-server is not running (pid=$SERVER_PID)." >&2
    echo "--- server log tail ---" >&2
    tail -n 80 "$LOG_FILE" >&2 || true
    exit 1
  fi
}

time_request() {
  local user_prompt="$1"
  local effective_prompt="${THINK_PREFIX}${user_prompt}"
  local req_temp req_top_p

  if [[ "$THINK_MODE" == "on" ]]; then
    req_temp="$THINK_TEMPERATURE"
    req_top_p="$THINK_TOP_P"
  else
    req_temp="$NONTHINK_TEMPERATURE"
    req_top_p="$NONTHINK_TOP_P"
  fi

  ensure_server_alive

  local prompt_json
  prompt_json="$(python3 - <<'PY' "$effective_prompt"
import json, sys
print(json.dumps(sys.argv[1]))
PY
)"

  local request_json
  request_json="$(cat <<EOF
{"model":"$MODEL_NAME","messages":[{"role":"user","content":$prompt_json}],"temperature":$req_temp,"top_k":$TOP_K,"top_p":$req_top_p,"min_p":$MIN_P,"presence_penalty":$PRESENCE_PENALTY,"max_tokens":$MAX_TOKENS,"stream":false}
EOF
)"

  local start_ms end_ms elapsed_ms
  start_ms="$(python3 - <<'PY'
import time
print(int(time.time() * 1000))
PY
)"

  if ! curl -sS \
    -o "$RESP_FILE" \
    -w 'time_total=%{time_total}\ntime_starttransfer=%{time_starttransfer}\n' \
    -H 'Content-Type: application/json' \
    -d "$request_json" \
    "http://$HOST:$PORT/v1/chat/completions" \
    > "$METRICS_FILE"; then
    echo
    echo "Error: request failed (curl)." >&2
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "llama-server is still running (pid=$SERVER_PID)." >&2
    else
      echo "llama-server crashed or exited (pid=$SERVER_PID)." >&2
    fi
    echo "--- server log tail ---" >&2
    tail -n 120 "$LOG_FILE" >&2 || true
    return 1
  fi

  if [[ ! -s "$RESP_FILE" ]]; then
    echo
    echo "Error: empty response body from server." >&2
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "llama-server is still running (pid=$SERVER_PID)." >&2
    else
      echo "llama-server crashed or exited (pid=$SERVER_PID)." >&2
    fi
    echo "--- server log tail ---" >&2
    tail -n 120 "$LOG_FILE" >&2 || true
    return 1
  fi

  end_ms="$(python3 - <<'PY'
import time
print(int(time.time() * 1000))
PY
)"
  elapsed_ms=$((end_ms - start_ms))

  echo
  echo "elapsed_ms=$elapsed_ms"
  cat "$METRICS_FILE"

  if ! python3 - <<'PY' "$RESP_FILE"
import json, sys
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = json.load(f)
msg = data.get('choices', [{}])[0].get('message', {}).get('content', '')
print('reply_preview=' + repr(msg[:220]))
print('usage=' + json.dumps(data.get('usage', {})))
PY
  then
    echo "Error: failed to parse JSON response." >&2
    echo "--- raw response ---" >&2
    cat "$RESP_FILE" >&2 || true
    return 1
  fi
  echo
}

while true; do
  read -r -p "Prompt($THINK_MODE)> " USER_PROMPT || break

  if [[ "${USER_PROMPT}" == "/status" ]]; then
    show_status
    continue
  fi

  if [[ "${USER_PROMPT}" == "/think" ]]; then
    THINK_MODE="on"
    THINK_PREFIX="/think "
    echo "thinking_mode=on"
    continue
  fi

  if [[ "${USER_PROMPT}" == "/no_think" ]]; then
    THINK_MODE="off"
    THINK_PREFIX="/no_think "
    echo "thinking_mode=off"
    continue
  fi

  if [[ "${USER_PROMPT}" == "/exit" ]]; then
    break
  fi

  if [[ -z "${USER_PROMPT// }" ]]; then
    echo "(empty prompt skipped)"
    continue
  fi

  ensure_server_alive
  if ! time_request "$USER_PROMPT"; then
    echo "Request failed. Use /status for diagnostics or /exit to stop." >&2
  fi
done
