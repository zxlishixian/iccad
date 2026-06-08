#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
secret_file="${ICCAD_NVIDIA_KEY_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/iccad/nvidia_api_key}"
config_file="${ICCAD_LLM_CONFIG_FILE:-$project_root/tools/llm_config_nvidia_qwen.yaml.example}"

if [[ ! -r "$secret_file" ]]; then
  echo "Missing NVIDIA key file: $secret_file" >&2
  echo 'Run tools/configure_nvidia_completion.sh once.' >&2
  exit 2
fi
if [[ ! -r "$config_file" ]]; then
  echo "Missing LLM config: $config_file" >&2
  exit 2
fi

export NVIDIA_API_KEY="$(<"$secret_file")"
export LLM_MODEL_CONFIG="$(<"$config_file")"
exec "$@"
