#!/usr/bin/env bash
set -euo pipefail

config_dir="${XDG_CONFIG_HOME:-$HOME/.config}/iccad"
secret_file="$config_dir/nvidia_api_key"
mkdir -p "$config_dir"
chmod 700 "$config_dir"

printf 'NVIDIA API key: ' >&2
IFS= read -r -s api_key
printf '\n' >&2
if [[ -z "$api_key" ]]; then
  echo 'Empty key; configuration was not changed.' >&2
  exit 2
fi
umask 077
printf '%s' "$api_key" > "$secret_file"
unset api_key
chmod 600 "$secret_file"
echo "Saved NVIDIA key to $secret_file (mode 600)." >&2
echo 'Run: tools/with_nvidia_completion.sh python tools/check_completion_endpoint.py' >&2
