#!/usr/bin/env python3
"""Diagnose OpenAI-compatible completion endpoint configuration.

Reads LLM_MODEL_CONFIG (YAML content in env var) or a config file, runs
connectivity + API checks, and produces a diagnostic report. No training.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import yaml


def _parse_config(raw: str, source: str) -> dict | None:
    """Parse YAML config from string. Returns full parsed dict or None."""
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        print(f"[config] YAML parse error ({source}): {exc}", file=sys.stderr)
        return None
    if not isinstance(data, dict):
        print(f"[config] parsed value is not a mapping ({source})", file=sys.stderr)
        return None
    return data


def load_completion_section(data: dict) -> dict | None:
    """Extract completion section and flatten to a client-ready dict."""
    section = data.get("completion") or data.get("chat") or data.get("generation")
    if not isinstance(section, dict):
        return None
    cfg = section.get("config") if isinstance(section.get("config"), dict) else section
    model = section.get("model_name") or cfg.get("model")
    base_url = cfg.get("base_url")
    api_key = cfg.get("api_key")
    api_key_env = cfg.get("api_key_env")
    if not api_key and api_key_env:
        api_key = os.getenv(str(api_key_env), "")
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")
    return {
        "model_name": str(section.get("model_name", model or "")),
        "model": str(model or ""),
        "base_url": str(base_url or ""),
        "api_key": str(api_key or ""),
        "api_key_env": str(api_key_env or ""),
        "client_library": str(cfg.get("client_library", "openai.AsyncOpenAI")),
        "timeout": float(cfg.get("timeout", 30.0)),
        "max_tokens": int(cfg.get("max_tokens", 320)),
    }


# ── Section printers ─────────────────────────────────────────────────────

def check_config(args: argparse.Namespace) -> dict:
    report: dict = {"config": {}, "network": {}, "client": {}, "completion": {}, "models": {}}
    print("\n" + "=" * 60)
    print("1. CONFIG PARSING")
    print("=" * 60)

    raw = ""
    source = "none"

    # --- Check if LLM_MODEL_CONFIG looks like a file path ---
    env_val = os.getenv(args.config_env, "")
    if env_val:
        env_val_stripped = env_val.strip()
        if env_val_stripped.endswith((".yaml", ".yml")) and "\n" not in env_val_stripped:
            print(f"[config] WARNING: {args.config_env} appears to be a file path, not YAML content!")
            print(f"  Value: {env_val_stripped}")
            print(f"  Use: export {args.config_env}=\"$(cat {env_val_stripped})\"")
            if not args.config_file:
                # Try to read as file as a fallback
                try:
                    raw = Path(env_val_stripped).read_text(encoding="utf-8")
                    source = f"file_path_from_env ({env_val_stripped})"
                    print(f"[config] fallback: read file content ({len(raw)} chars)")
                except OSError as exc:
                    print(f"[config] fallback failed: {exc}")
                    raw = ""
                    source = "invalid_file_path"
        else:
            raw = env_val
            source = f"env:{args.config_env}"

    if args.config_file:
        try:
            raw = Path(args.config_file).read_text(encoding="utf-8")
            source = f"file:{args.config_file}"
            print(f"[config] source={source} ({len(raw)} chars)")
        except OSError as exc:
            print(f"[config] ERROR reading file: {exc}")
            return report

    if not raw:
        print("[config] no config source available")
        report["config"]["error"] = "no config source"
        return report

    print(f"[config] source={source}")

    data = _parse_config(raw, source)
    if data is None:
        report["config"]["error"] = "yaml_parse_failed"
        return report

    has_completion = "completion" in data or "chat" in data or "generation" in data
    print(f"[config] has_completion={has_completion}")

    if not has_completion:
        print("[config] WARNING: no completion/chat/generation section found")
        report["config"]["error"] = "no_completion_section"
        return report

    cfg = load_completion_section(data)
    report["config"] = {
        "source": source,
        "has_completion": True,
        "model_name": cfg.get("model_name"),
        "model": cfg.get("model"),
        "base_url": cfg.get("base_url"),
        "client_library": cfg.get("client_library"),
        "api_key_env": cfg.get("api_key_env"),
        "api_key_present": bool(cfg.get("api_key")),
        "api_key_length": len(cfg.get("api_key", "")),
        "timeout": cfg.get("timeout"),
        "max_tokens": cfg.get("max_tokens"),
    }

    for k, v in report["config"].items():
        print(f"[config] {k}={v}")

    # API key warning
    if not cfg.get("api_key"):
        print("\n*** WARNING: api_key is empty! ***")
        print("  The OpenAI Python client sends 'Authorization: Bearer <key>'")
        print("  An empty key results in 'Authorization: Bearer ' which is an illegal HTTP header.")
        print("  Set NVIDIA_API_KEY or add api_key to the completion config.")
        if cfg.get("api_key_env"):
            print(f"  Current api_key_env: {cfg['api_key_env']}")
            print(f"  Env var value: '{os.getenv(str(cfg['api_key_env']), '')}'")

    report["_cfg"] = cfg  # pass through for later sections
    return report


def check_network(report: dict) -> dict:
    cfg = report.get("_cfg", {})
    base_url = cfg.get("base_url", "")
    print("\n" + "=" * 60)
    print("2. NETWORK CONNECTIVITY")
    print("=" * 60)

    net = {}
    if not base_url:
        print("[network] no base_url to check")
        net["error"] = "no_base_url"
        report["network"] = net
        return report

    print(f"[network] base_url={base_url}")

    # Check proxy env
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        val = os.getenv(var)
        if val:
            print(f"[network] {var}=set (len={len(val)})")
        else:
            print(f"[network] {var}=not set")

    # Try GET /models
    import urllib.request
    models_url = base_url.rstrip("/") + "/models"
    print(f"[network] trying GET {models_url} ...")
    try:
        req = urllib.request.Request(models_url)
        # Add auth header if we have a key
        if cfg.get("api_key"):
            req.add_header("Authorization", f"Bearer {cfg['api_key']}")
        resp = urllib.request.urlopen(req, timeout=10)
        status = resp.status
        body = resp.read().decode("utf-8", errors="replace")[:500]
        print(f"[network] models_status={status}")
        print(f"[network] models_body_excerpt={body[:200]}")
        net["models_status"] = status
        net["models_ok"] = 200 <= status < 300

        # Try to parse model list
        try:
            models_data = json.loads(body)
            if "data" in models_data:
                models = [m.get("id", str(m)) for m in models_data["data"][:20]]
                net["available_models"] = models
                print(f"[network] available_models (first 20): {models}")
        except (json.JSONDecodeError, KeyError):
            pass
    except Exception as exc:
        print(f"[network] models_request_failed: {type(exc).__name__}: {exc}")
        net["models_status"] = "failed"
        net["models_error"] = f"{type(exc).__name__}: {exc}"
        net["models_ok"] = False

    report["network"] = net
    return report


def check_client(report: dict) -> dict:
    print("\n" + "=" * 60)
    print("3. OPENAI CLIENT INIT")
    print("=" * 60)

    client_info = {}
    try:
        from openai import OpenAI, AsyncOpenAI
        client_info["openai_import_ok"] = True
        print("[client] openai_import_ok=true")
    except ImportError as exc:
        client_info["openai_import_ok"] = False
        client_info["error"] = str(exc)
        print(f"[client] openai_import_ok=false ({exc})")
        print("[client] FIX: pip install openai")
        report["client"] = client_info
        return report

    cfg = report.get("_cfg", {})
    try:
        client = OpenAI(
            api_key=cfg.get("api_key") or "dummy",
            base_url=cfg.get("base_url", ""),
            timeout=cfg.get("timeout", 30.0),
        )
        client_info["client_init_ok"] = True
        print("[client] client_init_ok=true")
    except Exception as exc:
        client_info["client_init_ok"] = False
        client_info["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[client] client_init_ok=false ({exc})")

    report["client"] = client_info
    return report


def check_completion(report: dict) -> dict:
    cfg = report.get("_cfg", {})
    print("\n" + "=" * 60)
    print("4. COMPLETION API CALL")
    print("=" * 60)

    comp = {}
    try:
        from openai import OpenAI
    except ImportError:
        comp["ok"] = False
        comp["error"] = "openai_not_installed"
        report["completion"] = comp
        return report

    api_key = cfg.get("api_key", "")
    if not api_key:
        print("[completion] WARNING: api_key is empty, using 'dummy' to avoid illegal header")
        print("  The real API will likely return 401 Unauthorized.")
        api_key = "dummy"

    client = OpenAI(
        api_key=api_key,
        base_url=cfg.get("base_url", ""),
        timeout=cfg.get("timeout", 30.0),
    )
    model = cfg.get("model", "")

    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": 'Return: {"ok": true, "failure_stage": "unknown"}'},
            ],
            temperature=0,
            top_p=1,
            max_tokens=128,
        )
        latency = time.perf_counter() - t0
        text = response.choices[0].message.content or ""
        comp["ok"] = True
        comp["latency_sec"] = round(latency, 3)
        comp["response_excerpt"] = text[:200]
        print(f"[completion] request_ok=true latency_sec={latency:.3f}")
        print(f"[completion] response_excerpt={text[:200]}")

        # Try to parse as JSON
        try:
            json.loads(text.strip())
            comp["parsed_json_ok"] = True
            print("[completion] parsed_json_ok=true")
        except (json.JSONDecodeError, ValueError):
            comp["parsed_json_ok"] = False
            print("[completion] parsed_json_ok=false")
    except Exception as exc:
        latency = time.perf_counter() - t0
        comp["ok"] = False
        comp["latency_sec"] = round(latency, 3)
        comp["exception_type"] = type(exc).__name__
        comp["exception_repr"] = repr(exc)
        print(f"[completion] request_ok=false")
        print(f"[completion] exception={type(exc).__name__}")
        print(f"[completion] exception_repr={exc}")

        # Diagnose
        causes = []
        exc_str = str(exc).lower()
        exc_type = type(exc).__name__
        if "illegal header" in exc_str or "bearer" in exc_str:
            causes.append("API key is empty or malformed → illegal Authorization header")
        if "401" in exc_str or "unauthorized" in exc_str or "authentication" in exc_str:
            causes.append("API key is invalid or missing (401 Unauthorized)")
        if "403" in exc_str or "forbidden" in exc_str:
            causes.append("API key lacks permission (403 Forbidden)")
        if "404" in exc_str or "not found" in exc_str:
            causes.append(f"Model '{model}' not found at endpoint")
        if "connection" in exc_str or "timeout" in exc_str or "connect" in exc_str:
            causes.append("base_url unreachable or timeout")
            causes.append("Check: firewall, proxy, VPN, or network connectivity")
        if "proxy" in exc_str or "PROXY" in exc_str:
            causes.append("Proxy configuration issue")
        if "rate" in exc_str or "429" in exc_str:
            causes.append("Rate limited (429)")

        if not causes:
            causes = [
                f"Unknown error: {type(exc).__name__}",
                "Check: base_url reachable, API key valid, model name correct",
            ]
        comp["possible_causes"] = causes
        for c in causes:
            print(f"[completion] possible_cause: {c}")

    report["completion"] = comp
    return report


def check_cache(args: argparse.Namespace) -> dict:
    print("\n" + "=" * 60)
    print("5. COMPLETION CACHE STATUS")
    print("=" * 60)

    cache_info = {"cache_dir": str(args.cache_dir)}
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_dir():
        print(f"[cache] cache_dir={cache_dir} does not exist")
        cache_info["exists"] = False
        cache_info["num_files"] = 0
        print("No action needed.")
        return cache_info

    files = list(cache_dir.glob("*.json"))
    cache_info["exists"] = True
    cache_info["num_files"] = len(files)
    print(f"[cache] num_files={len(files)}")

    if args.clear_cache:
        for f in files:
            f.unlink()
        print(f"[cache] cleared {len(files)} files")
        cache_info["cleared"] = len(files)
        return cache_info

    # Check for error caches (status != ok)
    error_caches = []
    ok_caches = 0
    for f in files[:50]:  # sample first 50
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if data.get("status") != "ok":
                error_caches.append(f.name)
            else:
                ok_caches += 1
        except (json.JSONDecodeError, OSError):
            error_caches.append(f.name)

    cache_info["ok_count_sample"] = ok_caches
    cache_info["error_count_sample"] = len(error_caches)
    print(f"[cache] ok={ok_caches} error={len(error_caches)} (sampled {min(50, len(files))})")
    if error_caches:
        print(f"[cache] error examples: {error_caches[:5]}")
        print("[cache] WARNING: error caches will cause false 'completion ok' if reused")

    return cache_info


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose OpenAI-compatible completion endpoint")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--config-env", default="LLM_MODEL_CONFIG",
                   help="Environment variable containing YAML config (default: LLM_MODEL_CONFIG)")
    g2 = p.add_mutually_exclusive_group()
    g2.add_argument("--config-file", type=Path,
                    help="Path to YAML config file (overrides env)")
    p.add_argument("--num-examples", type=int, default=1)
    p.add_argument("--timeout-sec", type=float, default=30.0)
    p.add_argument("--cache-dir", type=Path, default=Path("/tmp/regr_fail_completion_cache_check"))
    p.add_argument("--clear-cache", action="store_true")
    p.add_argument("--strict", action="store_true", help="Exit non-zero on any failure")
    p.add_argument("--output-json", type=Path, help="Write JSON diagnostic report")
    p.add_argument("--output-md", type=Path, help="Write markdown diagnostic report")
    return p.parse_args(argv)


def build_report(args: argparse.Namespace) -> dict:
    report: dict = {"config": {}, "network": {}, "client": {}, "completion": {}, "cache": {}}

    # 1. Config
    if args.config_file:
        args.config_env = "FILE_MODE"  # hack: prevent env read
    report = check_config(args)
    if report["config"].get("error"):
        report["cache"] = check_cache(args)
        return report

    # 2. Network
    report = check_network(report)

    # 3. Client init
    report = check_client(report)

    # 4. Completion API call
    for i in range(args.num_examples):
        if i > 0:
            print(f"\n--- Example {i + 1}/{args.num_examples} ---")
        report = check_completion(report)

    # 5. Cache status
    report["cache"] = check_cache(args)

    return report


def print_summary(report: dict) -> str:
    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)

    issues = []
    ok = []

    cfg = report.get("config", {})
    if cfg.get("error"):
        issues.append(f"Config error: {cfg['error']}")
    elif cfg.get("has_completion"):
        ok.append("Config parsed, completion section found")
        if not cfg.get("api_key_present"):
            issues.append("API KEY IS EMPTY — 'Bearer ' illegal header → connection will fail")
        else:
            ok.append(f"API key present (len={cfg.get('api_key_length', 0)})")

    net = report.get("network", {})
    if net.get("models_ok"):
        ok.append(f"Network OK: GET /models returned {net.get('models_status')}")
    elif net.get("error"):
        issues.append(f"Network: {net['error']}")
    else:
        issues.append(f"Network: /models request failed ({net.get('models_error', 'unknown')})")

    client = report.get("client", {})
    if client.get("openai_import_ok") and client.get("client_init_ok"):
        ok.append("OpenAI client init OK")
    else:
        issues.append("OpenAI client: not OK")

    comp = report.get("completion", {})
    if comp.get("ok"):
        ok.append(f"Completion API call OK (latency={comp.get('latency_sec', 0):.1f}s)")
    else:
        issues.append(f"Completion failed: {comp.get('exception_type', 'unknown')}")

    print("\nPASSES:")
    for item in ok:
        print(f"  ✓ {item}")
    print("\nISSUES:")
    for item in issues:
        print(f"  ✗ {item}")

    if not issues:
        print("\n  All checks passed. Completion endpoint should work.")
    else:
        print("\n  RECOMMENDED FIXES:")
        if not cfg.get("api_key_present"):
            print("  1. Set NVIDIA_API_KEY to a valid key")
            print("     export NVIDIA_API_KEY='nvapi-...'")
        print("  2. Verify: curl -I https://integrate.api.nvidia.com/v1/models")
        print("  3. Run this script again after fixing")

    return "PASS" if not issues else "FAIL"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    # Build report
    report = build_report(args)

    # Clean up internal key
    report.pop("_cfg", None)

    # Summary
    result = print_summary(report)

    # Write outputs
    if args.output_json:
        args.output_json.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"\nJSON report: {args.output_json}")

    if args.output_md:
        _write_md_report(args.output_md, report)
        print(f"Markdown report: {args.output_md}")

    if args.strict and result == "FAIL":
        return 1
    return 0


def _write_md_report(path: Path, report: dict) -> None:
    lines = ["# Completion Endpoint Diagnostic Report", ""]
    for section, data in report.items():
        lines.append(f"## {section}")
        lines.append("```")
        for k, v in data.items():
            lines.append(f"  {k}: {v}")
        lines.append("```")
        lines.append("")
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
