#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.cell_identify import cell_identify


def _parse_kv_overrides(kv_list: List[str]) -> Dict[str, Any]:
    """
    Parse --set key=value pairs.
    Values are parsed as JSON when possible, so you can pass numbers/bools/lists:
      --set min_cells=5
      --set method="palma"
      --set palma_alpha=1e-6
      --set some_list='[1,2,3]'
    If JSON parsing fails, value is treated as a raw string.
    """
    overrides: Dict[str, Any] = {}
    for item in kv_list:
        if "=" not in item:
            raise ValueError(f"Invalid --set item '{item}'. Expected format key=value.")
        key, raw_val = item.split("=", 1)
        key = key.strip()
        raw_val = raw_val.strip()

        if not key:
            raise ValueError(f"Invalid --set item '{item}': empty key.")

        # Try JSON parse first (handles numbers, booleans, null, lists, quoted strings).
        try:
            val = json.loads(raw_val)
        except json.JSONDecodeError:
            # Fall back to plain string (useful for unquoted paths)
            val = raw_val

        overrides[key] = val
    return overrides


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_temp_config(base_cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Path:
    merged = dict(base_cfg)
    merged.update(overrides)

    # NamedTemporaryFile(delete=False) is more Windows-friendly.
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    try:
        json.dump(merged, tmp, indent=2, sort_keys=True)
        tmp.flush()
        return Path(tmp.name)
    finally:
        tmp.close()


def _expand_configs(patterns: List[str]) -> List[Path]:
    """
    Expand config paths and glob patterns into a de-duplicated list of Paths.
    """
    found: List[Path] = []
    for p in patterns:
        # Expand user (~) and environment variables
        expanded = os.path.expandvars(os.path.expanduser(p))

        # If it contains glob wildcards, expand
        if any(ch in expanded for ch in ["*", "?", "["]):
            matches = glob.glob(expanded)
            found.extend(Path(m) for m in matches)
        else:
            found.append(Path(expanded))

    # De-duplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for path in found:
        rp = str(path.resolve())
        if rp not in seen:
            seen.add(rp)
            unique.append(path)
    return unique


def run_one_config(config_path: Path, overrides: Dict[str, Any], keep_temp: bool, dry_run: bool) -> int:
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}", file=sys.stderr)
        return 2

    try:
        base_cfg = _load_json(config_path)
    except Exception as e:
        print(f"[ERROR] Failed to read JSON config: {config_path}\n  {e}", file=sys.stderr)
        return 2

    if overrides:
        try:
            tmp_config_path = _write_temp_config(base_cfg, overrides)
        except Exception as e:
            print(f"[ERROR] Failed to write temporary config for overrides.\n  {e}", file=sys.stderr)
            return 2
        effective_path = tmp_config_path
    else:
        tmp_config_path = None
        effective_path = config_path

    try:
        if dry_run:
            # Print the effective config and do not run.
            effective_cfg = _load_json(effective_path)
            print(f"--- Effective config ({effective_path}) ---")
            print(json.dumps(effective_cfg, indent=2, sort_keys=True))
            print("--- End effective config ---")
            return 0

        print(f"[INFO] Running PalmaClust with config: {config_path}")
        if overrides:
            print(f"[INFO] Applied overrides via temp config: {effective_path}")

        cell_identify(str(effective_path))

        print(f"[INFO] Done: {config_path}")
        return 0

    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"[ERROR] Run failed for config: {config_path}\n  {e}", file=sys.stderr)
        return 1
    finally:
        if tmp_config_path is not None and (not keep_temp):
            try:
                tmp_config_path.unlink(missing_ok=True)
            except Exception:
                # Not fatal
                pass


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="PalmaClust",
        description="Run PalmaClust rare cell type detection from one or more JSON config files.",
    )
    p.add_argument(
        "-c", "--config",
        nargs="+",
        required=True,
        help="Path(s) to JSON config file(s). Supports glob patterns, e.g. cfg/*.json",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        nargs="*",
        default=[],
        help="Override config fields using key=value. Values can be JSON (numbers/bools/lists). "
             "Example: --set output_folder=\"results/run1\" method=\"palma\" min_cells=5",
    )
    p.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary merged config file when using --set overrides.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the effective config (after overrides) and exit without running.",
    )
    return p


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()

    # Expand configs + parse overrides
    configs = _expand_configs(args.config)
    if not configs:
        print("[ERROR] No config files matched.", file=sys.stderr)
        return 2

    try:
        overrides = _parse_kv_overrides(args.overrides)
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2

    # Run each config; return non-zero if any fails
    worst_rc = 0
    for cfg_path in configs:
        rc = run_one_config(cfg_path, overrides, keep_temp=args.keep_temp, dry_run=args.dry_run)
        worst_rc = max(worst_rc, rc)
    return worst_rc


if __name__ == "__main__":
    raise SystemExit(main())
