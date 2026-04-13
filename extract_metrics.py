#!/usr/bin/env python3
"""
Extract metrics (ECP, WNS, TNS, TNS_EVAL, Detailed Wirelength, CTS Wirelength)
and optimization parameters from nested logs:
Structure: backup_dir/platform/design/result_dump_k/logs_dump/*_runi.log
           backup_dir/platform/design/result_dump_k/config_i.mk
           backup_dir/platform/design/result_dump_k/constraint_i.sdc
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
from pathlib import Path
from typing import Iterable, Tuple, Dict, Optional, List


# --- Metric extraction patterns ---

ECP_PATTERN = re.compile(
    r'Report metrics stage 6, finish[\s\S]*?(?:core_)?cl[ock]*\s+period_min\s*=\s*([-\d.]+)'
)

DWL_PATTERN = re.compile(
    r'\[INFO DRT-0198\] Complete detail routing\..*?Total wire length\s*=\s*([\d.]+)',
    re.DOTALL
)

CTS_WL_PATTERN = re.compile(r'Total wirelength:\s*([\d.]+)')

WNS_PATTERN = re.compile(
    r'Report metrics stage 6, finish[\s\S]*?wns max\s+([-\d.]+)'
)

TNS_PATTERN = re.compile(
    r'Report metrics stage 6, finish[\s\S]*?tns max\s+([-\d.]+)'
)

TNS_EVAL_PATTERN = re.compile(
    r'\[TNS_EVAL\] tns_eval = ([-\d.eE+]+)'
)

DRC_PATTERN = re.compile(
    r'\[INFO DRT-0199\].*?Number of violations\s*=\s*(\d+)'
)


def extract_metrics_from_log(log_path: Path) -> Dict[str, Optional[float]]:

    metrics = {
        "ecp": None,
        "dwl": None,
        "cts_wl": None,
        "wns": None,
        "tns": None,
        "tns_eval": None,
        "drc": None,
    }

    try:
        content = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARN] Failed to read {log_path}: {e}")
        return metrics

    ecp_match = ECP_PATTERN.search(content)
    if ecp_match:
        metrics['ecp'] = float(ecp_match.group(1))

    dwl_match = DWL_PATTERN.search(content)
    if dwl_match:
        metrics['dwl'] = float(dwl_match.group(1))

    cts_wl_match = CTS_WL_PATTERN.search(content)
    if cts_wl_match:
        metrics['cts_wl'] = float(cts_wl_match.group(1))

    wns_match = WNS_PATTERN.search(content)
    if wns_match:
        metrics['wns'] = float(wns_match.group(1))

    tns_match = TNS_PATTERN.search(content)
    if tns_match:
        metrics['tns'] = float(tns_match.group(1))

    tns_eval_match = TNS_EVAL_PATTERN.search(content)
    if tns_eval_match:
        metrics['tns_eval'] = float(tns_eval_match.group(1))

    # DRC: take the last occurrence (final iteration of detail routing)
    drc_matches = DRC_PATTERN.findall(content)
    if drc_matches:
        metrics['drc'] = float(drc_matches[-1])

    return metrics


# --- Parameter extraction from config/SDC files ---

def extract_params_from_config(config_path: Path) -> Dict[str, Optional[str]]:
    """Extract optimization parameters from a config_N.mk file."""
    params = {
        "core_util": None,
        "cell_pad_global": None,
        "cell_pad_detail": None,
        "enable_dpo": None,
    }
    if not config_path.is_file():
        return params
    try:
        content = config_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return params

    patterns = {
        "core_util": r'CORE_UTILIZATION\s*=\s*(\S+)',
        "cell_pad_global": r'CELL_PAD_IN_SITES_GLOBAL_PLACEMENT\s*=\s*(\S+)',
        "cell_pad_detail": r'CELL_PAD_IN_SITES_DETAIL_PLACEMENT\s*=\s*(\S+)',
        "enable_dpo": r'ENABLE_DPO\s*=\s*(\S+)',
    }
    for key, pat in patterns.items():
        m = re.search(pat, content)
        if m:
            params[key] = m.group(1)
    return params


def extract_clk_period_from_sdc(sdc_path: Path) -> Optional[str]:
    """Extract clk_period from a constraint_N.sdc file."""
    if not sdc_path.is_file():
        return None
    try:
        content = sdc_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    m = re.search(r'set clk_period\s+([\d.]+)', content)
    return m.group(1) if m else None


def _extract_int_suffix(name: str, prefix: str) -> int:
    try:
        if prefix and name.startswith(prefix):
            parts = name.replace(prefix, "")
            return int(parts)
        match = re.search(r'run(\d+)', name)
        if match:
            return int(match.group(1))
    except ValueError:
        pass
    return -1


def discover_logs(base_dir: Path) -> Iterable[Tuple[int, int, Path]]:
    """
      base_dir/
        result_dump_0/
           logs_dump/
              *_run0.log
              *_run1.log
        result_dump_1/
           ...

    return: (Iteration ID, Task ID, Log Path)
    """

    result_dirs = sorted(base_dir.glob("result_dump_*"), key=lambda p: _extract_int_suffix(p.name, "result_dump_"))

    for r_dir in result_dirs:
        iter_id = _extract_int_suffix(r_dir.name, "result_dump_")
        if iter_id == -1: continue

        logs_dir = r_dir / "logs_dump"
        if not logs_dir.is_dir():
            continue

        log_files = sorted(logs_dir.glob("*_run*.log"), key=lambda p: _extract_int_suffix(p.stem, ""))

        for log_file in log_files:
            match = re.search(r"_run(\d+)\.log$", log_file.name)
            if match:
                task_id = int(match.group(1))
                yield iter_id, task_id, log_file


def _resolve_sdc_path(result_dump_dir: Path, task_id: int, platform: str, design: str) -> Path:
    """Find the SDC file for a given task in the backup directory."""
    if platform == "asap7" and design == "jpeg":
        return result_dump_dir / f"jpeg_encoder15_7nm_{task_id}.sdc"
    return result_dump_dir / f"constraint_{task_id}.sdc"


def _guess_platform_design(source_root: Path) -> Tuple[str, str]:
    """Infer platform and design from backup_dir/{platform}/{design} path."""
    design = source_root.name
    platform = source_root.parent.name if source_root.parent else "unknown"
    return platform, design


def build_markdown(rows: List[Dict], source_root: Path) -> str:
    timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Column definitions: (header, key, format_spec)
    # IMPORTANT: first 3 metric columns (ecp, dwl, cts_wl) must stay in this
    # order — count.py and print.py parse by positional index (parts[2..4]).
    metric_cols = [
        ("ecp", "ecp", ".4f"),
        ("dwl (um)", "dwl", ".4f"),
        ("cts_wl (um)", "cts_wl", ".4f"),
        ("drc", "drc", ".0f"),
        ("wns", "wns", ".4f"),
        ("tns", "tns", ".4f"),
        ("tns_eval", "tns_eval", ".4f"),
        ("CP_0", "cp0", ""),
    ]
    param_cols = [
        ("clk_period", "clk_period", ""),
        ("core_util", "core_util", ""),
        ("gp_pad", "cell_pad_global", ""),
        ("dp_pad", "cell_pad_detail", ""),
        ("dpo", "enable_dpo", ""),
    ]

    all_cols = metric_cols + param_cols
    headers = ["result_dump", "base"] + [c[0] for c in all_cols]
    sep = ["| :---:"] * len(headers)

    lines = [
        "# Log Metrics Summary",
        "",
        f"- Source root: `{source_root}`",
        f"- Generated at: {timestamp}",
        "",
        "| " + " | ".join(headers) + " |",
        " ".join(sep) + " |",
    ]

    rows.sort(key=lambda x: (x['iter_id'], x['task_id']))

    for row in rows:
        cells = [str(row['iter_id']), str(row['task_id'])]
        for header, key, fmt in all_cols:
            val = row.get(key)
            if val is None:
                cells.append("N/A")
            elif fmt:
                cells.append(f"{float(val):{fmt}}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")

    if not rows:
        cells = ["N/A"] * len(headers)
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def main() -> None:
    repo_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Extract metrics from nested result_dump_k/logs_dump directories."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Root directory (e.g., backup_dir/<platform>/<design>).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output markdown file path.",
    )
    args = parser.parse_args()

    source_root = args.input.resolve()
    platform, design = _guess_platform_design(source_root)

    if args.output:
        output_path = args.output if args.output.is_absolute() else Path.cwd() / args.output
    else:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = repo_root / "output_results" / f"{platform}_{design}_{ts}.md"

    # Read CP_0 (shared across all runs, written by eval_tns.sh)
    cp0_file = repo_root / "designs" / platform / design / "cp0.txt"
    cp0_value = None
    if cp0_file.is_file():
        try:
            cp0_value = cp0_file.read_text().strip()
        except Exception:
            pass

    print(f"Scanning for result_dump_* directories in: {source_root}")

    rows = []
    for iter_id, task_id, log_path in discover_logs(source_root):
        metrics = extract_metrics_from_log(log_path)

        # Locate config and SDC files in the backup directory
        result_dump_dir = log_path.parent.parent  # logs_dump -> result_dump_N
        config_path = result_dump_dir / f"config_{task_id}.mk"
        sdc_path = _resolve_sdc_path(result_dump_dir, task_id, platform, design)

        params = extract_params_from_config(config_path)
        clk_period = extract_clk_period_from_sdc(sdc_path)

        row = {
            "iter_id": iter_id,
            "task_id": task_id,
            **metrics,
            "clk_period": clk_period,
            "cp0": cp0_value,
            **params,
        }
        rows.append(row)

    print(f"Found {len(rows)} logs across all iterations. Writing report to: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_markdown(rows, source_root), encoding="utf-8")


if __name__ == "__main__":
    main()
