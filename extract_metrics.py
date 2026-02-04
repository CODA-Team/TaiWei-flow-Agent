#!/usr/bin/env python3
"""
Extract metrics (ECP, Detailed Wirelength, CTS Wirelength) from nested logs:
Structure: backup_dir/platform/design/result_dump_k/logs_dump/*_runi.log
"""

from __future__ import annotations

import argparse
import datetime as _dt
import re
from pathlib import Path
from typing import Iterable, Tuple, Dict, Optional, List



ECP_PATTERN = re.compile(
    r'Report metrics stage 6, finish[\s\S]*?(?:core_)?cl[ock]*\s+period_min\s*=\s*([-\d.]+)'
)

DWL_PATTERN = re.compile(
    r'\[INFO DRT-0198\] Complete detail routing\..*?Total wire length\s*=\s*([\d.]+)', 
    re.DOTALL
)

CTS_WL_PATTERN = re.compile(r'Total wirelength:\s*([\d.]+)')


def extract_metrics_from_log(log_path: Path) -> Dict[str, Optional[float]]:

    metrics = {
        "ecp": None,
        "dwl": None,
        "cts_wl": None
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

    return metrics


def _extract_int_suffix(name: str, prefix: str) -> int:
    """辅助函数：从 result_dump_10 或 run5 中提取数字"""
    try:
       
        if prefix and name.startswith(prefix):
            parts = name.replace(prefix, "")
            return int(parts)
        #  _run{i}.log 
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


def build_markdown(rows: List[Dict], source_root: Path) -> str:
    timestamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Log Metrics Summary",
        "",
        f"- Source root: `{source_root}`",
        f"- Generated at: {timestamp}",
        "",
        "| result_dump | base | ecp | dwl (um) | cts_wl (um) |",
        "| :---: | :---: | :---: | :---: | :---: |",
    ]
    

    rows.sort(key=lambda x: (x['iter_id'], x['task_id']))

    for row in rows:
        ecp_str = f"{row['ecp']:.4f}" if row['ecp'] is not None else "N/A"
        dwl_str = f"{row['dwl']:.4f}" if row['dwl'] is not None else "N/A"
        cts_str = f"{row['cts_wl']:.4f}" if row['cts_wl'] is not None else "N/A"
        
        lines.append(
            f"| {row['iter_id']} | {row['task_id']} | {ecp_str} | {dwl_str} | {cts_str} |"
        )
        
    if not rows:
        lines.append("| - | - | - | - | - |")
        
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

    if args.output:
        output_path = args.output if args.output.is_absolute() else Path.cwd() / args.output
    else:
        design = source_root.name
        platform = source_root.parent.name if source_root.parent else "unknown"
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = repo_root / "output_results" / f"{platform}_{design}_{ts}.md"

    print(f"Scanning for result_dump_* directories in: {source_root}")
    
    rows = []
    # discover_logs return (iter_id, task_id, path)
    for iter_id, task_id, log_path in discover_logs(source_root):
        metrics = extract_metrics_from_log(log_path)
        rows.append({
            "iter_id": iter_id,
            "task_id": task_id,
            "ecp": metrics["ecp"],
            "dwl": metrics["dwl"],
            "cts_wl": metrics["cts_wl"]
        })

    print(f"Found {len(rows)} logs across all iterations. Writing report to: {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_markdown(rows, source_root), encoding="utf-8")


if __name__ == "__main__":
    main()