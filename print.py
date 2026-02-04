import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re
import sys
import os

def extract_table(file_content: str) -> pd.DataFrame:
    """
    Extract result_dump, ECP, total_wirelength
    """
    lines = file_content.splitlines()
    rows = []
    
    in_table = False
    header_found = False
    
    num_re = re.compile(r'([\d.]+)')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if '| result_dump |' in line and '| base |' in line:
            header_found = True
            continue
        
        if header_found and re.match(r'^\|[\s\-:|]+\|$', line):
            in_table = True
            continue

        if in_table and line.startswith('|'):
            parts = [p.strip() for p in line.split('|')]
            
            parts = [p for p in parts if p]

            
            if len(parts) >= 4:
                try:
                    
                    def parse_val(s):
                        if not s or "N/A" in s:
                            return None
                        m = num_re.search(s)
                        return float(m.group(1)) if m else None

                  
                    # | result_dump | base | ecp | dwl | cts_wl |
                    r_dump = int(parts[0])
                    # base = int(parts[1]) 
                    ecp = parse_val(parts[2])
                    dwl = parse_val(parts[3])

                    rows.append({
                        "result_dump": r_dump,
                        "ECP": ecp,
                        "total_wirelength": dwl
                    })
                except (ValueError, IndexError):
                    continue

    return pd.DataFrame(rows)


def prepare_data(df: pd.DataFrame, mode: str):
    """
    Calculate Envelope 
    """

    if mode == "DWL":
        df = df.dropna(subset=["total_wirelength"]).copy()
        df["metric"] = df["total_wirelength"]
        ylabel = "Total Wirelength (um)"
    elif mode == "ECP":
        df = df.dropna(subset=["ECP"]).copy()
        df["metric"] = df["ECP"]
        ylabel = "Effective Clock Period"
    else: # COMBO
        df = df.dropna(subset=["ECP", "total_wirelength"]).copy()
        df["metric"] = df["ECP"] + df["total_wirelength"]
        ylabel = "ECP + Total Wirelength"

    df_best = df.groupby("result_dump")["metric"].min().reset_index()
    
  
    df_best = df_best.sort_values("result_dump").reset_index(drop=True)
    df_best["envelope"] = df_best["metric"].cummin()
    
    return df_best, ylabel


def extract_envelope_breakpoints(df: pd.DataFrame) -> pd.DataFrame: 
    mask = df["envelope"].diff().fillna(0) < 0
    mask.iloc[0] = True
    return df[mask]


def plot_envelope(df_breakpoints: pd.DataFrame, ylabel: str, filename: str, mode: str, total_iters: int):
    plt.figure(figsize=(12, 6))

    x_vals = df_breakpoints["result_dump"].tolist()
    y_vals = df_breakpoints["envelope"].tolist()
    
    if x_vals[-1] < total_iters:
        x_vals.append(total_iters)
        y_vals.append(y_vals[-1])

    plt.step(x_vals, y_vals, where="post", linewidth=2, color="blue", label="Cumulative Best")

    plt.scatter(df_breakpoints["result_dump"], df_breakpoints["envelope"], color="red", label="New Best Found", zorder=5)

    seen_vals = set()
    for _, row in df_breakpoints.iterrows():
        val = row["envelope"]
  
        if val in seen_vals:
            continue
        seen_vals.add(val)
        
        label_text = f"{val:.2f}"
        plt.annotate(
            label_text,
            (row["result_dump"], val),
            textcoords="offset points",
            xytext=(0, -15), 
            ha="center",
            fontsize=9,
            color="darkred",
            fontweight='bold'
        )

    plt.xlabel("Iteration (Result Dump)")
    plt.ylabel(ylabel)
    plt.title(f"{mode} Optimization Envelope\nSource: {filename}")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()

 
    if total_iters <= 25:
        plt.xticks(range(1, total_iters + 1))
    
    plt.tight_layout()
    
   
    base_name = os.path.splitext(filename)[0] 
    out = f"{base_name}_{mode}_envelope.png"
    plt.savefig(out)
    print(f"Saved plot to {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot optimization envelope from log metrics.")
    parser.add_argument("filename", help="Path to markdown log file")
    parser.add_argument("-o", "--objective", required=True,
                        choices=["DWL", "ECP", "COMBO"],
                        help="Optimization objective")
    args = parser.parse_args()

    try:
        with open(args.filename, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{args.filename}' not found.")
        sys.exit(1)


    df_raw = extract_table(content)
    if df_raw.empty:
        print("No valid data found in table. Check file format.")
        return

 
    total_iters = int(df_raw["result_dump"].max())


    df_env, ylabel = prepare_data(df_raw, args.objective)
    
    if df_env.empty:
        print(f"No valid data for objective {args.objective} (all N/A?).")
        return

    
    df_bp = extract_envelope_breakpoints(df_env)
    plot_envelope(df_bp, ylabel, args.filename, args.objective, total_iters)


if __name__ == "__main__":
    main()