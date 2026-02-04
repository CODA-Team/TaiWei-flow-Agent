#!/usr/bin/env python3
"""
    Analyze the log_metrics Markdown file
    Find the optimal target value and its corresponding other column (supports DWL/ECP/COMBO)
    Count the occurrences of N/A
    Count the occurrences of N/A in each result_dump
"""

import re
import sys
import argparse
from collections import defaultdict


def parse_markdown_table(file_path):
    """
    Parse Markdown 
    
    return:
        records: record [{result_dump, base, ecp, dwl, cts_wl }, ...]
    """
    records = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # find content
    in_table = False
    header_found = False
    
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
            
            # (result_dump, base, ecp, dwl, cts_wl)
            if len(parts) >= 4: 
                try:
                    # parse result_dump and base
                    result_dump = int(parts[0])
                    base = int(parts[1])
                    
                    # N/A
                    def parse_metric(val_str):
                        if not val_str or 'N/A' in val_str:
                            return None
                        
                        match = re.search(r'([\d.]+)', val_str)
                        if match:
                            return float(match.group(1))
                        return None

                    # parse ECP (may be N/A)
                    ecp = parse_metric(parts[2])
                    
                    # parse DWL (may be N/A)
                    dwl = parse_metric(parts[3])
                    
                    # parse CTS WL 
                    cts_wl = parse_metric(parts[4]) if len(parts) > 4 else None
                    
                    
                    records.append({
                        'result_dump': result_dump,
                        'base': base,
                        'ecp': ecp,
                        'dwl': dwl,
                        'cts_wl': cts_wl
                    })

                except (ValueError, IndexError):
                
                    continue
    
    return records


def analyze_records(records, objective: str):
    """
    analyze records
    
    return:
        min_info: (refer to objective)
        na_count: N/A total numbers
        na_per_dump:  N/A of each result_dump
    """
    obj = objective.upper()

 
    def metric(rec):
        if obj == "DWL":
            return rec['dwl']
        if obj == "ECP":
            return rec['ecp']
        if obj == "COMBO":
            wl = rec['dwl']
            cp = rec['ecp']
            if wl is None or cp is None:
                return None
            return wl + cp
        return None

    valid_records = []
    for rec in records:
        m = metric(rec)
        if m is not None:
            rec = rec.copy()
            rec['metric'] = m
            valid_records.append(rec)
    
    min_info = None
    if valid_records:
        min_record = min(valid_records, key=lambda x: x['metric'])
        min_info = {
            'result_dump': min_record['result_dump'],
            'base': min_record['base'],
            'ecp': min_record['ecp'],
            'dwl': min_record['dwl'],
            'metric': min_record['metric'],
        }
    
    na_count = sum(1 for r in records if r['dwl'] is None or r['ecp'] is None)
    
   
    na_per_dump = defaultdict(int)
    for record in records:
        if record['dwl'] is None or record['ecp'] is None:
            na_per_dump[record['result_dump']] += 1
    
    return min_info, na_count, na_per_dump


def print_analysis(min_info, na_count, na_per_dump, records, objective: str):
    """
    print information
    """
    print("=" * 80)
    print("results of analyzing")
    print("=" * 80)
    
    # 最优目标
    print(f"\n【最优 {objective.upper()}】")
    if min_info:
        print(f"  Result Dump:      {min_info['result_dump']}")
        print(f"  Base:             {min_info['base']}")
        print(f"  ECP:              {min_info['ecp']}")
        print(f"  Total Wirelength: {min_info['dwl']}")
        print(f"  Metric({objective.upper()}): {min_info['metric']:.4f}")
    else:
        print("  No valid target data found")
    
    # N/A count
    print(f"\n[N/A count]")
    print(f"  Total number of records:         {len(records)}")
    print(f"  N/A total number:       {na_count}")
    percentage = (na_count/len(records)*100) if records else 0
    print(f"  N/A proportion:         {percentage:.2f}%")
    
    # N/A statistics for each result_dump
    print(f"\n[N/A statistics for each result_dump]")
    if na_per_dump:
        # Find the one with the most N/A occurrences.
        max_na_dump = max(na_per_dump.items(), key=lambda x: x[1])
        print(f" Result Dump with the most N/A occurrences: {max_na_dump[0]} (total {max_na_dump[1]} number of times)")
        
        # Show the top 10 N/A results.
        print(f"\n  Result dump of the top 10 N/A results.:")
        sorted_dumps = sorted(na_per_dump.items(), key=lambda x: x[1], reverse=True)[:10]
        for dump_id, count in sorted_dumps:
            print(f"    Result Dump {dump_id:3d}: {count:2d} numbers of N/A")
    else:
        print("  All records have valid metrics data.")
    
    # Fully valid result_dump
    print(f"\n[Fully valid result_dump]")
    all_dumps = set(r['result_dump'] for r in records)
    dumps_with_na = set(na_per_dump.keys())
    valid_dumps = all_dumps - dumps_with_na
    
    if valid_dumps:
        print(f"  Result dump numbers with no N/A : {len(valid_dumps)}")
        print(f"  Result dump IDs: {sorted(valid_dumps)}")
    else:
        print("  Result dump with no N/A")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='analyze log_metrics Markdown file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
example:
  python count.py output_results/asap7_aes_DWL_metrics.md -o DWL
  python count.py --detailed output_results/asap7_aes_DWL_metrics.md -o COMBO
        '''
    )
    parser.add_argument('file', help='Markdown path')
    parser.add_argument('--detailed', action='store_true', 
                       help='Show detailed N/A counts for each result_dump.')
    parser.add_argument('-o', '--objective', required=True,
                       choices=['DWL', 'ECP', 'COMBO'],
                       help='Select the following as the analysis targets: DWL (bus length), ECP (clock cycles), and COMBO (the sum of both).')
    
    args = parser.parse_args()
    
    try:
        print(f"reading: {args.file}")
        records = parse_markdown_table(args.file)
        
        if not records:
            print("Error: No valid data record found.")
            sys.exit(1)
        
        print(f"Successfully parsing {len(records)} records\n")
        
        min_info, na_count, na_per_dump = analyze_records(records, args.objective)
        
        print_analysis(min_info, na_count, na_per_dump, records, args.objective)
        
        if args.detailed and na_per_dump:
            print("\n[detailed N/A count (all Result Dump)]")
            for dump_id in sorted(na_per_dump.keys()):
                count = na_per_dump[dump_id]
                print(f"  Result Dump {dump_id:3d}: {count:2d} number of times N/A")
    
    except FileNotFoundError:
        print(f"error: file '{args.file}' ")
        sys.exit(1)
    except Exception as e:
        print(f"error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.append('-h')
    
    main()