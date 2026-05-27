#!/usr/bin/env python3
"""
count_objects.py
================
Count exact number of valid Active Satellite and Debris objects from
OLD_dataset (2024) and 26_Data set (2026).
"""
import os
import numpy as np
from datetime import datetime, timezone
from sgp4.api import Satrec, jday

# Folder paths
DIR_2024 = "OLD_dataset"
DIR_2026 = "26_Data set"

# File maps
FILES_2024 = {
    'cosmos_1408'      : 'cosmos-1408-debris.txt',
    'cosmos_2251'      : 'cosmos-2251-debris.txt',
    'fengyun_1c'       : 'fengyun-1c-debris.txt',
    'iridium_33'       : 'iridium-33-debris.txt',
    'active_satellites': 'active-sate.txt',
}

FILES_2026 = {
    'cosmos_1408'      : 'cosmos_1408_2026.txt',
    'cosmos_2251'      : 'cosmos_2251_2026.txt',
    'fengyun_1c'       : 'fengyun_1c_2026.txt',
    'iridium_33'       : 'iridium_33_2026.txt',
    'active_satellites': 'active_satellites_2026.txt',
}

GROUPS = {
    'cosmos_1408'      : ('DEBRIS',     1),
    'cosmos_2251'      : ('DEBRIS',     1),
    'fengyun_1c'       : ('DEBRIS',     1),
    'iridium_33'       : ('DEBRIS',     1),
    'active_satellites': ('ACTIVE SAT', 0),
}

def count_valid(filepath, eval_time=None):
    if not os.path.exists(filepath):
        return 0, 0, 0
    if eval_time is None:
        eval_time = datetime.now(timezone.utc)
    jd_now, fr_now = jday(eval_time.year, eval_time.month, eval_time.day,
                          eval_time.hour, eval_time.minute, eval_time.second)
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as fh:
        lines = [l.rstrip() for l in fh if l.strip()]
    total_in_file = 0
    valid = 0
    skipped = 0
    for i in range(0, len(lines) - 2, 3):
        line1 = lines[i+1].strip()
        line2 = lines[i+2].strip()
        if not (line1.startswith('1 ') and line2.startswith('2 ')):
            continue
        total_in_file += 1
        try:
            sat = Satrec.twoline2rv(line1, line2)
            e, r, v = sat.sgp4(jd_now, fr_now)
            if e == 0:
                valid += 1
            else:
                skipped += 1
        except Exception:
            skipped += 1
    return total_in_file, valid, skipped

def print_dataset_report(year, directory, file_map, eval_time):
    print("=" * 65)
    print(f"   EXACT OBJECT COUNT FROM {year} DATASET ({directory})")
    print("=" * 65)
    print(f"  {'Group':<25} {'In File':>8} {'Valid':>8} {'Skipped':>8}  Type")
    print("  " + "-" * 60)
    
    total_debris = 0
    total_active = 0
    total_all    = 0

    for grp, (tag, lbl) in GROUPS.items():
        filepath = os.path.join(directory, file_map[grp])
        tin, valid, skip = count_valid(filepath, eval_time)
        print(f"  {grp:<25} {tin:>8} {valid:>8} {skip:>8}  [{tag}]")
        if lbl == 1:
            total_debris += valid
        else:
            total_active += valid
        total_all += valid

    print("  " + "-" * 60)
    print(f"  {'GRAND TOTAL':<25} {'':>8} {total_all:>8}")
    print()
    print(f"   Total DEBRIS objects     (label = 1)  :  {total_debris}")
    print(f"   Total ACTIVE SAT objects (label = 0)  :  {total_active}")
    print(f"   Grand Total                           :  {total_all}")
    print("=" * 65)
    if total_all > 0:
        print(f"   Debris share  :  {total_debris/total_all*100:.2f}%")
        print(f"   Active share  :  {total_active/total_all*100:.2f}%")
    print()

if __name__ == "__main__":
    # Eval time for 2024 dataset
    eval_2024 = datetime(2024, 11, 17, 18, 0, 0, tzinfo=timezone.utc)
    # Eval time for 2026 dataset
    eval_2026 = datetime(2026, 5, 28, 0, 0, 0, tzinfo=timezone.utc)
    
    print_dataset_report("2024", DIR_2024, FILES_2024, eval_2024)
    print_dataset_report("2026", DIR_2026, FILES_2026, eval_2026)
