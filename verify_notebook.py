import json
import os
import sys

# Ensure stdout uses UTF-8 if possible, or ignore encoding errors
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')

def verify_nb(filename):
    if not os.path.exists(filename):
        print(f"Error: {filename} does not exist!")
        return False
    try:
        with open(filename, 'r', encoding='utf-8') as fh:
            nb = json.load(fh)
        print(f"File: {filename}")
        print("  Valid JSON: True")
        print("  Cells:", len(nb['cells']))
        print("  File size:", round(os.path.getsize(filename) / 1024), "KB")
        print()
        for i, c in enumerate(nb['cells']):
            first = (c['source'][0] if c['source'] else '').strip()[:70]
            # Clean non-ASCII for safe console print
            clean_first = first.encode('ascii', 'ignore').decode('ascii')
            print(f"    [{i:02d}] {c['cell_type']:8s} | {clean_first}")
        print("-" * 65)
        return True
    except Exception as exc:
        print(f"Error loading {filename}: {exc}")
        return False

if __name__ == "__main__":
    verify_nb('detection.ipynb')
    verify_nb('Space_Debris_Complete.ipynb')
