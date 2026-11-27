#!/usr/bin/env python3
"""Runs table2image_mv.py on every dataset folder of ONE MV level (resume with --skip_existing)."""
import argparse, json, os, re, subprocess, sys
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--datasets_dir', required=True)            # .../NEWtabular_byrate/mr10
ap.add_argument('--output_root', required=True)             # .../output_mv_baselines/table2image
ap.add_argument('--script_path', default=str(Path(__file__).with_name('table2image_mv.py')))
ap.add_argument('--image_data_root', default=None)
ap.add_argument('--timeout', type=int, default=14400)
ap.add_argument('--skip_existing', action='store_true')
ap.add_argument('--only', nargs='*', default=None)
a = ap.parse_args()

level = os.path.basename(os.path.normpath(a.datasets_dir))
assert re.match(r'^mr\d\d$', level), f"cannot read MV level from {a.datasets_dir}"
out = os.path.join(a.output_root, level)
os.makedirs(os.path.join(out, 'logs'), exist_ok=True)
folders = sorted(p for p in Path(a.datasets_dir).iterdir() if p.is_dir() and not p.name.startswith('.'))
status = {}
for i, folder in enumerate(folders, 1):
    if a.only and folder.name not in a.only:
        continue
    tag = f'{folder.name}_{level}'
    jpath = os.path.join(out, 'per_dataset', f'{tag}.json')
    if a.skip_existing and os.path.exists(jpath):
        print(f'[{i}/{len(folders)}] {tag}: skipped (already done)'); status[tag] = 'success'; continue
    files = sorted(f for pat in ('*.csv', '*.arff', '*.data') for f in folder.glob(pat))
    if not files:
        print(f'[{i}/{len(folders)}] {tag}: no data file'); status[tag] = 'failed'; continue
    cmd = [sys.executable, a.script_path, '--csv', str(files[0]), '--mv_level', level, '--output_root', out]
    if a.image_data_root:
        cmd += ['--image_data_root', a.image_data_root]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
        open(os.path.join(out, 'logs', f'{tag}.log'), 'w').write(r.stdout + '\n--- stderr ---\n' + r.stderr)
        if os.path.exists(jpath):
            status[tag] = 'success'
        elif 'Skipping' in r.stdout:
            status[tag] = 'skipped'
        else:
            status[tag] = 'failed'
    except subprocess.TimeoutExpired:
        status[tag] = 'timeout'
    print(f'[{i}/{len(folders)}] {tag}: {status[tag]}')

rows = [json.load(open(p)) for p in sorted(Path(out, 'per_dataset').glob('*.json'))]
if rows:
    pd.DataFrame(rows).to_csv(os.path.join(out, 'results_summary.csv'), index=False)
counts = {s: list(status.values()).count(s) for s in set(status.values())}
json.dump({'mv_level': level, 'counts': counts, 'not_done': {k: v for k, v in status.items() if v in ('failed', 'timeout')}},
          open(os.path.join(out, 'final_summary.json'), 'w'), indent=2)
print('DONE', counts)
