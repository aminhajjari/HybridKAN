
"""
Generate tabular datasets with randomly introduced missing values (ONCE, reused by all models).

  SRC (read only): /home/gkianfar/scratch/Amin/Tab2Vis/Unzippeddata/CSV
  DST            : /home/gkianfar/scratch/Amin/Sedo/NEWtabular

  NEWtabular/<dataset>/<dataset>_missing_<pct>.<original extension>     (pct = 0,10,20,...)
  NEWtabular_meta/{manifest.json, stats.csv, generation.log}

Definition of the rate
  eligible cells = N rows x D eligible feature columns
  newly injected = round(rate x N x D) cells, drawn uniformly from eligible cells that are
  currently observed (pre-existing missing cells are never counted as injected and never
  picked again). Pre-existing missingness is reported separately.
  Eligible = every column except the target and identifier-like columns (name looks like
  id/index/..., or integer column 0..N-1 sequence). Excluded columns are logged.

Reproducibility
  RNG = default_rng([seed, crc32(source relative path)]); ONE random matrix per file, so masks
  are nested (10% subset of 20% subset of ...). Rate 0 writes the unmodified rows (nothing masked).

Format preservation (text-level edit, nothing is re-serialised)
  .csv/.data/.dat/.txt : delimiter, header, column order, every untouched cell kept byte-for-byte
  .arff                : @relation/@attribute/@data lines and comments kept byte-for-byte
  marker: .csv -> empty cell, everything else -> '?'

Safety: never writes inside SRC; existing outputs are skipped unless --overwrite.

Commands
  python make_missing_datasets.py --inspect
  python make_missing_datasets.py --rates 0 0.1 0.2 0.3 0.4 0.5
  python make_missing_datasets.py --datasets adult credit-g --rates 0.1 0.3 --seed 7 --dst .../NEWtabular_seed7
  python make_missing_datasets.py --verify --rates 0 0.1 0.2 0.3 0.4 0.5
"""
import argparse
import csv
import hashlib
import json
import logging
import re
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

SRC_DEFAULT = "/home/gkianfar/scratch/Amin/Tab2Vis/Unzippeddata/CSV"
DST_DEFAULT = "/home/gkianfar/scratch/Amin/Sedo/NEWtabular"
RATES_DEFAULT = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SEED_DEFAULT = 42
EXTS_DEFAULT = ['.csv', '.arff', '.data', '.dat']        # add .txt with --exts if you have such data files

MISSING_MARKERS = ['?', '', ' ', 'nan', 'NaN', 'NA', 'null', 'None', '-']   # same as main.py
TARGET_CANDIDATES = [                                                      # same as main.py
    'target', 'class', 'outcome', 'Class', 'binaryClass', 'status', 'Target',
    'TR', 'speaker', 'Home/Away', 'Outcome', 'Leaving_Certificate', 'technology',
    'signal', 'label', 'Label', 'click', 'percent_pell_grant', 'Survival',
    'diagnosis', 'y', 'Author', 'Utility',
]
ID_NAME = re.compile(r'^(id|index|idx|row_?id|rowid|sample_?id|instance_?id|record_?id|unnamed.*|.*_id|id_.*)$', re.I)
ARFF_ATTR = re.compile(r"""^\s*@attribute\s+('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|\S+)""", re.I)
log = logging.getLogger("missing")


# ------------------------------------------------------------------ text helpers
def read_text(path):
    raw = path.read_bytes()
    for enc in ('utf-8', 'latin-1'):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue


def split_line(line, sep):
    """Split on sep outside quotes; keep raw token text (quotes included)."""
    toks, cur, q = [], [], None
    for ch in line:
        if q:
            cur.append(ch)
            if ch == q:
                q = None
        elif ch in ("'", '"') and not ''.join(cur).strip():
            q = ch
            cur.append(ch)
        elif ch == sep:
            toks.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
    toks.append(''.join(cur))
    return toks


def split_eol(line):
    body = line.rstrip('\r\n')
    return body, line[len(body):]


def clean_tok(t):
    return t.strip().strip("'\"")


def missing_matrix(rows):
    a = np.array(rows, dtype=object).astype(str)
    a = np.char.strip(np.char.strip(a), "'\"")
    return np.isin(a, MISSING_MARKERS)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


# ------------------------------------------------------------------ parsing
def parse_file(path, min_class_samples=0):
    text, enc = read_text(path)
    lines = text.splitlines(keepends=True)
    ext = path.suffix.lower()
    marker = '' if ext == '.csv' else '?'
    names = None

    if ext == '.arff':
        start = next((i + 1 for i, ln in enumerate(lines) if ln.strip().lower().startswith('@data')), None)
        if start is None:
            raise ValueError("no @data section")
        sep = ','
        names = [clean_tok(m.group(1)) for ln in lines[:start] if (m := ARFF_ATTR.match(ln))]
    elif ext == '.csv':
        first = next(i for i, ln in enumerate(lines) if ln.strip())
        sep = ','
        names = [clean_tok(t) for t in split_line(split_eol(lines[first])[0], sep)]
        start = first + 1
    else:  # .data / .dat / .txt : no header, delimiter detected like main.py
        first = next(i for i, ln in enumerate(lines) if ln.strip())
        sep = next((s for s in [',', ' ', '\t', ';']
                    if len(split_line(split_eol(lines[first])[0], s)) > 1), None)
        if sep is None:
            raise ValueError("cannot detect delimiter (not a table?)")
        start = first

    row_idx, rows = [], []
    for i in range(start, len(lines)):
        body, _ = split_eol(lines[i])
        s = body.strip()
        if not s or s.startswith('%'):
            continue
        if s.startswith('{'):
            raise ValueError("sparse ARFF rows not supported")
        row_idx.append(i)
        rows.append(split_line(body, sep))
    if not rows:
        raise ValueError("no data rows")
    n_cols = len(rows[0])
    if any(len(r) != n_cols for r in rows):
        raise ValueError("rows have different numbers of columns (embedded newlines / odd quoting?)")
    if names is not None and len(names) != n_cols:
        if ext == '.csv':
            raise ValueError("header / row column count mismatch")
        names = None   # arff names unreliable -> only used for id detection

    if ext == '.arff' or names is None:
        target_idx = n_cols - 1
    else:
        target_idx = next((j for j, n in enumerate(names) if n in TARGET_CANDIDATES), n_cols - 1)

    # optional: same rare-class row filter as main.py (off by default)
    if min_class_samples > 0:
        y = [clean_tok(r[target_idx]) for r in rows]
        cnt = Counter(y)
        if sum(1 for v in cnt.values() if v >= min_class_samples) < 2:
            raise ValueError("fewer than 2 classes with enough samples")
        keep = [cnt[v] >= min_class_samples for v in y]
        drop_lines = {ri for ri, k in zip(row_idx, keep) if not k}
        new_lines, remap = [], {}
        for i, ln in enumerate(lines):
            if i not in drop_lines:
                remap[i] = len(new_lines)
                new_lines.append(ln)
        row_idx = [remap[ri] for ri, k in zip(row_idx, keep) if k]
        rows = [r for r, k in zip(rows, keep) if k]
        lines = new_lines

    return dict(lines=lines, row_idx=row_idx, rows=rows, sep=sep, marker=marker, target_idx=target_idx,
                enc=enc, n_cols=n_cols, names=names, ext=ext)


def eligible_columns(info):
    """Feature columns that may be corrupted; returns (indices, {col: reason})."""
    t, m, rows = info['target_idx'], info['n_cols'], info['rows']
    n = len(rows)
    elig, excluded = [], {}
    for j in range(m):
        name = info['names'][j] if info['names'] else f"col{j}"
        if j == t:
            excluded[name] = 'target'
            continue
        if info['names'] and ID_NAME.match(name):
            excluded[name] = 'identifier-like name'
            continue
        try:
            v = np.array([clean_tok(r[j]) for r in rows], dtype=float)
            if np.all(v == np.round(v)) and len(np.unique(v)) == n and v.max() - v.min() == n - 1:
                excluded[name] = 'sequential integer id'
                continue
        except ValueError:
            pass
        elig.append(j)
    return elig, excluded


# ------------------------------------------------------------------ masking
def build_masks(info, elig, rates, rng):
    """Exact-count, nested masks over observed eligible cells. Returns ({rate: (mask, k)}, orig_missing)."""
    rows = info['rows']
    n, m, D = len(rows), info['n_cols'], len(elig)
    orig = missing_matrix(rows)
    u = rng.random((n, m))                       # ONE draw shared by all rates
    e = np.array(elig)
    flat = np.where(orig[:, e], np.inf, u[:, e]).ravel()
    order = np.argsort(flat, kind='stable')
    observed = int(np.isfinite(flat).sum())
    out = {}
    for r in rates:
        k = int(round(r * n * D))
        if k > observed:
            log.warning("rate %.2f needs %d cells but only %d observed eligible cells -> capped", r, k, observed)
            k = observed
        mask = np.zeros((n, m), dtype=bool)
        rr, cc = np.divmod(order[:k], D)
        mask[rr, e[cc]] = True
        out[r] = (mask, k)
    return out, orig


def render_masked(info, mask):
    lines = list(info['lines'])
    sep, marker = info['sep'], info['marker']
    for r, li in enumerate(info['row_idx']):
        if not mask[r].any():
            continue
        body, eol = split_eol(lines[li])
        toks = list(info['rows'][r])
        for j in np.where(mask[r])[0]:
            lead = toks[j][:len(toks[j]) - len(toks[j].lstrip())]
            toks[j] = lead + marker
        lines[li] = sep.join(toks) + eol
    return ''.join(lines).encode(info['enc'])


# ------------------------------------------------------------------ validation
def try_load(path, info):
    import pandas as pd
    ext = info['ext']
    if ext == '.arff':
        import scipy.io.arff as arff
        data, _ = arff.loadarff(str(path))
        return len(data), len(data.dtype.names)
    df = pd.read_csv(path, sep=info['sep'], header=0 if ext == '.csv' else None)
    return df.shape


def validate(info, mask, k, out_path, src_path):
    """Re-read the written file and check it against the source rows. Returns (ok, notes)."""
    notes = []
    out = parse_file(out_path)
    if len(out['rows']) != len(info['rows']) or out['n_cols'] != info['n_cols']:
        return False, ["row/column count changed"]
    A = np.array(info['rows'], dtype=object)
    B = np.array(out['rows'], dtype=object)
    diff = A != B
    if (diff & ~mask).any():
        notes.append("unmasked cells changed")
    if mask.any() and not (diff[mask].all() and all(clean_tok(x) == info['marker'] for x in B[mask])):
        notes.append("masked cells do not hold the missing marker")
    if mask[:, info['target_idx']].any():
        notes.append("target corrupted")
    if int(mask.sum()) != k:
        notes.append("masked count mismatch")
    rs = set(info['row_idx'])
    ro = set(out['row_idx'])
    if [l for i, l in enumerate(info['lines']) if i not in rs] != [l for i, l in enumerate(out['lines']) if i not in ro]:
        notes.append("header/metadata lines changed")
    try:
        shp = try_load(out_path, info)
        if tuple(shp) != (len(info['rows']), info['n_cols']):
            notes.append(f"loaded shape {tuple(shp)} != expected")
    except Exception as e:
        try:
            try_load(src_path, info)
            notes.append(f"load check failed: {e}")
        except Exception:   # the original is not loadable by this reader either -> not caused by us
            log.warning("  (note) %s: reader cannot load the ORIGINAL file either (%s)", out_path.name, e)
    return (not notes), notes


# ------------------------------------------------------------------ discovery
def discover(src, exts, only):
    folders = {}
    for f in sorted(src.rglob('*')):
        if f.is_file() and f.suffix.lower() in exts:
            rel = f.parent.relative_to(src)
            if only and not (set(rel.parts) & set(only)):
                continue
            folders.setdefault(f.parent, []).append(f)
    for folder, files in sorted(folders.items()):
        for f in files:
            base = folder.name if (len(files) == 1 and folder != src) else f.stem
            yield f, folder.relative_to(src), base


def parse_rates(vals):
    out = []
    for v in vals:
        v = v / 100 if v > 1 else v
        if not 0 <= v < 1:
            raise SystemExit(f"bad rate {v}")
        if round(v * 100) not in [round(x * 100) for x in out]:
            out.append(v)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default=SRC_DEFAULT)
    ap.add_argument('--dst', default=DST_DEFAULT)
    ap.add_argument('--meta_dir', default=None, help="default: <dst>_meta")
    ap.add_argument('--rates', type=float, nargs='+', default=RATES_DEFAULT, help="fractions (0.1) or percents (10)")
    ap.add_argument('--seed', type=int, default=SEED_DEFAULT)
    ap.add_argument('--datasets', nargs='*', default=None, help="only these dataset folders (default: all)")
    ap.add_argument('--exts', nargs='+', default=EXTS_DEFAULT)
    ap.add_argument('--min_class_samples', type=int, default=0,
                    help="drop rows of classes with fewer samples (main.py uses 10). Default 0 = keep all rows")
    ap.add_argument('--overwrite', action='store_true', help="regenerate existing files (default: skip them)")
    ap.add_argument('--inspect', action='store_true', help="show structure, write nothing")
    ap.add_argument('--verify', action='store_true', help="re-generate in memory and compare with files on disk")
    args = ap.parse_args()

    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    if not src.is_dir():
        raise SystemExit(f"SRC not found: {src}")
    if dst == src or src in dst.parents:
        raise SystemExit("DST must not be SRC or inside SRC (originals must stay untouched)")
    meta = Path(args.meta_dir).resolve() if args.meta_dir else dst.parent / (dst.name + '_meta')
    rates = parse_rates(args.rates)
    exts = {e.lower() if e.startswith('.') else '.' + e.lower() for e in args.exts}
    items = list(discover(src, exts, args.datasets))
    if not items:
        raise SystemExit("No matching data files found")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not (args.inspect or args.verify):
        meta.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(meta / 'generation.log')
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)

    # ---------------- inspect
    if args.inspect:
        log.info("Found %d data files under %s\n", len(items), src)
        cur = None
        for f, rel_dir, base in items:
            if rel_dir != cur:
                log.info("[%s]", rel_dir)
                cur = rel_dir
            try:
                info = parse_file(f, args.min_class_samples)
                elig, excl = eligible_columns(info)
                t = info['target_idx']
                tname = info['names'][t] if info['names'] else f"col[{t}]"
                y = [clean_tok(r[t]) for r in info['rows']]
                orig = missing_matrix(info['rows'])[:, elig]
                log.info("  %s: fmt=%s rows=%d cols=%d target='%s' classes=%d eligible_features=%d "
                         "existing_missing=%.2f%%", f.name, info['ext'], len(y), info['n_cols'], tname,
                         len(set(y)), len(elig), 100 * orig.mean())
                extra = {k: v for k, v in excl.items() if v != 'target'}
                if extra:
                    log.info("      excluded (not target): %s", extra)
                log.info("      -> %s", dst / rel_dir / f"{base}_missing_{{pct}}{f.suffix}")
            except Exception as e:
                log.info("  %s: CANNOT PARSE (%s)", f.name, e)
        return

    # ---------------- generate / verify
    mf = meta / 'manifest.json'
    manifest = json.loads(mf.read_text()) if mf.exists() else {}
    stats_keys = ['dataset', 'format', 'rate_requested', 'actual_new_rate', 'existing_missing_rate', 'samples',
                  'eligible_cols', 'eligible_cells', 'new_missing', 'seed', 'excluded_cols', 'validated', 'output']
    n_ok = n_skip = n_fail = n_bad = 0

    for f, rel_dir, base in items:
        key = str(rel_dir / f.name)
        ext = f.suffix
        outs = {r: dst / rel_dir / f"{base}_missing_{int(round(r * 100))}{ext}" for r in rates}
        todo = rates if args.verify else [r for r in rates if args.overwrite or not outs[r].exists()]
        if not todo:
            log.info("[SKIP] %s: all outputs exist", key)
            n_skip += 1
            continue
        try:
            info = parse_file(f, args.min_class_samples)
            elig, excl = eligible_columns(info)
            if not elig:
                raise ValueError("no eligible feature columns")
            rng = np.random.default_rng([args.seed, zlib.crc32(key.encode())])
            masks, orig = build_masks(info, elig, rates, rng)
        except Exception as e:
            log.error("[FAIL] %s: %s", key, e)
            n_fail += 1
            continue

        n, D = len(info['rows']), len(elig)
        src_hash = sha256(f)
        existing = float(orig[:, elig].mean())
        for r in todo:
            mask, k = masks[r]
            data = render_masked(info, mask)
            if args.verify:
                ok = outs[r].exists() and hashlib.sha256(data).hexdigest() == sha256(outs[r])
                prev = manifest.get(str(outs[r].relative_to(dst)), {}).get('source_sha256')
                if prev and prev != src_hash:
                    ok = False
                    log.error("[VERIFY] %s: ORIGINAL FILE CHANGED since generation", key)
                log.info("[VERIFY %s] %s", "OK" if ok else "MISMATCH", outs[r].relative_to(dst))
                n_ok += ok
                n_bad += (not ok)
                continue
            outs[r].parent.mkdir(parents=True, exist_ok=True)
            outs[r].write_bytes(data)
            ok, notes = validate(info, mask, k, outs[r], f)
            rec = dict(dataset=rel_dir.name or base, format=info['ext'], rate_requested=r,
                       actual_new_rate=round(k / (n * D), 6), existing_missing_rate=round(existing, 6),
                       samples=n, eligible_cols=D, eligible_cells=n * D, new_missing=k, seed=args.seed,
                       excluded_cols='; '.join(f"{a}({b})" for a, b in excl.items()),
                       validated=ok, output=str(outs[r]))
            manifest[str(outs[r].relative_to(dst))] = {**rec, 'source': str(f), 'source_sha256': src_hash,
                                                       'output_sha256': sha256(outs[r]), 'notes': notes}
            lvl = log.info if ok else log.error
            lvl("[%s] %s | fmt=%s rate=%.2f actual_new=%.4f existing=%.4f N=%d D=%d cells=%d new=%d seed=%d%s",
                "OK" if ok else "INVALID", outs[r].relative_to(dst), info['ext'], r, rec['actual_new_rate'],
                existing, n, D, n * D, k, args.seed, "" if ok else f" notes={notes}")
            n_ok += ok
            n_bad += (not ok)
    if not args.verify:
        mf.write_text(json.dumps(manifest, indent=2))
        with open(meta / 'stats.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=stats_keys, extrasaction='ignore')
            w.writeheader()
            w.writerows(manifest.values())
        log.info("\nDone. files ok=%d invalid=%d | datasets skipped=%d failed=%d", n_ok, n_bad, n_skip, n_fail)
        log.info("Output: %s\nMeta: %s (manifest.json, stats.csv, generation.log)", dst, meta)
    else:
        log.info("\nVerify: %d OK, %d mismatching, %d datasets failed to parse", n_ok, n_bad, n_fail)


if __name__ == '__main__':
    main()
