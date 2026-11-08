
"""
Missing-value generator for SPARSE ARFF files (e.g. connect-4), companion of make_missing_datasets.py.

make_missing_datasets.py refuses sparse rows ("{idx value,idx value,...}"). This script handles them and
keeps the file sparse. Same conventions as the main generator, so connect-4 behaves like the other datasets:

  SRC (read only): /home/gkianfar/scratch/Amin/Tab2Vis/Unzippeddata/CSV/<dataset>/<file>.arff
  DST            : /home/gkianfar/scratch/Amin/Sedo/NEWtabular/<dataset>/<dataset>_missing_<pct>.arff
  META           : /home/gkianfar/scratch/Amin/Sedo/NEWtabular_meta/{manifest.json, stats.csv, generation.log}
                   (entries are merged into the existing files, nothing else in them is touched)

Definition of the rate (same as the main generator)
  The file is treated as the full N x (number of attributes) table. A cell that is not listed in a sparse row
  is an observed 0 (or the first nominal value), so it can be masked like any other cell.
  eligible cells = N rows x D eligible feature columns (all attributes except the last one = target, and
  identifier-like names). newly injected = round(rate x N x D) observed cells, drawn uniformly.
  Cells that are already '?' are never counted as injected and never picked again.

Reproducibility
  RNG = default_rng([seed, crc32("<dataset>/<file name>")]); one random matrix per file -> nested masks.
  Rate 0 writes the file byte-for-byte unchanged.

Format preservation
  Everything outside the touched rows is copied byte-for-byte (header, @attribute lines, comments, untouched
  rows). A touched row is re-emitted as "{idx value,idx value,...}" with its original values; a masked cell
  becomes the entry "idx ?". Rows stay sorted by index, as the sparse ARFF format requires.

Commands
  python make_missing_sparse.py --inspect
  python make_missing_sparse.py --rates 0 0.1 0.2 0.3 0.4 0.5
  python make_missing_sparse.py --verify --rates 0 0.1 0.2 0.3 0.4 0.5
"""
import argparse
import csv
import hashlib
import json
import logging
import re
import zlib
from pathlib import Path

import numpy as np

SRC_DEFAULT = "/home/gkianfar/scratch/Amin/Tab2Vis/Unzippeddata/CSV"
DST_DEFAULT = "/home/gkianfar/scratch/Amin/Sedo/NEWtabular"
DATASET_DEFAULT = "connect-4"
RATES_DEFAULT = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SEED_DEFAULT = 42

MISSING_MARKERS = ['?', '', ' ', 'nan', 'NaN', 'NA', 'null', 'None', '-']   # same as main.py
ID_NAME = re.compile(r'^(id|index|idx|row_?id|rowid|sample_?id|instance_?id|record_?id|unnamed.*|.*_id|id_.*)$', re.I)
ARFF_ATTR = re.compile(r"""^\s*@attribute\s+('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*"|\S+)""", re.I)
log = logging.getLogger("missing_sparse")


# ------------------------------------------------------------------ helpers
def read_text(path):
    raw = path.read_bytes()
    for enc in ('utf-8', 'latin-1'):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue


def split_eol(line):
    body = line.rstrip('\r\n')
    return body, line[len(body):]


def clean_tok(t):
    return t.strip().strip("'\"")


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def split_entries(inner):
    """Split the inside of '{...}' on commas outside quotes."""
    if "'" not in inner and '"' not in inner:
        return [t.strip() for t in inner.split(',') if t.strip()]
    toks, cur, q = [], [], None
    for ch in inner:
        if q:
            cur.append(ch)
            if ch == q:
                q = None
        elif ch in ("'", '"'):
            q = ch
            cur.append(ch)
        elif ch == ',':
            toks.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
    if q:
        raise ValueError("unbalanced quote in a sparse row")
    toks.append(''.join(cur))
    return [t.strip() for t in toks if t.strip()]


# ------------------------------------------------------------------ parsing
def parse_sparse(path):
    text, enc = read_text(path)
    lines = text.splitlines(keepends=True)
    start = next((i + 1 for i, ln in enumerate(lines) if ln.strip().lower().startswith('@data')), None)
    if start is None:
        raise ValueError("no @data section")
    names = [clean_tok(m.group(1)) for ln in lines[:start] if (m := ARFF_ATTR.match(ln))]
    m = len(names)
    if m < 2:
        raise ValueError("fewer than 2 @attribute lines")

    row_idx, rows = [], []
    for i in range(start, len(lines)):
        s = split_eol(lines[i])[0].strip()
        if not s or s.startswith('%'):
            continue
        if not (s.startswith('{') and s.endswith('}')):
            raise ValueError(f"line {i + 1} is not a plain sparse row (dense or weighted row?)")
        ent = {}
        for tok in split_entries(s[1:-1]):
            parts = tok.split(None, 1)
            if len(parts) != 2:
                raise ValueError(f"line {i + 1}: bad entry '{tok}'")
            j = int(parts[0])
            if not 0 <= j < m:
                raise ValueError(f"line {i + 1}: index {j} outside 0..{m - 1}")
            if j in ent:
                raise ValueError(f"line {i + 1}: index {j} appears twice")
            ent[j] = parts[1].strip()
        row_idx.append(i)
        rows.append(ent)
    if not rows:
        raise ValueError("no data rows")
    return dict(lines=lines, row_idx=row_idx, rows=rows, names=names, n_cols=m, target_idx=m - 1, enc=enc)


def eligible_columns(info):
    elig, excluded = [], {}
    for j, name in enumerate(info['names']):
        if j == info['target_idx']:
            excluded[name] = 'target'
        elif ID_NAME.match(name):
            excluded[name] = 'identifier-like name'
        else:
            elig.append(j)
    return elig, excluded


def missing_matrix(info):
    n, m = len(info['rows']), info['n_cols']
    out = np.zeros((n, m), dtype=bool)
    mk = set(MISSING_MARKERS)
    for r, ent in enumerate(info['rows']):
        for j, v in ent.items():
            if clean_tok(v) in mk:
                out[r, j] = True
    return out


# ------------------------------------------------------------------ masking (same scheme as make_missing_datasets.py)
def build_masks(info, elig, rates, rng):
    n, m, D = len(info['rows']), info['n_cols'], len(elig)
    orig = missing_matrix(info)
    u = rng.random((n, m))
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
    for r, li in enumerate(info['row_idx']):
        cols = np.flatnonzero(mask[r])
        if cols.size == 0:
            continue
        ent = dict(info['rows'][r])
        for j in cols:
            ent[int(j)] = '?'
        body, eol = split_eol(lines[li])
        lead = body[:len(body) - len(body.lstrip())]
        lines[li] = lead + '{' + ','.join(f"{j} {ent[j]}" for j in sorted(ent)) + '}' + eol
    return ''.join(lines).encode(info['enc'])


# ------------------------------------------------------------------ validation
def validate(info, mask, k, out_path, load_check=False):
    notes = []
    out = parse_sparse(out_path)
    if len(out['rows']) != len(info['rows']) or out['n_cols'] != info['n_cols']:
        return False, ["row/column count changed"]
    rs, ro = set(info['row_idx']), set(out['row_idx'])
    if [l for i, l in enumerate(info['lines']) if i not in rs] != [l for i, l in enumerate(out['lines']) if i not in ro]:
        notes.append("header/metadata lines changed")
    bad_unmasked = bad_masked = 0
    for r, (a, b) in enumerate(zip(info['rows'], out['rows'])):
        cols = np.flatnonzero(mask[r])
        if cols.size == 0:
            bad_unmasked += (a != b)
            continue
        cs = set(cols.tolist())
        bad_masked += sum(clean_tok(b.get(j, '')) != '?' for j in cs)
        bad_unmasked += ({j: v for j, v in a.items() if j not in cs} != {j: v for j, v in b.items() if j not in cs})
    if bad_unmasked:
        notes.append(f"{bad_unmasked} rows have changed unmasked cells")
    if bad_masked:
        notes.append(f"{bad_masked} masked cells do not hold '?'")
    if mask[:, info['target_idx']].any():
        notes.append("target corrupted")
    if int(mask.sum()) != k:
        notes.append("masked count mismatch")
    if load_check:
        try:
            import arff as arff_lib
            with open(out_path, 'r') as fh:
                data = arff_lib.load(fh)['data']
            if len(data) != len(info['rows']) or len(data[0]) != info['n_cols']:
                notes.append(f"liac-arff loaded shape ({len(data)}, {len(data[0])}) != expected")
        except Exception as e:
            notes.append(f"liac-arff load failed: {e}")
    return (not notes), notes


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
    ap.add_argument('--dataset', default=DATASET_DEFAULT, help="dataset folder under --src (default: connect-4)")
    ap.add_argument('--rates', type=float, nargs='+', default=RATES_DEFAULT, help="fractions (0.1) or percents (10)")
    ap.add_argument('--seed', type=int, default=SEED_DEFAULT)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--inspect', action='store_true', help="show structure, write nothing")
    ap.add_argument('--verify', action='store_true', help="re-generate in memory and compare with files on disk")
    ap.add_argument('--load_check', action='store_true', help="also load each output with liac-arff (slow)")
    args = ap.parse_args()

    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    if not src.is_dir():
        raise SystemExit(f"SRC not found: {src}")
    if dst == src or src in dst.parents:
        raise SystemExit("DST must not be SRC or inside SRC (originals must stay untouched)")
    meta = Path(args.meta_dir).resolve() if args.meta_dir else dst.parent / (dst.name + '_meta')
    rates = parse_rates(args.rates)
    folder = src / args.dataset
    files = sorted(p for p in folder.glob('*') if p.is_file() and p.suffix.lower() == '.arff')
    if not files:
        raise SystemExit(f"No .arff file in {folder}")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not (args.inspect or args.verify):
        meta.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(meta / 'generation.log')
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)

    mf = meta / 'manifest.json'
    manifest = json.loads(mf.read_text()) if mf.exists() else {}
    stats_keys = ['dataset', 'format', 'rate_requested', 'actual_new_rate', 'existing_missing_rate', 'samples',
                  'eligible_cols', 'eligible_cells', 'new_missing', 'seed', 'excluded_cols', 'validated', 'output']
    n_ok = n_bad = 0

    for f in files:
        rel_dir = f.parent.relative_to(src)
        base = folder.name if len(files) == 1 else f.stem
        key = str(rel_dir / f.name)
        outs = {r: dst / rel_dir / f"{base}_missing_{int(round(r * 100))}{f.suffix}" for r in rates}
        todo = rates if (args.verify or args.inspect) else [r for r in rates if args.overwrite or not outs[r].exists()]
        if not todo:
            log.info("[SKIP] %s: all outputs exist", key)
            continue
        try:
            info = parse_sparse(f)
            elig, excl = eligible_columns(info)
            if not elig:
                raise ValueError("no eligible feature columns")
            orig_missing = missing_matrix(info)
        except Exception as e:
            log.error("[FAIL] %s: %s", key, e)
            continue
        n, D = len(info['rows']), len(elig)
        nnz = sum(len(r) for r in info['rows'])

        if args.inspect:
            log.info("%s: sparse ARFF rows=%d attributes=%d target='%s' eligible_features=%d stored_entries=%d "
                     "(density %.1f%%) existing_missing=%.2f%%", key, n, info['n_cols'],
                     info['names'][info['target_idx']], D, nnz, 100 * nnz / (n * info['n_cols']),
                     100 * orig_missing[:, elig].mean())
            extra = {a: b for a, b in excl.items() if b != 'target'}
            if extra:
                log.info("   excluded (not target): %s", extra)
            continue

        rng = np.random.default_rng([args.seed, zlib.crc32(key.encode())])
        masks, orig = build_masks(info, elig, rates, rng)
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
            ok, notes = validate(info, mask, k, outs[r], args.load_check)
            rec = dict(dataset=rel_dir.name or base, format='.arff', rate_requested=r,
                       actual_new_rate=round(k / (n * D), 6), existing_missing_rate=round(existing, 6),
                       samples=n, eligible_cols=D, eligible_cells=n * D, new_missing=k, seed=args.seed,
                       excluded_cols='; '.join(f"{a}({b})" for a, b in excl.items()),
                       validated=ok, output=str(outs[r]))
            manifest[str(outs[r].relative_to(dst))] = {**rec, 'source': str(f), 'source_sha256': src_hash,
                                                       'output_sha256': sha256(outs[r]), 'notes': notes,
                                                       'sparse': True}
            lvl = log.info if ok else log.error
            lvl("[%s] %s | fmt=.arff(sparse) rate=%.2f actual_new=%.4f existing=%.4f N=%d D=%d cells=%d new=%d seed=%d%s",
                "OK" if ok else "INVALID", outs[r].relative_to(dst), r, rec['actual_new_rate'], existing,
                n, D, n * D, k, args.seed, "" if ok else f" notes={notes}")
            n_ok += ok
            n_bad += (not ok)

    if args.inspect:
        return
    if args.verify:
        log.info("\nVerify: %d OK, %d mismatching", n_ok, n_bad)
        return
    mf.write_text(json.dumps(manifest, indent=2))
    with open(meta / 'stats.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=stats_keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(manifest.values())
    log.info("\nDone. files ok=%d invalid=%d", n_ok, n_bad)
    log.info("Output: %s\nMeta: %s (manifest.json, stats.csv, generation.log)", dst, meta)


if __name__ == '__main__':
    main()
