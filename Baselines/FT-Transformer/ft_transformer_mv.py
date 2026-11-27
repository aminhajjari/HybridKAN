#!/usr/bin/env python3
"""
ft_transformer_mv.py  -  ft_transformer baseline on ONE missing-value (MV) level.

The data handling is a copy of main_mv.py (your HybridKAN script), so every baseline sees EXACTLY the same
feature matrix and the same train/test split as your model:
  * same loader (scipy ARFF -> liac-arff fallback for sparse ARFF, csv, .data)
  * same missing markers, target detection, rare-class (<10) filter, >20-class skip
  * NO ">50% missing -> drop column" rule (removed, as in main_mv.py)
  * categorical -> LabelEncoder, median imputation, StandardScaler, then
    train_test_split(test_size=0.2, random_state=42, stratify=y)       (same order as main_mv.py)
  * metrics as in main_mv.py: accuracy in %, AUC = column 1 for binary, one-vs-rest macro for multi-class

Option  --missing native : XGBoost / LightGBM receive the NaNs and handle them natively (no imputation);
                           FT-Transformer is always imputed.  Default: impute (identical to your model).

Output (per model and level, nothing is shared between levels or models):
  <output_root>/<model>/<mrXX>/per_dataset/<dataset>_<mrXX>.json      one file per run (status success/skipped/failed/timeout)
  <output_root>/<model>/<mrXX>/results_summary.csv                    dataset,num_samples,num_features,num_classes,best_accuracy,best_auc,...
  <output_root>/<model>/<mrXX>/final_summary.json
Resume: runs whose JSON says success/skipped are not repeated (failed / timeout are retried).

Example
  python ft_transformer_mv.py --datasets_dir .../NEWtabular_byrate/mr10 --output_root .../output_mv_baselines \
         
"""
import argparse
import json
import os
import re
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io.arff as arff
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

SEED = 42                          # model seed (the split always uses random_state=42, as in main_mv.py)
ALL_MODELS = ['ft_transformer']
EXTS = ('.csv', '.arff', '.data')


class SkipDataset(Exception):
    pass


# =============================================================================================
# DATA  (copied from main_mv.py; exit() calls turned into SkipDataset / exceptions)
# =============================================================================================
def load_dataset(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext == '.csv':
        return pd.read_csv(file_path), None
    elif file_ext == '.arff':
        try:
            data, meta = arff.loadarff(file_path)
            df = pd.DataFrame(data)
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = df[col].str.decode('utf-8')
                    except AttributeError:
                        pass
            return df, meta
        except Exception as e:
            try:
                import arff as arff_lib            # sparse ARFF: scipy fails, liac-arff reads it
                with open(file_path, 'r') as f:
                    dataset = arff_lib.load(f)
                df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])
                return df, None
            except Exception as e2:
                raise Exception(f"All ARFF parsers failed. Errors: (1) {e}, (2) {e2}")
    elif file_ext == '.data':
        for sep in [',', ' ', '\t', ';']:
            try:
                df = pd.read_csv(file_path, sep=sep, header=None)
                if df.shape[1] > 1:
                    return df, None
            except Exception:
                continue
        raise Exception("Could not determine delimiter for .data file")
    raise ValueError(f"Unsupported file format: {file_ext}")


def prepare_data(data_path, missing_mode='impute'):
    """Returns X_train, X_test, y_train, y_test, info.  Same steps and order as main_mv.py."""
    df, arff_meta = load_dataset(data_path)
    if df.empty:
        raise ValueError("Dataset is empty after loading")
    if df.shape[1] < 2:
        raise ValueError(f"Dataset has only {df.shape[1]} column(s), need at least 2")

    missing_markers = ['?', '', ' ', 'nan', 'NaN', 'NA', 'null', 'None', '-']
    df = df.replace(missing_markers, np.nan)

    target_col = None
    if arff_meta is not None:
        try:
            target_col = list(arff_meta.names())[-1]
            if target_col not in df.columns:
                target_col = None
        except Exception:
            target_col = None
    if target_col is None:
        cands = ['target', 'class', 'outcome', 'Class', 'binaryClass', 'status', 'Target', 'TR', 'speaker',
                 'Home/Away', 'Outcome', 'Leaving_Certificate', 'technology', 'signal', 'label', 'Label', 'click',
                 'percent_pell_grant', 'Survival', 'diagnosis', 'y', 'Author', 'Utility']
        target_col = next((c for c in df.columns if c in cands), None)
    if target_col is None:
        target_col = df.columns[-1]

    vc = df[target_col].value_counts()
    rare = vc[vc < 10]
    if len(rare) > 0:
        valid = vc[vc >= 10].index.tolist()
        if len(valid) < 2:
            raise SkipDataset("insufficient samples per class")
        df = df[df[target_col].isin(valid)]

    # (the ">50% missing -> drop column" rule is intentionally NOT applied)
    if df.shape[1] <= 1:
        raise ValueError("All feature columns were dropped. Dataset unusable.")

    if df[target_col].dtype == 'object' or not pd.api.types.is_numeric_dtype(df[target_col]):
        y = LabelEncoder().fit_transform(df[target_col].astype(str))
    else:
        y = df[target_col].astype(int).values
    uniq = sorted(set(y))
    num_classes = len(uniq)
    if num_classes > 20:
        raise SkipDataset(f"{num_classes} classes (>20)")
    if num_classes < 2:
        raise ValueError(f"Dataset has only {num_classes} class. Need at least 2.")

    X_df = df.drop(columns=[target_col])
    for col in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))
        else:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')

    n_missing = int(X_df.isnull().sum().sum())
    if missing_mode == 'impute':
        X = SimpleImputer(strategy='median').fit_transform(X_df)
    else:                                              # native: keep NaN (StandardScaler ignores NaN)
        X = X_df.to_numpy(dtype=float)
    vm = {u: i for i, u in enumerate(uniq)}
    y = np.array([vm[v] for v in y])

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    info = dict(num_samples=int(X.shape[0]), num_features=int(X.shape[1]), num_classes=int(num_classes),
                num_train=int(len(X_train)), num_test=int(len(X_test)), target_col=str(target_col),
                missing_cells_in_features=n_missing, missing_mode=missing_mode)
    return X_train, X_test, y_train, y_test, info


# =============================================================================================
# METRICS  (same definitions as main_mv.py)
# =============================================================================================
def metrics(y_true, proba):
    acc = float((proba.argmax(1) == y_true).mean() * 100.0)
    try:
        if proba.shape[1] == 2:
            auc = float(roc_auc_score(y_true, proba[:, 1]))
        else:
            auc = float(roc_auc_score(y_true, proba, multi_class='ovr', average='macro'))
    except Exception:
        auc = float('nan')
    return acc, auc


# =============================================================================================
# MODELS
# =============================================================================================
def fit_predict_ft_transformer(Xtr, ytr, Xte, nc, a):
    """FT-Transformer (Gorishniy et al. 2021), own compact PyTorch implementation, rtdl default sizes."""
    import copy
    import torch
    from torch import nn

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nf = Xtr.shape[1]
    d_tok, n_blocks, heads = 192, 3, 8

    class FT(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.empty(nf, d_tok))          # feature tokenizer: token_i = x_i * w_i + b_i
            self.b = nn.Parameter(torch.empty(nf, d_tok))
            bound = 1 / np.sqrt(d_tok)
            nn.init.uniform_(self.w, -bound, bound)
            nn.init.uniform_(self.b, -bound, bound)
            self.cls = nn.Parameter(torch.empty(1, 1, d_tok))
            nn.init.uniform_(self.cls, -bound, bound)
            layer = nn.TransformerEncoderLayer(d_model=d_tok, nhead=heads, dim_feedforward=512, dropout=0.1,
                                               activation='gelu', batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, num_layers=n_blocks, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(d_tok), nn.ReLU(), nn.Linear(d_tok, nc))

        def forward(self, x):
            t = x.unsqueeze(-1) * self.w + self.b                  # (B, nf, d)
            t = torch.cat([self.cls.expand(len(x), -1, -1), t], 1)
            return self.head(self.enc(t)[:, 0])

    # hold out 10% of the TRAIN rows for early stopping (the test set is never used for model selection)
    try:
        tr_i, va_i = train_test_split(np.arange(len(Xtr)), test_size=0.1, random_state=SEED, stratify=ytr)
    except ValueError:
        tr_i, va_i = train_test_split(np.arange(len(Xtr)), test_size=0.1, random_state=SEED)
    Xt = torch.tensor(Xtr[tr_i], dtype=torch.float32)
    yt = torch.tensor(ytr[tr_i], dtype=torch.long)
    Xv = torch.tensor(Xtr[va_i], dtype=torch.float32).to(dev)
    yv = torch.tensor(ytr[va_i], dtype=torch.long).to(dev)
    Xe = torch.tensor(Xte, dtype=torch.float32)

    bs = 256 if nf <= 100 else max(8, int(256 * (100 / nf) ** 2))    # attention is O(nf^2) in memory
    model = FT().to(dev)
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if (n in ('b', 'cls') or p.ndim == 1) else decay).append(p)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 1e-5}, {'params': no_decay, 'weight_decay': 0.0}], lr=1e-4)
    lossf = nn.CrossEntropyLoss()

    def predict(X):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), max(bs, 512)):
                out.append(torch.softmax(model(X[i:i + max(bs, 512)].to(dev)), 1).cpu())
        return torch.cat(out).numpy()

    best, best_state, bad, ep_best = float('inf'), None, 0, 0
    for ep in range(1, a.ft_epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[idx].to(dev)), yt[idx].to(dev))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(lossf(model(Xv), yv))
        if vl < best - 1e-5:
            best, best_state, bad, ep_best = vl, copy.deepcopy(model.state_dict()), 0, ep
        else:
            bad += 1
            if bad >= a.ft_patience:
                break
    model.load_state_dict(best_state)
    return predict(Xe), dict(params=f'FT-Transformer d={d_tok} blocks={n_blocks} heads={heads} lr=1e-4 wd=1e-5 '
                                    f'batch={bs} early-stop patience={a.ft_patience} on 10% of train',
                             best_epoch=ep_best, n_parameters=int(sum(p.numel() for p in model.parameters())))


RUNNERS = {'ft_transformer': fit_predict_ft_transformer}



# =============================================================================================
# DRIVER
# =============================================================================================
class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def find_datasets(level_dir):
    out = []
    for folder in sorted(Path(level_dir).iterdir()):
        if folder.name.startswith('.') or not folder.is_dir():
            continue
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS and not p.name.startswith('.'))
        out.append((folder.name, files[0] if files else None))
    return out


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    os.replace(tmp, path)


def summarize(model_dir, level, model, expected_names):
    rows, status = [], {}
    pdir = Path(model_dir) / 'per_dataset'
    for n in expected_names:
        p = pdir / f'{n}_{level}.json'
        st = json.load(open(p)) if p.exists() else None
        status[n] = (st or {}).get('status', 'pending')
        if st and st.get('status') == 'success':
            rows.append({k: st.get(k) for k in ('run_tag', 'num_samples', 'num_features', 'num_classes',
                                               'best_accuracy', 'best_auc', 'fit_seconds', 'num_train', 'num_test',
                                               'missing_cells_in_features', 'missing_mode')} | {'dataset': st['run_tag'],
                                                                                             'model': model, 'mv_level': level})
    df = pd.DataFrame(rows)
    if len(df):
        cols = ['dataset', 'model', 'mv_level', 'num_samples', 'num_features', 'num_classes', 'best_accuracy', 'best_auc',
                'fit_seconds', 'num_train', 'num_test', 'missing_cells_in_features', 'missing_mode']
        df[cols].to_csv(Path(model_dir) / 'results_summary.csv', index=False)
    counts = {}
    for s in status.values():
        counts[s] = counts.get(s, 0) + 1
    summ = dict(model=model, mv_level=level, datasets=len(expected_names), counts=counts,
                mean_accuracy=float(df.best_accuracy.mean()) if len(df) else None,
                mean_auc=float(df.best_auc.mean()) if len(df) else None,
                failed={n: s for n, s in status.items() if s in ('failed', 'timeout', 'pending')},
                COMPLETE=all(s in ('success', 'skipped') for s in status.values()),
                finished=datetime.now().isoformat())
    write_json(str(Path(model_dir) / 'final_summary.json'), summ)
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets_dir', required=True, help='ONE MV level, e.g. .../NEWtabular_byrate/mr10')
    ap.add_argument('--output_root', required=True)
    ap.add_argument('--models', nargs='+', default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument('--mv_level', default=None, help='default: name of --datasets_dir')
    ap.add_argument('--missing', choices=['impute', 'native'], default='impute')
    ap.add_argument('--timeout', type=int, default=14400, help='seconds per (dataset, model); original script: 4 h')
    ap.add_argument('--skip_existing', action='store_true', help='resume: skip runs already success/skipped')
    ap.add_argument('--expected_datasets', type=int, default=67, help='0 = no check')
    ap.add_argument('--only', nargs='*', default=None, help='restrict to these dataset folders (debugging)')
    ap.add_argument('--n_jobs', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count() or 4)))
    ap.add_argument('--ft_epochs', type=int, default=100)
    ap.add_argument('--ft_patience', type=int, default=16)
    a = ap.parse_args()

    level = a.mv_level or os.path.basename(os.path.normpath(a.datasets_dir))
    if not re.match(r'^mr\d\d$', level):
        sys.exit(f"cannot determine MV level from '{a.datasets_dir}', use --mv_level mr10")
    datasets = find_datasets(a.datasets_dir)
    if a.expected_datasets and len(datasets) != a.expected_datasets:
        sys.exit(f"expected {a.expected_datasets} dataset folders in {a.datasets_dir}, found {len(datasets)} "
                 f"(--expected_datasets 0 to override)")
    names_all = [n for n, _ in datasets]
    todo = [(n, f) for n, f in datasets if not a.only or n in set(a.only)]

    print('=' * 70)
    print(f"BASELINES  level={level}  models={a.models}  datasets={len(todo)}  missing={a.missing}  n_jobs={a.n_jobs}")
    print('=' * 70)

    signal.signal(signal.SIGALRM, _alarm)
    for model in a.models:
        mdir = os.path.join(a.output_root, model, level)
        os.makedirs(os.path.join(mdir, 'per_dataset'), exist_ok=True)
        for i, (name, fpath) in enumerate(todo, 1):
            tag = f'{name}_{level}'
            jpath = os.path.join(mdir, 'per_dataset', f'{tag}.json')
            if a.skip_existing and os.path.exists(jpath):
                try:
                    if json.load(open(jpath)).get('status') in ('success', 'skipped'):
                        print(f'[{model}] {i}/{len(todo)} {tag}: skipped (already done)')
                        continue
                except Exception:
                    pass
            rec = dict(run_tag=tag, dataset=name, mv_level=level, model=model, data_file=str(fpath),
                       started=datetime.now().isoformat())
            t0 = time.time()
            try:
                if fpath is None:
                    raise FileNotFoundError('no data file in folder')
                Xtr, Xte, ytr, yte, info = prepare_data(str(fpath), a.missing if model != 'ft_transformer' else 'impute')
                rec.update(info)
                if model == 'ft_transformer':
                    rec['missing_mode'] = 'impute'
                signal.alarm(a.timeout)
                t1 = time.time()
                proba, extra = RUNNERS[model](Xtr, ytr, Xte, info['num_classes'], a)
                signal.alarm(0)
                acc, auc = metrics(yte, proba)
                rec.update(status='success', best_accuracy=acc, best_auc=auc, fit_seconds=time.time() - t1, **extra)
                print(f'[{model}] {i}/{len(todo)} {tag}: acc={acc:.2f}% auc={auc:.4f} ({time.time() - t1:.1f}s)')
            except SkipDataset as e:
                signal.alarm(0)
                rec.update(status='skipped', reason=str(e))
                print(f'[{model}] {i}/{len(todo)} {tag}: skipped by design ({e})')
            except Timeout:
                rec.update(status='timeout', error=f'timeout after {a.timeout}s')
                print(f'[{model}] {i}/{len(todo)} {tag}: TIMEOUT')
            except Exception as e:
                signal.alarm(0)
                rec.update(status='failed', error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc()[-3000:])
                print(f'[{model}] {i}/{len(todo)} {tag}: FAILED {type(e).__name__}: {e}')
            rec['elapsed_total'] = time.time() - t0
            rec['finished'] = datetime.now().isoformat()
            write_json(jpath, rec)
        summ = summarize(mdir, level, model, names_all)
        print(f"--> {model} {level}: {summ['counts']}  mean acc={summ['mean_accuracy']}  mean auc={summ['mean_auc']}  "
              f"COMPLETE={summ['COMPLETE']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
