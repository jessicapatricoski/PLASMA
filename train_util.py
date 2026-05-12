import pandas as pd
import numpy as np
import random
import math
import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from dataclasses import dataclass
import json
from sklearn.metrics import auc, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, f1_score, recall_score, balanced_accuracy_score, precision_score, accuracy_score, confusion_matrix
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from torch.utils.data import Dataset, DataLoader
from model import PLASMA

# PARAMS + DATA BUILDERS
class ModelParams:
    def __init__(self, columns, **kwargs):
        if kwargs.get('_from_json', False): return
        defaults = dict(
            version = 'CEM',
            clin = dict(input_drop=0, n=[16], out=16, dropout=[.1], act='gelu', norm=False),
            expr = dict(input_drop=0, n=[32, 16], out=16, dropout=[.3, .1], act='gelu', norm=True),
            mut  = dict(input_drop=.1, n=[32, 16], out=16, dropout=[.3, .1], act='leaky_relu', norm=False),
            fus  = dict(embed=16, n=[16], dropout=[0.3], act='leaky_relu'),
            att  = dict(use=True, n_heads=4, dropout=.1),
            aux  = dict(weight=0.1, decay=8), # initial auxiliary loss weight (decays during training)
            mod_dropout = 0.15,
            clin_opt = dict(opt='adamw', lr=3e-4, weight_decay=1e-4),
            expr_opt = dict(opt='adamw', lr=3e-4, weight_decay=1e-4),
            mut_opt  = dict(opt='adamw', lr=3e-4, weight_decay=1e-4),
            fus_opt  = dict(opt='adamw', lr=3e-4, weight_decay=1e-4),
            es = dict(use=True, patience=5, metric='auc', grace_epochs=12),
            sched = dict(patience=5, metric='auc', factor=.5, cooldown=0),
            max_grad_norm = None,          
            use_pos_weight = True, # use n_neg/n_pos in BCEWithLogitsLoss for class imbalance
            threshold_target = 'gmean',
            epochs = 30,
            bs = 12,
            restore_best = True    
        )
        for param in defaults.keys():
            dflt = defaults[param]
            new = {**dflt, **kwargs.get(param, {})} if isinstance(dflt, dict) else kwargs.get(param, dflt)
            setattr(self, param, new)
        self.features = [list(columns['clin']), list(columns['expr']), list(columns['mut'])]

class MultiModalDataset(Dataset):
    def __init__(self, clin_df, expr_df, mut_df, labels):
        super().__init__()
        self.ids = clin_df.index.to_numpy()
        assert len(set(self.ids)) == len(self.ids), f"Duplicate IDs found in dataset index; ensemble aggregation would silently drop samples."
        self.clin_data = torch.tensor(clin_df.to_numpy(), dtype=torch.float32)
        self.expr_data = torch.tensor(expr_df.to_numpy(), dtype=torch.float32)
        self.mut_data = torch.tensor(mut_df.to_numpy(), dtype=torch.float32)
        self.labels = torch.tensor(labels.to_numpy(), dtype=torch.float32)
    def __getitem__(self, idx):
        return (self.clin_data[idx], self.expr_data[idx], self.mut_data[idx], self.labels[idx], self.ids[idx])
    def __len__(self):
        return len(self.labels)

def build_datasets(params, xtr, xva, xte, ytr, yva, yte, trcr_x, trcr_y):
    def ds(x, y):
        return MultiModalDataset(x[params.features[0]], x[params.features[1]], x[params.features[2]], y)
    return {'train': ds(xtr, ytr), 'val': ds(xva, yva), 'test_tcga': ds(xte, yte), 'test_trcr': ds(trcr_x, trcr_y)}

def build_loaders(datasets, params, seed=42, pin_memory=True):
    g = torch.Generator().manual_seed(seed)
    train     = DataLoader(datasets['train'], batch_size=params.bs, shuffle=True, generator=g, drop_last=False, num_workers=0, pin_memory=pin_memory)
    val       = DataLoader(datasets['val'], batch_size=params.bs, shuffle=False, drop_last=False, num_workers=0, pin_memory=pin_memory)
    test_tcga = DataLoader(datasets['test_tcga'], batch_size=params.bs, shuffle=False, drop_last=False, num_workers=0, pin_memory=pin_memory)
    test_trcr = DataLoader(datasets['test_trcr'], batch_size=params.bs, shuffle=False, drop_last=False, num_workers=0, pin_memory=pin_memory)
    return train, val, test_tcga, test_trcr

def generate_seeds(n=10, seed=42, print_seeds=True):
    random.seed(seed)
    s_ = random.sample(range(1, 10_000), n)
    if print_seeds: print(f"seeds = {s_}")
    return s_

# TRAINING HELPERS
def get_optimizers_and_schedulers(model, params):
    optimizers, schedulers = {}, {}
    is_fusion = len(params.version) > 1
    if 'C' in params.version:
        optm = {'adam': optim.Adam, 'adamw': optim.AdamW}[params.clin_opt['opt']]
        optimizers['enc_clin'] = optm(model.enc_clin.parameters(), lr=params.clin_opt['lr'], weight_decay=params.clin_opt['weight_decay'])
    if 'E' in params.version:
        optm = {'adam': optim.Adam, 'adamw': optim.AdamW}[params.expr_opt['opt']]
        optimizers['enc_expr'] = optm(model.enc_expr.parameters(), lr=params.expr_opt['lr'], weight_decay=params.expr_opt['weight_decay'])
    if 'M' in params.version:
        optm = {'adam': optim.Adam, 'adamw': optim.AdamW}[params.mut_opt['opt']]
        optimizers['enc_mut'] = optm(model.enc_mut.parameters(), lr=params.mut_opt['lr'], weight_decay=params.mut_opt['weight_decay'])
    optm = {'adam': optim.Adam, 'adamw': optim.AdamW}[params.fus_opt['opt']]
    if is_fusion:
        fusion_params = list(model.head.parameters()) + list(model.fusion.parameters())
        if hasattr(model, 'aux_heads'): fusion_params += list(model.aux_heads.parameters())
        optimizers['head'] = optm(fusion_params, lr=params.fus_opt['lr'], weight_decay=params.fus_opt['weight_decay'])
    else:
        optimizers['head'] = optm(model.head.parameters(), lr=params.fus_opt['lr'], weight_decay=params.fus_opt['weight_decay'])
    sched_mode = 'min' if params.sched['metric'] == 'loss' else 'max'
    for k, o in optimizers.items():
        schedulers[k] = optim.lr_scheduler.ReduceLROnPlateau(o, mode=sched_mode, factor=params.sched['factor'], patience=params.sched['patience'], cooldown=params.sched['cooldown'])
    return optimizers, schedulers

def get_input(c, e, m, params):
    x_out = {}
    if 'C' in params.version: x_out['x_c'] = c
    if 'E' in params.version: x_out['x_e'] = e
    if 'M' in params.version: x_out['x_m'] = m
    return x_out

def _aux_loss_weight(epoch, init_weight=0.3, decay_epochs=25):
    """ Linear decay of auxiliary loss weight from init_weight (epoch 1) → 0 (epoch decay_epochs+1). Epoch is 1-indexed; at epoch 1 the weight is exactly init_weight, 
    and at epoch decay_epochs+1 it is 0 (and stays 0 thereafter); ended up setting this to 0 """
    return init_weight * max(0.0, 1.0 - (epoch - 1) / decay_epochs)

def train_one_epoch(model, dataloader, optimizers, loss_fn, device, params, epoch=1):
    model.train()
    tracking = {k: [] for k in ['ids', 'losses', 'batch_sizes', 'logits', 'probs', 'targets']}
    aux_weight = _aux_loss_weight(epoch, params.aux['weight'], params.aux['decay'])
    for (clin_x, expr_x, mut_x, y, ids) in dataloader:
        tracking['ids'].extend(list(ids))
        clin_x, expr_x, mut_x, y = [t.to(device).float() for t in (clin_x, expr_x, mut_x, y)]
        for o in optimizers.values():
            o.zero_grad()
        logits, probs, _, aux_logits = model(**get_input(clin_x, expr_x, mut_x, params)) # PLASMA.forward returns (logits, probs, attn_weights, aux_logits)
        primary_loss = loss_fn(logits, y)
        tracking['batch_sizes'].append(int(y.shape[0]))
        # if using aux weight, build total loss = primary + aux (fusion mode + early epochs only)
        loss = primary_loss
        if aux_logits and aux_weight > 0:
            for key, alogit in aux_logits.items():
                loss = loss + aux_weight * loss_fn(alogit, y)
        loss.backward()
        if params.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.max_grad_norm)
        for o in optimizers.values():
            o.step()
        tracking['losses'].append(loss.item())
        tracking['logits'].append(logits.detach().cpu().numpy())
        tracking['probs'].append(probs.detach().cpu().numpy())
        tracking['targets'].append(y.detach().cpu().numpy())
    # sample-weighted mean across batches (unbiased when last batch is smaller)
    losses = np.asarray(tracking['losses'], dtype=float)
    bsz = np.asarray(tracking['batch_sizes'], dtype=float)
    mean_loss = float(np.sum(losses * bsz) / np.sum(bsz)) if bsz.sum() > 0 else float('nan')
    for k in ['logits', 'probs', 'targets']:
        tracking[k] = np.concatenate(tracking[k], axis=0)
    tracking['ids'] = np.array(tracking['ids'])
    return mean_loss, tracking

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device, params):
    model.eval()
    tracking = {k: [] for k in ['ids', 'losses', 'batch_sizes', 'logits', 'probs', 'targets', 'attn']}
    tracking['aux_logits'] = {k: [] for k in params.version}
    if loss_fn is None: loss_fn = nn.BCEWithLogitsLoss()
    for (clin_x, expr_x, mut_x, y, ids) in dataloader:
        tracking['ids'].extend(list(ids))
        clin_x, expr_x, mut_x, y = [t.to(device).float() for t in (clin_x, expr_x, mut_x, y)]
        logits, probs, attn, aux_logits = model(**get_input(clin_x, expr_x, mut_x, params))
        loss = loss_fn(logits, y)
        tracking['losses'].append(loss.item())
        tracking['batch_sizes'].append(int(y.shape[0]))
        tracking['logits'].append(logits.detach().cpu().numpy())
        tracking['probs'].append(probs.detach().cpu().numpy())
        tracking['targets'].append(y.detach().cpu().numpy())
        if attn is not None:
            tracking['attn'].append(attn.detach().cpu().numpy())
        for k, alogit in aux_logits.items():
            tracking['aux_logits'][k].append(alogit.detach().cpu().numpy())
    for k in ['logits', 'probs', 'targets']:
        tracking[k] = np.concatenate(tracking[k], axis=0)
    tracking['attn'] = None if len(tracking['attn']) == 0 else np.concatenate(tracking['attn'], axis=0)
    for k in tracking['aux_logits']:
        if tracking['aux_logits'][k]:
            tracking['aux_logits'][k] = np.concatenate(tracking['aux_logits'][k], axis=0)
    tracking['ids'] = np.array(tracking['ids'])
    auc_ = roc_auc_score(tracking['targets'], tracking['probs'])
    prec, rec, _ = precision_recall_curve(tracking['targets'], tracking['probs'])
    auprc_ = auc(rec, prec)
    return auc_, auprc_, tracking

def check_progress(current, best, counter, epoch, es_params, mode='max'):
    def is_improved():
        if best is None: return True
        if mode == 'max': return current > best + 1e-6
        if mode == 'min': return current < best - 1e-6
    improved = is_improved()
    if improved:
        best, counter = current, 0
    elif epoch > es_params['grace_epochs']:
        counter += 1
    time_to_quit = (es_params['use']) and (counter >= es_params['patience'])
    return best, counter, improved, time_to_quit

def train_seed(model, params, train_loader, val_loader, device):
    """ Train for one seed with auxiliary loss support (if applicable) and per-encoder scheduler stepping.
    - pos_weight=n_neg/n_pos is applied to BCEWithLogitsLoss when params.use_pos_weight is True, to handle class imbalance (computed from the training set only)
    - Val loss is the sample-weighted mean over batches (unbiased for non-divisible batch counts)
    - The 'head' (fusion + prediction head) scheduler is stepped by the full val metric """
    # compute pos_weight from the training labels only
    if getattr(params, 'use_pos_weight', True):
        y_train = train_loader.dataset.labels
        n_pos = float(y_train.sum().item())
        n_neg = float(y_train.numel() - n_pos)
        pos_weight = torch.tensor([n_neg / n_pos], device=device) if n_pos > 0 else None
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    epoch_losses = {'train': [], 'val': []}
    optimizers, schedulers = get_optimizers_and_schedulers(model, params)
    best_score, best_state, best_epoch = None, None, None
    counter = 0
    for epoch in range(1, params.epochs + 1):
        train_loss, _ = train_one_epoch(model, train_loader, optimizers, loss_fn, device, params, epoch=epoch)
        epoch_losses['train'].append(train_loss)
        val_auc, val_auprc, val_tracking = evaluate(model, val_loader, loss_fn, device, params)
        # sample-weighted val loss (unbiased for unequal last-batch)
        v_losses = np.asarray(val_tracking['losses'], dtype=float)
        v_bsz    = np.asarray(val_tracking['batch_sizes'], dtype=float)
        val_loss = float(np.sum(v_losses * v_bsz) / np.sum(v_bsz)) if v_bsz.sum() > 0 else float('nan')
        epoch_losses['val'].append(val_loss)
        # scheduler stepping
        aux_logits_val = val_tracking.get('aux_logits', {})
        val_targets = val_tracking['targets']
        is_fusion = len(params.version) > 1
        use_metric = params.sched['metric']
        if is_fusion and aux_logits_val:
            # step per-encoder schedulers with their own auxiliary val signal (if using, else fallback)
            for key in params.version:
                sched_key = {'C': 'enc_clin', 'E': 'enc_expr', 'M': 'enc_mut'}[key]
                aux_raw = aux_logits_val.get(key)  # raw logits, shape (N,)
                fallback = val_loss if use_metric == 'loss' else val_auc
                if sched_key in schedulers and aux_raw is not None and len(aux_raw) > 0:
                    try:
                        if use_metric == 'loss':
                            aux_metric = F.binary_cross_entropy_with_logits(torch.tensor(aux_raw, dtype=torch.float32), torch.tensor(val_targets, dtype=torch.float32)).item()
                        else:
                            aux_metric = roc_auc_score(val_targets, aux_raw) # roc_auc_score is monotone-invariant, so raw logits work directly
                        schedulers[sched_key].step(aux_metric)
                    except Exception as e:
                        print(f"[warn] per-encoder scheduler step failed for '{sched_key}' ({type(e).__name__}: {e}); falling back to combined val metric")
                        schedulers[sched_key].step(fallback)
                elif sched_key in schedulers:
                    schedulers[sched_key].step(fallback)
            if 'head' in schedulers:  # step head/fusion scheduler with combined val
                schedulers['head'].step(val_loss if use_metric == 'loss' else val_auc)
        else: # single-modality or no aux logits: step all with combined val
            use_val = val_loss if use_metric == 'loss' else val_auc
            for s in schedulers.values():
                s.step(use_val)
        check_score, mode = {'auc': [val_auc, 'max'], 'loss': [val_loss, 'min']}[params.es['metric']]
        best_score, counter, is_improved, time_to_quit = check_progress(check_score, best_score, counter, epoch, params.es, mode=mode)
        if is_improved:
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
        elif time_to_quit:
            break
    if params.restore_best and best_state is not None:
        model.load_state_dict(best_state)
    return model, best_score, best_epoch, epoch_losses

def set_global_determinism(seed, strict=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if strict:
        torch.use_deterministic_algorithms(True)
        torch.set_deterministic_debug_mode("error")

# THRESHOLD SELECTION
def choose_threshold_youden(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    return float(thr[best_idx]), {'fpr': float(fpr[best_idx]), 'tpr': float(tpr[best_idx])}

def choose_threshold_max_gmean(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    spec = 1.0 - fpr
    gmean = np.sqrt(tpr * spec)
    best_idx = int(np.argmax(gmean))
    return float(thr[best_idx]), {'tpr': float(tpr[best_idx]), 'spec': float(spec[best_idx]), 'gmean': float(gmean[best_idx])}

def choose_threshold_min_spec_gmean(y_true, y_prob, min_spec=0.50):
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    ok = np.isfinite(y_prob)
    y_true = y_true[ok]; y_prob = y_prob[ok]
    thr_candidates = np.unique(y_prob)
    best, best_meta, best_thr = None, None, None
    for thr in thr_candidates:
        y_hat = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        gmean = np.sqrt(spec * rec)
        if spec >= min_spec:
            if (best is None) or (gmean > best + 1e-12):
                best = gmean; best_thr = float(thr)
                best_meta = {"tpr": float(rec), "spec": float(spec), "gmean": float(gmean), "min_spec": float(min_spec)}
    if best is None:
        print(f"No threshold achieves min_spec={min_spec}.")
    return best_thr, best_meta

def choose_threshold(y_true, y_prob, target, min_spec=0.50):
    target = (target or 'youden').lower()
    if target in {'youden', 'j'}: return choose_threshold_youden(y_true, y_prob)
    if target in {'gmean', 'max_gmean'}: return choose_threshold_max_gmean(y_true, y_prob)
    if target in {'min_spec_gmean'}: return choose_threshold_min_spec_gmean(y_true, y_prob, min_spec=min_spec)
    raise ValueError(f"Unknown threshold_target: {target}")

# CALIBRATION
class Calibrator:
    def __init__(self, kind='platt'):
        self.kind = (kind or 'none').lower()
        self.model = None

    def fit(self, val_logits, val_labels):
        val_logits = np.asarray(val_logits).reshape(-1, 1).astype(float)
        val_labels = np.asarray(val_labels).astype(int).ravel()
        if self.kind == 'none':    self.model = None; return self
        if self.kind == 'platt':
            lr = LogisticRegression(solver='lbfgs', max_iter=1000)
            lr.fit(val_logits, val_labels); self.model = lr; return self
        if self.kind == 'isotonic':
            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(val_logits.ravel(), val_labels); self.model = ir; return self
        raise ValueError(f'Unknown calibrator kind: {self.kind}')

    def transform(self, logits):
        logits = np.asarray(logits).reshape(-1, 1).astype(float)
        if self.kind == 'none': return expit(logits.ravel())
        if self.kind == 'platt': return self.model.predict_proba(logits)[:, 1]
        if self.kind == 'isotonic': return self.model.transform(logits.ravel())
        raise RuntimeError('Calibrator not fit or unknown kind.')

# METRICS
def _to_1d_numpy(x):
    if isinstance(x, torch.Tensor): x = x.detach().cpu().numpy()
    if isinstance(x, (pd.Series, pd.Index)): x = x.to_numpy()
    elif isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0].to_numpy() if x.shape[1] == 1 else x.to_numpy().squeeze()
    else: x = np.asarray(x)
    x = np.asarray(x).squeeze()
    if x.ndim != 1: raise ValueError(f"Expected 1D array, got shape {x.shape}")
    return x

def compute_metrics(y_true, y_prob, threshold=0.5, plot_metrics=False):
    y_true = _to_1d_numpy(y_true).astype(int)
    y_prob = _to_1d_numpy(y_prob).astype(float)
    pred   = (y_prob >= threshold).astype(int)
    y_unique = np.unique(y_true)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    prec_c, rec_c, _ = precision_recall_curve(y_true, y_prob)
    mdict = {
        'auc': float(roc_auc_score(y_true, y_prob)) if len(y_unique)>1 else float('nan'),
        'auprc': float(auc(rec_c, prec_c)) if len(y_unique)>1 else float('nan'),
        'avg_prec': float(average_precision_score(y_true, y_prob)) if len(y_unique)>1 else float('nan'),
        'acc': float(accuracy_score(y_true, pred)),
        'bal_acc': float(balanced_accuracy_score(y_true, pred)),
        'prec': float(precision_score(y_true, pred, zero_division=0)),
        'rec': float(recall_score(y_true, pred, zero_division=0)),
        'spec': float(tn / (tn + fp)) if (tn + fp) > 0 else float('nan'),
        'f1': float(f1_score(y_true, pred, zero_division=0)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
    }
    if plot_metrics:
        fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
        mdict.update({'prc_prec': prec_c.tolist(), 'prc_rec': rec_c.tolist(), 'fpr': fpr.tolist(), 'tpr': tpr.tolist()})
    return mdict

def bootstrap_metrics(y_true, y_prob, threshold, n_boot=1000, seed=0):
    y_true = _to_1d_numpy(y_true).astype(int)
    y_prob = _to_1d_numpy(y_prob).astype(float)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    keys = list(compute_metrics(y_true, y_prob, threshold).keys())
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        m = compute_metrics(y_true[idx], y_prob[idx], threshold)
        for k in keys: samples[k].append(m[k])
    out = {}
    for k in keys:
        arr = np.asarray(samples[k], dtype=float)
        out[k] = (float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 50.0)), float(np.nanpercentile(arr, 97.5)))
    return out

# TRAIN
def ensemble_split_by_id(per_seed_outputs, seeds, split):
    """ Aggregate per-seed predictions by patient ID. The ensemble aggregates in LOGIT space (mean-of-logits), which is the standard ensembling method for binary classifiers
    'probs' is sigmoid(mean_logits), NOT mean(sigmoid(logits)); this ensures that 'probs' and 'probs_cal' (downstream) are derived from the same underlying aggregation """
    seed_maps = []
    for s in seeds:
        out = per_seed_outputs[s][split]
        id2prob = {i: p for i, p in zip(out['ids'], out['probs'])}
        id2log  = {i: z for i, z in zip(out['ids'], out['logits'])}
        id2y    = {i: y for i, y in zip(out['ids'], out['targets'])}
        seed_maps.append((id2prob, id2log, id2y))
    common_ids = set(seed_maps[0][0].keys())
    for id2prob, _, _ in seed_maps[1:]: common_ids &= set(id2prob.keys())
    common_ids = sorted(common_ids)
    logits_stack = np.stack([[id2log[i]  for i in common_ids] for (_, id2log, _)  in seed_maps], axis=0)
    y = np.array([seed_maps[0][2][i] for i in common_ids], dtype=float)
    mean_logits = logits_stack.mean(axis=0)
    return {'ids': np.array(common_ids), 'probs': expit(mean_logits), 'logits': mean_logits, 'targets': y}

def multiseed_train_eval(params, datadict, device, seeds, calibrator_kind='platt', n_boot=1000, boot_seed=0, print_thresholds=False, strict_determinism=False, pin_memory=True, plot_metrics=False):
    """ Train <n_seeds> models, ensemble them, then calibrate and threshold """
    per_seed_outputs = {}
    per_seed_models  = []
    per_seed_epoch_losses = {}
    per_seed_best = {}  # {seed: {'best_score': ..., 'best_epoch': ...}}
    datasets = build_datasets(params, **datadict)
    for seed in seeds:
        set_global_determinism(seed, strict=strict_determinism)
        train_loader, val_loader, test_tcga_loader, test_trcr_loader = build_loaders(datasets, params, seed=seed, pin_memory=pin_memory)
        model = PLASMA(params).to(device)
        model, best_score, best_epoch, epoch_losses = train_seed(model, params, train_loader, val_loader, device)
        per_seed_models.append(model)
        per_seed_best[seed] = {'best_score': best_score, 'best_epoch': best_epoch}
        _, _, out_val  = evaluate(model, val_loader, None, device, params)
        _, _, out_test = evaluate(model, test_tcga_loader, None, device, params)
        _, _, out_trx  = evaluate(model, test_trcr_loader, None, device, params)
        per_seed_outputs[seed] = {'tcga_val': out_val, 'tcga_test': out_test, 'tracerx': out_trx}
        per_seed_epoch_losses[seed] = epoch_losses
    ens = {k: ensemble_split_by_id(per_seed_outputs, seeds, k) for k in ['tcga_val', 'tcga_test', 'tracerx']}
    cal = Calibrator(kind=calibrator_kind).fit(ens['tcga_val']['logits'], ens['tcga_val']['targets'].astype(int))
    for split in ['tcga_val', 'tcga_test', 'tracerx']:
        ens[split]['probs_cal'] = cal.transform(ens[split]['logits'])
    thr, meta = choose_threshold(y_true=ens['tcga_val']['targets'].astype(int), y_prob=ens['tcga_val']['probs_cal'], target=params.threshold_target)
    if print_thresholds: print(f'Threshold: {thr} ({meta})')
    results = {}
    for split in ['tcga_val', 'tcga_test', 'tracerx']:
        y = ens[split]['targets'].astype(int)
        p = ens[split]['probs_cal']
        point = compute_metrics(y, p, threshold=thr, plot_metrics=plot_metrics)
        ci    = bootstrap_metrics(y, p, threshold=thr, n_boot=n_boot, seed=boot_seed) if n_boot > 0 else None
        results[split] = {'metrics': point, 'bootstrap_ci': ci, 'prevalence': float(np.mean(y))}
    return {'per_seed_models': per_seed_models,
            'per_seed_outputs': per_seed_outputs,
            'per_seed_epoch_losses': per_seed_epoch_losses,
            'per_seed_best': per_seed_best,
            'ensemble': ens,
            'calibrator': cal,
            'threshold': thr,
            'results': results,
            'datasets': datasets}