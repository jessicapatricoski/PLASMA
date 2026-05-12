""" Hyperparameter tuning for PLASMA via stratified k-fold cross-validation on the dev set (train + val merged).
Optuna-based TPE search over a default or custom HP space. Uses fold-level MedianPruner to abort unpromising trials after a few folds.
1. xtr + xva are merged into a dev pool; xte (TCGA-test) and TRACERx are held out completely.
2. Stratified k-fold CV on dev; within each fold, 15% of the training portion is reserved as an inner val for restore_best/LR scheduling. The held-out fold is used only to score the HP config.
3. HP configs are ranked by mean cross-fold AUROC
4. Once best HPs are chosen, the full multiseed_train_eval pipeline is run on the original train/val/test splits and all test numbers are reported.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from train_util import ModelParams, MultiModalDataset, train_seed, evaluate, generate_seeds, set_global_determinism
from model import PLASMA

def _build_dev_pool(datadict):
    """ Merge datadict['xtr'/'ytr'] and ['xva'/'yva'] into a single dev pool; TCGA-test and TRACERx are never touched """
    dev_x = pd.concat([datadict['xtr'], datadict['xva']], axis=0)
    dev_y = pd.concat([datadict['ytr'], datadict['yva']], axis=0)
    assert len(set(dev_x.index)) == len(dev_x.index), "Duplicate patient IDs in merged dev set (xtr + xva). Check for overlap between train and val splits before running CV."
    assert (dev_x.index == dev_y.index).all(), "xtr+xva and ytr+yva indices don't align after concatenation."
    return dev_x, dev_y

def _make_fold_loaders(params, dev_x, dev_y, dev_y_arr, tr_idx, va_idx, inner_val_frac, loader_seed, fold_seed, pin_memory=True):
    """ Build the three DataLoaders needed for one CV fold. tr_idx gets split into (inner_tr, inner_va) for restore_best signal; va_idx is the held-out fold used only for HP scoring """
    inner_tr, inner_va = train_test_split(tr_idx, test_size=inner_val_frac, stratify=dev_y_arr[tr_idx], random_state=fold_seed)
    def _ds(x_df, y_df):
        return MultiModalDataset(x_df[params.features[0]], x_df[params.features[1]], x_df[params.features[2]], y_df)
    ds_tr = _ds(dev_x.iloc[inner_tr], dev_y.iloc[inner_tr])
    ds_va = _ds(dev_x.iloc[inner_va], dev_y.iloc[inner_va])
    ds_ot = _ds(dev_x.iloc[va_idx],   dev_y.iloc[va_idx])
    g = torch.Generator().manual_seed(loader_seed)
    tr_l = DataLoader(ds_tr, batch_size=params.bs, shuffle=True, generator=g, drop_last=False, num_workers=0, pin_memory=pin_memory)
    va_l = DataLoader(ds_va, batch_size=params.bs, shuffle=False, num_workers=0, pin_memory=pin_memory)
    ot_l = DataLoader(ds_ot, batch_size=params.bs, shuffle=False, num_workers=0, pin_memory=pin_memory)
    return tr_l, va_l, ot_l

def _train_and_score_fold(params, dev_x, dev_y, dev_y_arr, tr_idx, va_idx, n_seeds_per_fold, inner_val_frac, base_seed, fold_i, device, pin_memory=True):
    """ Train <n_seeds> models on this fold and return mean fold AUROC / AUPRC """
    fold_seeds = generate_seeds(n=n_seeds_per_fold, seed=base_seed + 10_000 * (fold_i + 1), print_seeds=False)
    seed_aucs, seed_auprcs = [], []
    for s in fold_seeds:
        set_global_determinism(s)
        tr_l, va_l, ot_l = _make_fold_loaders(
            params, dev_x, dev_y, dev_y_arr, tr_idx, va_idx,
            inner_val_frac=inner_val_frac,
            loader_seed=s,
            fold_seed=base_seed + fold_i, # stable across trials
            pin_memory=pin_memory,
        )
        model = PLASMA(params).to(device)
        model, _, _, _ = train_seed(model, params, tr_l, va_l, device)
        outer_auc, outer_auprc, _ = evaluate(model, ot_l, None, device, params)
        seed_aucs.append(outer_auc)
        seed_auprcs.append(outer_auprc)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {'auc': float(np.mean(seed_aucs)), 'auprc': float(np.mean(seed_auprcs)), 'seed_aucs': seed_aucs, 'seed_auprcs': seed_auprcs}

def cross_validate_hparams(params, datadict, device, k=5, n_seeds_per_fold=1, inner_val_frac=0.15, seed=42, pin_memory=True, verbose=False):
    """ Score a single HP configuration via stratified k-fold CV on (xtr + xva) """
    dev_x, dev_y = _build_dev_pool(datadict)
    dev_y_arr = dev_y.to_numpy().ravel()
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_results = []
    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(np.arange(len(dev_y_arr)), dev_y_arr)):
        res = _train_and_score_fold(params, dev_x, dev_y, dev_y_arr, tr_idx, va_idx, n_seeds_per_fold=n_seeds_per_fold, inner_val_frac=inner_val_frac, base_seed=seed, fold_i=fold_i, device=device, pin_memory=pin_memory)
        res['fold'] = fold_i
        fold_results.append(res)
        if verbose:
            print(f"  Fold {fold_i+1}/{k}: AUC={res['auc']:.4f}, AUPRC={res['auprc']:.4f}")
    aucs   = np.array([f['auc']   for f in fold_results])
    auprcs = np.array([f['auprc'] for f in fold_results])
    return {
        'mean_auc':   float(aucs.mean()),
        'std_auc':    float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
        'mean_auprc': float(auprcs.mean()),
        'std_auprc':  float(auprcs.std(ddof=1)) if len(auprcs) > 1 else 0.0,
        'fold_results': fold_results # [{'fold', 'auc', 'auprc', 'seed_aucs', 'seed_auprcs'}, ...]
    }

# TUNE
def default_search_space(trial):
    return {
        'clin_opt': dict(opt='adam', lr=.0003, weight_decay=trial.suggest_categorical('clin_wd', [.0003, .0001, .001])),
        'expr_opt': dict(opt='adam', lr=.0003, weight_decay=trial.suggest_categorical('expr_wd', [.0003, .0001, .001])),
        'mut_opt': dict(opt='adam', lr=.0003, weight_decay=trial.suggest_categorical('mut_wd', [.0003, .0001, .001])),
        'fus_opt': dict(opt='adam', lr=.0001, weight_decay=trial.suggest_categorical('fus_wd', [.0003, .0001, .001])),
        'fus': dict(embed=16, n=[16], dropout=[trial.suggest_categorical('fus_drop', [0, .1, .2])], act='leaky_relu')
    }

def _deep_merge(base, override):
    # override wins; one level of nested-dict merging
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out

def tune_hyperparameters(datadict, columns, device, n_trials=80, k=5, n_seeds_per_fold=1, inner_val_frac=0.15, seed=42, base_overrides=None, search_space=None, 
                         study_name='plasma_hp', storage=None, n_startup_trials=10, n_warmup_folds=2, dedup=True, pin_memory=True, verbose=True):
    """ Optuna TPE search with fold-level MedianPruner; when dedup == True, short-circuit trials whose params dict exactly matches a previously-completed trial and 
    return the cached value without retraining (safe when training is deterministic) """
    import optuna
    base_overrides = base_overrides or {}
    search_fn = search_space or default_search_space
    # pre-compute fold splits once; identical across all trials, so every HP config is evaluated on the exact same partitions
    dev_x, dev_y = _build_dev_pool(datadict)
    dev_y_arr = dev_y.to_numpy().ravel()
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_splits = list(skf.split(np.arange(len(dev_y_arr)), dev_y_arr))
    def objective(trial):
        hp_overrides = search_fn(trial)
        all_overrides = _deep_merge(base_overrides, hp_overrides)
        # TPE samples each parameter dimension independently from its KDE-fitted posterior, so once a combo is identified as promising it can be re-proposed many
        # times; with deterministic training, re-running yields a bit-identical value. Return the cached value (and copy diagnostics) instead
        if dedup:
            for prev in trial.study.trials:
                if prev.number == trial.number or prev.params != trial.params:
                    continue
                if prev.state.name == 'COMPLETE':
                    for k_, v_ in prev.user_attrs.items():
                        trial.set_user_attr(k_, v_)
                    trial.set_user_attr('duplicate_of', int(prev.number))
                    return prev.value
                if prev.state.name == 'PRUNED':
                    raise optuna.TrialPruned()
        params = ModelParams(columns, **all_overrides)
        fold_aucs, fold_auprcs = [], []
        for fold_i, (tr_idx, va_idx) in enumerate(fold_splits):
            res = _train_and_score_fold(
                params, dev_x, dev_y, dev_y_arr, tr_idx, va_idx, n_seeds_per_fold=n_seeds_per_fold, inner_val_frac=inner_val_frac, base_seed=seed, fold_i=fold_i, device=device, pin_memory=pin_memory
            )
            fold_aucs.append(res['auc'])
            fold_auprcs.append(res['auprc'])
            trial.report(float(np.mean(fold_aucs)), step=fold_i)
            if trial.should_prune():
                trial.set_user_attr('fold_aucs', list(map(float, fold_aucs)))
                trial.set_user_attr('fold_auprcs', list(map(float, fold_auprcs)))
                raise optuna.TrialPruned()
        fold_aucs_arr = np.asarray(fold_aucs, dtype=float)
        # log diagnostics for post-hoc tie-breaking
        trial.set_user_attr('fold_aucs', fold_aucs_arr.tolist())
        trial.set_user_attr('fold_auprcs', list(map(float, fold_auprcs)))
        trial.set_user_attr('fold_mean_auc', float(fold_aucs_arr.mean()))
        trial.set_user_attr('fold_std_auc', float(fold_aucs_arr.std(ddof=1)) if len(fold_aucs_arr) > 1 else 0.0)
        trial.set_user_attr('fold_mean_auprc', float(np.mean(fold_auprcs)))
        return float(fold_aucs_arr.mean())
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_folds)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner, study_name=study_name, storage=storage, load_if_exists=(storage is not None))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)
    return {'study': study, 'best_value': float(study.best_value), 'best_params': dict(study.best_params)}
 
def best_trial_with_tiebreak(study, tol=1e-6, secondary='fold_std', custom_key=None, return_meta=False):
    """ Pick a best trial from a study, breaking ties on a secondary criterion. Optuna's study.best_trial returns the earliest-numbered trial among any tied for the 
    top value (no built-in tie-break) """
    completed = [t for t in study.trials if t.state.name == 'COMPLETE']
    if not completed:
        raise ValueError("Study has no COMPLETE trials.")
    direction = study.direction.name  # 'MAXIMIZE' or 'MINIMIZE'
    if direction == 'MAXIMIZE':
        top = max(t.value for t in completed)
        tied = [t for t in completed if (top - t.value) <= tol]
    else:
        top = min(t.value for t in completed)
        tied = [t for t in completed if (t.value - top) <= tol]
    if len(tied) == 1 or secondary == 'first':
        chosen = min(tied, key=lambda t: t.number)
    elif custom_key is not None:
        chosen = min(tied, key=custom_key)
    elif secondary == 'last':
        chosen = max(tied, key=lambda t: t.number)
    elif secondary == 'fold_std':
        chosen = min(tied, key=lambda t: t.user_attrs.get('fold_std_auc', float('inf')))
    elif secondary == 'fold_mean_auprc':
        chosen = max(tied, key=lambda t: t.user_attrs.get('fold_mean_auprc', float('-inf')))
    else: raise ValueError(f"Unknown secondary criterion: {secondary!r}")
    if return_meta:
        meta = {
            'top_value': float(top),
            'tol': float(tol),
            'n_tied': len(tied),
            'tied_numbers': sorted(t.number for t in tied),
            'secondary': 'custom' if custom_key is not None else secondary,
            'chosen_number': int(chosen.number),
        }
        return chosen, meta
    return chosen