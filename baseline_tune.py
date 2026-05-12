""" Hyperparameter tuning for ML baselines on the same TCGA train+val dev pool used by PLASMA, via stratified k-fold CV with Optuna GridSampler. Matches the protocol of
hp_tune.tune_hyperparameters as closely as the model class permits (same fold machinery and same seed to produce identical patient cohorts) """

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from hp_tune import _build_dev_pool, best_trial_with_tiebreak

def _get_score(clf, X):
    if hasattr(clf, 'decision_function'):
        s = clf.decision_function(X)
        return np.asarray(s).ravel()
    proba = clf.predict_proba(X)[:, 1]
    return np.asarray(proba).ravel()

# MODEL FACTORIES
def _make_lr(seed, **hp):
    return LogisticRegression(
        l1_ratio=hp.get('l1_ratio', 0.0), C=hp.get('C', 1.0), tol=hp.get('tol', 1e-4), solver='liblinear', max_iter=1000,
        class_weight='balanced', random_state=seed, dual=True
    )

def _make_rf(seed, **hp):
    return RandomForestClassifier(
        n_estimators=hp.get('n_estimators', 200), max_depth=hp.get('max_depth', 3),
        min_samples_leaf=hp.get('min_samples_leaf', 5),
        class_weight='balanced', random_state=seed,
        max_features=hp.get('max_features', 'sqrt')
    )

def _make_mlp(seed, **hp): # inner-val early stopping w/ validation_fraction=0.15 to match PLASMA's inner_val_fract
    return MLPClassifier(
        hidden_layer_sizes=hp.get('hidden_layer_sizes', (64, 32, 16)), alpha=hp.get('alpha', 1e-4),
        learning_rate_init=hp.get('learning_rate_init', 1e-3),
        activation='relu', early_stopping=True, validation_fraction=0.15, n_iter_no_change=10,
        batch_size=12, random_state=seed
    )

def _make_svm(seed, **hp): 
    return SVC(
        kernel='rbf', C=hp.get('C', 1.0), gamma=hp.get('gamma', 'scale'),
        class_weight='balanced', shrinking=hp.get('shrinking', True), probability=True, 
        # deterministic since we don't use predict_proba, we set seed for factory-signature consistency
        random_state=seed
    )

def _make_knn(seed, **hp):
    """ KNN: deterministic given training data and HPs; we accept seed for factory-signature consistency """
    return KNeighborsClassifier(
        n_neighbors=hp.get('n_neighbors', 10), weights=hp.get('weights', 'uniform'), metric='minkowski',
        p=hp.get('p', 2), leaf_size=hp.get('leaf_size', 30)
    )

_MODEL_FACTORIES = {'LR': _make_lr, 'RF': _make_rf, 'MLP': _make_mlp, 'SVM': _make_svm, 'KNN': _make_knn}
_STOCHASTIC = {'LR': False, 'RF': True, 'MLP': True, 'SVM': False, 'KNN': False} # whether the model is genuinely stochastic — only these honor n_seeds_per_fold > 1

# SEARCH SPACES (small by design as baselines have fewer meaningful HPs)
_GRID_SPACES = {
    'LR':  {
        'C': [1e-2, 1e-1, 1.0, 10.0, 100.0],
        'l1_ratio': [0, 1], # penalty is deprecated; 0 = l2, 1 = l1
        'tol': [1e-4, 1e-3, 1e-2]
    },
    'RF':  {
        'max_depth': [3, 5, 8, None],
        'min_samples_leaf': [1, 3, 5],
        'n_estimators': [50, 100, 150],
        'max_features': ['sqrt', 'log2'],
    },
    'MLP': {
        'alpha': [1e-5, 1e-4, 1e-3],
        'lr0': [1e-4, 5e-4, 1e-3, 5e-3],
        # had to serialize as strings because the saved studies (SQLite) can't do tuples and they get re-loaded as lists otherwise (throws warning)
        'hidden_layer_sizes': ['(64,)', '(128,)', '(64,32)', '(128,64)', '(64,32,16)', '(128,64,32)'],
    },
    'SVM': {
        'C': [1, 10, 100],
        'gamma': ['scale', 'auto', 1e-2, 1e-1],
        'shrinking': [True, False]
    },
    'KNN': {
        'n_neighbors': [10, 30, 50],
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
        'leaf_size': [10, 30, 50]
    }
}

def _grid_size(name):
    n = 1
    for v in _GRID_SPACES[name].values():
        n *= len(v)
    return n
 
def _search_space(name, trial):
    if name not in _GRID_SPACES:
        raise ValueError(f"Unknown baseline: {name}")
    out = {k: trial.suggest_categorical(k, v) for k, v in _GRID_SPACES[name].items()}
    if 'lr0' in out:
        out['learning_rate_init'] = out.pop('lr0')
    if 'hidden_layer_sizes' in out and isinstance(out['hidden_layer_sizes'], str):
        out['hidden_layer_sizes'] = tuple(int(x) for x in out['hidden_layer_sizes'].strip('()').rstrip(',').split(','))
    return out

# PER-FOLD SCORING
def _score_fold(name, hp, dev_X, dev_y_arr, tr_idx, va_idx, n_seeds_per_fold, base_seed, fold_i):
    """ Train n_seeds models on fold fold_i and return mean held-out AUC. For deterministic models, n_seeds_per_fold is overridden to 1 as refitting the same deterministic model gives identical scores """
    factory = _MODEL_FACTORIES[name]
    is_stoch = _STOCHASTIC[name]
    n_seeds = n_seeds_per_fold if is_stoch else 1
    X_tr_fold = dev_X[tr_idx]
    y_tr_fold = dev_y_arr[tr_idx]
    X_va_fold = dev_X[va_idx]
    y_va_fold = dev_y_arr[va_idx]
    seed_aucs = []
    for s in range(n_seeds):
        seed = base_seed + 10_000 * (fold_i + 1) + s
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            clf = factory(seed=seed, **hp)
            clf.fit(X_tr_fold, y_tr_fold)
        scores = _get_score(clf, X_va_fold)
        seed_aucs.append(roc_auc_score(y_va_fold, scores))
    return float(np.mean(seed_aucs))

# TUNE
def tune_baseline(name, datadict, columns, k=3, n_seeds_per_fold=3, seed=9999, tiebreak='fold_std', tiebreak_tol=1e-6, storage=None, study_name=None, verbose=False):
    """ Exhaustive grid search for one baseline; every combination in _GRID_SPACES[name] is evaluated exactly once """
    if name not in _MODEL_FACTORIES:
        raise ValueError(f"Unknown baseline {name!r}; expected one of {list(_MODEL_FACTORIES)}")
    import optuna
    # build dev pool (xtr + xva, ALL features concatenated)
    dev_x_df, dev_y_ser = _build_dev_pool(datadict)
    feat_cols = list(columns['clin']) + list(columns['expr']) + list(columns['mut'])
    dev_X = dev_x_df[feat_cols].to_numpy()
    dev_y = dev_y_ser.to_numpy().ravel().astype(int)
    # identical fold splits to PLASMA when seed matches
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    fold_splits = list(skf.split(np.arange(len(dev_y)), dev_y))
    is_stoch = _STOCHASTIC[name]
    n_seeds_eff = n_seeds_per_fold if is_stoch else 1
    if not is_stoch and verbose:
        print(f"[{name}] deterministic model — using 1 seed per fold")
    def objective(trial):
        hp = _search_space(name, trial) # already has lr0 -> learning_rate_init renamed
        fold_aucs = [_score_fold(name, hp, dev_X, dev_y, tr, va, n_seeds_eff, seed, fi) for fi, (tr, va) in enumerate(fold_splits)]
        fold_aucs_arr = np.asarray(fold_aucs, dtype=float)
        trial.set_user_attr('fold_aucs', fold_aucs_arr.tolist())
        trial.set_user_attr('fold_mean_auc', float(fold_aucs_arr.mean()))
        trial.set_user_attr('fold_std_auc', float(fold_aucs_arr.std(ddof=1)) if len(fold_aucs_arr) > 1 else 0.0)
        return float(fold_aucs_arr.mean())
    sampler = optuna.samplers.GridSampler(_GRID_SPACES[name], seed=seed)
    n_trials_eff = _grid_size(name)
    study = optuna.create_study(direction='maximize', sampler=sampler, study_name=(study_name or name), storage=storage, load_if_exists=(storage is not None))
    # if resuming, only run remaining trials (Optuna won't re-run completed ones)
    n_remaining = n_trials_eff - sum(t.state.name == 'COMPLETE' for t in study.trials)
    if n_remaining > 0:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            study.optimize(objective, n_trials=n_remaining, show_progress_bar=verbose)
    elif verbose:
        print(f"[{name}] all {n_trials_eff} grid combos already complete; loaded from storage")
    chosen, tiebreak_meta = best_trial_with_tiebreak(study, tol=tiebreak_tol, secondary=tiebreak, return_meta=True)
    # chosen.params has the trial-parameter names (incl. 'lr0' for MLP); rename for the factory
    final_hp = dict(chosen.params)
    if 'lr0' in final_hp:
        final_hp['learning_rate_init'] = final_hp.pop('lr0')
    if 'hidden_layer_sizes' in final_hp and isinstance(final_hp['hidden_layer_sizes'], str):
        s = final_hp['hidden_layer_sizes'].strip('()').rstrip(',')
        final_hp['hidden_layer_sizes'] = tuple(int(x) for x in s.split(','))
    if verbose:
        n_tied = tiebreak_meta['n_tied']
        if n_tied > 1: print(f"[{name}] best CV AUC: {chosen.value:.4f} ({n_tied} tied; tiebreak={tiebreak_meta['secondary']} -> trial #{chosen.number})")
        else: print(f"[{name}] best CV AUC: {chosen.value:.4f} (trial #{chosen.number})")
        print(f"[{name}] best HPs:    {final_hp}")
    return {
        'name': name,
        'best_hp': final_hp,
        'best_value': float(chosen.value),
        'best_params': dict(chosen.params),
        'study': study,
        'is_stochastic': is_stoch,
        'tiebreak_meta': tiebreak_meta,
        'n_trials': n_trials_eff,
    }

def tune_all_baselines(datadict, columns, names=('LR', 'RF', 'MLP', 'SVM', 'KNN'), k=3, n_seeds_per_fold=3, seed=9999, tiebreak='fold_std', tiebreak_tol=1e-6, storage_dir=None, verbose=True):
    out = {}
    for nm in names:
        if verbose: print(f"\n=== Tuning {nm} (grid: {_grid_size(nm)} combos) ===")
        storage = f'sqlite:///{storage_dir}/{nm}.db' if storage_dir else None
        out[nm] = tune_baseline(nm, datadict, columns, k=k, n_seeds_per_fold=n_seeds_per_fold, seed=seed, tiebreak=tiebreak, tiebreak_tol=tiebreak_tol, storage=storage, study_name=nm, verbose=verbose)
    return out
 
def summarize_tuning(tuned):
    for nm, res in tuned.items():
        meta = res.get('tiebreak_meta', {})
        n_tied = meta.get('n_tied', 1)
        n_tied = n_tied if n_tied > 1 else 0 # 1 is the default, it indicates no ties (one trial total at that value)
        n_trials = res.get('n_trials', '?')
        tied_tag = f" ({n_trials} trials, {n_tied} tied)"
        print(f"{nm}:\tbest_auc = {round(res['best_value'], 4)}{tied_tag}\tbest HP = {res['best_hp']}")