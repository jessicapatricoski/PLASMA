from statsmodels.stats.multitest import multipletests
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def permutation_test_auc_onesided(y_true, y_prob, n_permutations=1000, seed=0):
    """ One-sided label-permutation test for AUC vs chance. The null distribution is constructed by permuting the labels; the p-value is the proportion of permuted AUCs 
    at least as large as the observed AUC. H0: AUC <= 0.5 (model is no better than chance); H1: AUC > 0.5 (model is genuinely predictive) """
    observed_auc = roc_auc_score(y_true, y_prob)
    rng = np.random.RandomState(seed)
    null_aucs = np.array([roc_auc_score(rng.permutation(y_true), y_prob) for _ in range(n_permutations)])
    p_onesided = (null_aucs >= observed_auc).mean()
    return observed_auc, null_aucs, p_onesided

def bootstrap_auc_ci(y_true, y_prob, n_boot=10000, seed=0, ci=95):
    """ Bootstrap percentile CI for a single AUROC. Resamples patient indices with replacement; recomputes AUC on each bootstrap sample. Returns the observed AUC 
    plus the (1-alpha)/2 and (1+alpha)/2 percentile bounds. Single-class bootstrap draws (where AUC is undef) are dropped """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    obs_auc = roc_auc_score(y_true, y_prob)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    alpha = (100 - ci) / 2
    aucs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2: continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    aucs = np.asarray(aucs, dtype=float)
    return {'auc': float(obs_auc), 'ci_low': float(np.percentile(aucs, alpha)), 'ci_high': float(np.percentile(aucs, 100 - alpha)), 'n_boot_valid': int(len(aucs))}

def run_ablation_stats(ablation_mtes, versions=None, n_permutations=1000, perm_seed=0, n_boot=10000, boot_seed=0, ci=95):
    """ One-sided permutation tests of AUC > 0.5 for each modality variant, per split, paired with bootstrap CIs on AUROC and other reporting columns """
    if versions is None:
        versions = list(ablation_mtes.keys())
    splits = ['tcga_test', 'tracerx']
    split_labels = {'tcga_test': 'TCGA Test', 'tracerx': 'TracerX'}
    perm_records = {split: [] for split in splits}
    for split in splits:
        raw_pvals, rows = [], []
        for v in versions:
            mte = ablation_mtes[v]
            y = mte['ensemble'][split]['targets'].astype(int)
            p = mte['ensemble'][split]['probs_cal']
            # permutation p-value (test statistic = AUROC; reported as p_raw)
            auc, _, pval = permutation_test_auc_onesided(y, p, n_permutations=n_permutations, seed=perm_seed)
            # bootstrap CI on AUROC (independent resampling procedure)
            boot = bootstrap_auc_ci(y, p, n_boot=n_boot, seed=boot_seed, ci=ci)
            raw_pvals.append(pval)
            rows.append({
                'version': v,
                'n': int(len(y)), # cohort size (df proxy for nonparametric test)
                'auc': round(auc * 100, 2), # test statistic
                'auc_ci_low': round(boot['ci_low']  * 100, 2),
                'auc_ci_high': round(boot['ci_high'] * 100, 2),
                'effect_size': round((auc - 0.5) * 100, 2), # deviation from chance baseline, in pp
                'p_raw': round(pval, 4)
            })
        _, p_bonf, _, _ = multipletests(raw_pvals, method='bonferroni')
        for i, row in enumerate(rows):
            row['p_bonf'] = round(float(p_bonf[i]), 4)
            row['sig_bonf'] = bool(p_bonf[i] < 0.05)
        perm_records[split] = rows
    perm_dfs = {}
    for split in splits:
        label = split_labels[split]
        perm_dfs[label] = pd.DataFrame(perm_records[split]).set_index('version').sort_values('auc', ascending=False)
    return perm_dfs