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

def run_ablation_stats(ablation_mtes, versions=None, n_permutations=1000, perm_seed=0):
    """ One-sided permutation tests of AUC > 0.5 for each modality variant, per split; Bonferroni corrections applied across variants within each split """
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
            auc, _, pval = permutation_test_auc_onesided(y, p, n_permutations=n_permutations, seed=perm_seed)
            raw_pvals.append(pval)
            rows.append({'version': v, 'auc': round(auc * 100, 2), 'p_raw': round(pval, 4)})
        _, p_bonf, _, _ = multipletests(raw_pvals, method='bonferroni')
        for i, row in enumerate(rows):
            row['p_bonf']   = round(float(p_bonf[i]), 4)
            row['sig_bonf'] = bool(p_bonf[i] < 0.05)
        perm_records[split] = rows
    perm_dfs = {}
    for split in splits:
        label = split_labels[split]
        perm_dfs[label] = pd.DataFrame(perm_records[split]).set_index('version').sort_values('auc', ascending=False)
    return perm_dfs