import pandas as pd
import numpy as np
import warnings
from collections import defaultdict, Counter
import re

import sklearn
from sklearn.feature_selection import SelectFromModel, SelectKBest, SelectPercentile, f_classif, mutual_info_classif, SelectFdr, SelectFpr, SelectFwe, VarianceThreshold
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler, PowerTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from scipy.stats import mannwhitneyu, chi2
from sklearn.metrics import roc_auc_score

from jp_utils import get_new_cols, drop_useless_cols, load_data, check_filename, write_file

def get_current_cols(X, refcols):
    present = [k for k in ['clin', 'expr', 'mut'] if k in refcols]
    ref_sets = {k: set(refcols[k]) for k in present}
    curcols = {k: [] for k in present}
    unknown = []
    for c in X.columns:
        assigned = False
        for k in present:
            if c in ref_sets[k]:
                curcols[k].append(c)
                assigned = True
                break
        if not assigned:
            unknown.append(c)
    if unknown:
        raise ValueError(f"{len(unknown)} column(s) in X not found in refcols: {unknown[:5]}{'…' if len(unknown)>5 else ''}")
    return {k: sorted(v) for k, v in curcols.items()}

class RemoveNearlyConstant(BaseEstimator, TransformerMixin):
    def __init__(self, refcols, t=1e-8):
        self.refcols, self.t = refcols, t
    def fit(self, X, y=None):
        cols = get_current_cols(X, self.refcols)
        self.vt = VarianceThreshold(threshold=self.t).set_output(transform="pandas")
        self.vt.fit(X)
        kept = list(X.columns[self.vt.get_support()])
        out_str = 'RemoveNearlyConstant - features removed:'
        for k in cols.keys():
            out_str += f" {len(cols[k])-len(get_new_cols(kept, cols[k]))} {k} "
        print(out_str)
        return self
    def transform(self, X):
        return self.vt.transform(X)

class RemoveRarelyMutated(BaseEstimator, TransformerMixin):
    def __init__(self, refcols, f=0.01):
        self.refcols, self.f = refcols, f # refcols is the original column dictionary (luad.cols)
    def fit(self, X, y=None):
        cols = get_current_cols(X, self.refcols)
        temp = cols['mut']
        mutation_freq = (X[cols['mut']] > 0).mean()
        cols['mut'] = mutation_freq[mutation_freq >= self.f].index.tolist()
        self.all_cols_ = []
        for k in cols.keys():
            self.all_cols_.extend(cols[k])
        print(f"RemoveRarelyMutated - {len(temp)-len(cols['mut'])} mut removed (mutated in less than {round(self.f*100)}% of tumors), {len(cols['mut'])} mut remaining")
        return self
    def transform(self, X):
        return X[self.all_cols_].copy()

class RemoveLowExpression(BaseEstimator, TransformerMixin): # drop very low-variance/low-expression genes to provide more stable data
    def __init__(self, refcols, min_expr=1.0, min_frac=0.2, min_var=0.01):
        """ min_expr = Minimum log2 expr considered 'on'; min_frac = Fraction of samples that must exceed min_expr; min_var = Minimum variance required across samples """
        self.refcols, self.min_expr, self.min_frac, self.min_var = refcols, min_expr, min_frac, min_var
    def fit(self, X, y=None):
        cols = get_current_cols(X, self.refcols)
        expr_X = X[cols['expr']]
        n = expr_X.shape[0]
        mask_expr = (expr_X > self.min_expr).sum(axis=0) >= (self.min_frac * n)
        mask_var = expr_X.var(axis=0) >= self.min_var
        cols['expr'] = expr_X.columns[mask_expr & mask_var].tolist()
        self.all_cols_ = []
        for k in cols.keys():
            self.all_cols_.extend(cols[k])
        print(f"RemoveLowExpression - {len(expr_X.columns)-len(cols['expr'])} expr removed ({len(cols['expr'])} expr remaining)")
        return self
    def transform(self, X):
        return X[self.all_cols_].copy()

class CrossDatasetExpressionSurvival(BaseEstimator, TransformerMixin): # not to be used in a pipeline, since it takes two X's and no Y
    def __init__(self, refcols, min_var_target=1e-3, max_abs_zshift=3.0):
        """ Drop genes with near-zero variance in target + Drop genes with extreme mean shift without ever seeing target labels """
        self.refcols = refcols
        self.min_var_target = min_var_target
        self.max_abs_zshift = max_abs_zshift
    def fit(self, X_source, X_target): 
        cols = get_current_cols(X_source, self.refcols)
        s = X_source[cols['expr']].copy()
        t = X_target[cols['expr']].copy()
        mu_s = s.mean(axis=0)
        sd_s = s.std(axis=0, ddof=1).replace(0, np.nan)
        mu_t = t.mean(axis=0)
        var_t = t.var(axis=0, ddof=1)
        zshift = ((mu_t - mu_s) / sd_s).abs().fillna(np.inf)
        keep_mask = (var_t >= float(self.min_var_target)) & (zshift <= float(self.max_abs_zshift))
        kept = [g for g in cols['expr'] if bool(keep_mask.loc[g])]
        cols['expr'] = kept
        self.all_cols_ = []
        for k in cols.keys():
            self.all_cols_.extend(cols[k])
        print(f"CrossDatasetExpressionSurvival - {len(s.columns)-len(cols['expr'])} expr removed ({len(cols['expr'])} expr remaining)")
        return self
    def transform(self, X):
        return X[self.all_cols_].copy()

class MiniCorrelationAnalysis(BaseEstimator, TransformerMixin):
    """ Identify and drop highly correlated features in two stages: (1) Prioritize known genes and remove their correlated partners. (2) For remaining pairs, drop the 
    feature with the lower primary metric (or use secondary metric on ties) """
    def __init__(self, refcols, hgnc, data_path='data/', threshold=0.95, primary_method='variance', secondary_method='corr count', prioritize_known_genes=True, corr_metric_expr='pearson', corr_metric_mut='phi', min_mut_positives=5):
        self.refcols = refcols
        self.hgnc = hgnc
        self.data_path = data_path
        self.threshold = threshold
        self.primary_method, self.secondary_method = primary_method, secondary_method
        self.prioritize_known_genes = prioritize_known_genes
        self.corr_metric_expr, self.corr_metric_mut, self.min_mut_positives = corr_metric_expr, corr_metric_mut, min_mut_positives
        self._get_known_genes_from_literature()
    def _get_known_genes_from_literature(self):
        known_dict = load_data('json', 'known_genes.json', self.data_path)
        given_symbols = set(g.upper() for source in known_dict.values() for g in source['genes'])
        self.known_genes = set()
        for sym in given_symbols:
            apprvd = self.hgnc['Symbol'].get(sym)
            if apprvd is not None: self.known_genes.add(apprvd)
        escaped = map(re.escape, sorted(self.known_genes))
        self.known_gene_pattern = fr"^(?:{'|'.join(escaped)})_"
    def _count_outliers(self, arr):
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        return np.sum((arr < (q1 - 1.5*iqr)) | (arr > (q3 + 1.5*iqr)))
    def _upper_triangle_pairs(self, corr_mx, features):
        iu = np.triu_indices(len(features), k=1)
        corr_vals = corr_mx[iu]
        mask = corr_vals >= self.threshold
        var1 = np.array(features)[iu[0][mask]]
        var2 = np.array(features)[iu[1][mask]]
        return var1, var2, corr_vals[mask]
    def _corr_pairs_expr(self, X, expr_cols):
        corr_mx = X[expr_cols].corr(method=self.corr_metric_expr).abs().values
        v1, v2, cv = self._upper_triangle_pairs(corr_mx, expr_cols)
        return pd.DataFrame({'var1': v1, 'var2': v2, 'correlation': cv, 'modality': 'expr'})
    def _corr_pairs_mut_phi(self, X, mut_cols): # for binary mutation matrix, Pearson == Phi coefficient
        B = X[mut_cols].astype(int)
        pos = B.sum(axis=0)
        eligible = pos.index[pos >= self.min_mut_positives].tolist() # filter to genes with sufficient positives
        if len(eligible) < 2:
            skipped = len(mut_cols) - len(eligible)
            if skipped > 0: print(f"[MiniCorr] φ: skipped {skipped} mut cols (<{self.min_mut_positives} positives)")
            return pd.DataFrame(columns=['var1','var2','correlation','modality'])
        corr_mx = B[eligible].corr(method='pearson').abs().values  # phi on binary
        v1, v2, cv = self._upper_triangle_pairs(corr_mx, eligible)
        return pd.DataFrame({'var1': v1, 'var2': v2, 'correlation': cv, 'modality': 'mut'})
    def _corr_pairs_mut_jaccard(self, X, mut_cols):
        B = X[mut_cols].astype(bool).to_numpy(copy=False)
        pos = B.sum(axis=0)
        elig_idx = np.where(pos >= self.min_mut_positives)[0]
        if elig_idx.size < 2: return pd.DataFrame(columns=['var1','var2','correlation','modality'])
        A = B[:, elig_idx].astype(np.uint8)
        names = [mut_cols[i] for i in elig_idx]
        inter = A.T @ A # intersections: A^T A (on 0/1); shape (p, p)
        pc = A.sum(axis=0)[:, None] # unions: |Ai| + |Aj| - inter; (p,1)
        union = pc + pc.T - inter
        with np.errstate(divide='ignore', invalid='ignore'):
            J = np.where(union > 0, inter / union, 0.0)
        iu = np.triu_indices_from(J, k=1)
        keep = J[iu] >= self.threshold
        v1 = np.array(names)[iu[0][keep]]
        v2 = np.array(names)[iu[1][keep]]
        vals = J[iu][keep]
        return pd.DataFrame({'var1': v1, 'var2': v2, 'correlation': vals, 'modality': 'mut'})
    def _to_series(self, metric, features):
        if isinstance(metric, pd.Series): return metric.reindex(features)
        else: return pd.Series(metric).reindex(features)  # dict-like
    def fit(self, X, y=None):
        self.remove = {}
        self.cols = get_current_cols(X, self.refcols)
        feats_all = self.cols['expr'] + self.cols['mut']
        self.metrics = {'variance': X[feats_all].var(), 'outliers': {f: self._count_outliers(X[f].values) for f in feats_all}}
        # build within-modality correlated-pair tables
        pairs_expr = self._corr_pairs_expr(X, self.cols['expr'])
        pairs_mut = self._corr_pairs_mut_jaccard(X, self.cols['mut']) if self.corr_metric_mut == 'jaccard' else self._corr_pairs_mut_phi(X, self.cols['mut'])
        self.corr_pairs = pd.concat([pairs_expr, pairs_mut], axis=0, ignore_index=True) # combine (still kept separate by 'modality' tag)
        # corr count per feature (only among pairs that exceeded threshold)
        if not self.corr_pairs.empty:
            keys, counts = np.unique(np.concatenate([self.corr_pairs['var1'].values, self.corr_pairs['var2'].values]), return_counts=True)
            self.metrics['corr count'] = dict(zip(keys, counts))
        else:
            self.metrics['corr count'] = {}
        self.mP = self._to_series(self.metrics[self.primary_method], feats_all)
        self.mS = self._to_series(self.metrics[self.secondary_method], feats_all)
        self._step1() # Step 1: known-gene priority (applied per modality implicitly via pairs table)
        self.mP, self.mS = [self.metrics[m] for m in [self.primary_method, self.secondary_method]] # Prepare primary / secondary metrics
        self._step2() # Step 2: resolve remaining pairs
        self._step3() # Step 3: update cols per modality
        self.all_cols_ = self.cols['clin'] + self.cols['expr'] + self.cols['mut']
        return self
    def _step1(self):
        if self.corr_pairs.empty:
            self.remove['step 1'] = {'expr': [], 'mut': []}
            return
        if self.prioritize_known_genes:
            mk1 = self.corr_pairs['var1'].str.match(self.known_gene_pattern, case=False, na=False)
            mk2 = self.corr_pairs['var2'].str.match(self.known_gene_pattern, case=False, na=False)
            drop_only_var2 = self.corr_pairs.loc[ mk1 & ~mk2, 'var2']  # keep known, drop the other
            drop_only_var1 = self.corr_pairs.loc[~mk1 &  mk2, 'var1']
            to_remove = list(set(drop_only_var1) | set(drop_only_var2))
            if to_remove: # filter pairs that include removed features
                mask = ~self.corr_pairs['var1'].isin(to_remove) & ~self.corr_pairs['var2'].isin(to_remove)
                self.corr_pairs = self.corr_pairs[mask]
            self.remove['step 1'] = {'expr': get_new_cols(to_remove, self.cols['expr']), 'mut' : get_new_cols(to_remove, self.cols['mut'])}
        else:
            self.remove['step 1'] = {'expr': [], 'mut': []}
    def _step2(self):
        to_remove = set()
        for f1, f2 in zip(self.corr_pairs['var1'], self.corr_pairs['var2']): # iterate pairs; they are within-modality already
            if f1 in to_remove or f2 in to_remove:
                continue
            drop = self._select_drop_feature(f1, f2)
            to_remove.add(drop)
        self.remove['step 2'] = {'expr': get_new_cols(to_remove, self.cols['expr']), 'mut' : get_new_cols(to_remove, self.cols['mut'])}
    def _step3(self):
        rem_expr = set(self.remove['step 1']['expr']) | set(self.remove['step 2']['expr'])
        rem_mut = set(self.remove['step 1']['mut']) | set(self.remove['step 2']['mut'])
        self.cols['expr'] = [c for c in self.cols['expr'] if c not in rem_expr]
        self.cols['mut'] = [c for c in self.cols['mut']  if c not in rem_mut]
        print(f"MiniCorrelationAnalysis (within-modality) - removed: {len(rem_expr)} expr ({len(self.cols['expr'])} remain), {len(rem_mut)} mut ({len(self.cols['mut'])} remain)")
    def _select_drop_feature(self, f1, f2):
        def compare_metrics(m1, m2, mthd):
            if mthd == 'variance': return f1 if m1 < m2 else f2
            else: return f1 if m1 > m2 else f2  # 'corr count' (larger is worse)
        mP1, mP2 = self.mP[f1], self.mP[f2]
        if mP1 != mP2:
            return compare_metrics(mP1, mP2, self.primary_method)
        mS1, mS2 = self.mS[f1], self.mS[f2]
        return compare_metrics(mS1, mS2, self.secondary_method)
    def transform(self, X):
        return X[self.all_cols_].copy()

class FilterVariance(BaseEstimator, TransformerMixin):
    def __init__(self, refcols, coltype, percentile=25):
        self.refcols = refcols
        self.coltype = coltype
        self.percentile = percentile
    def fit(self, X, y=None):
        cols = get_current_cols(X, self.refcols)
        Xs = X[cols[self.coltype]]
        vars_ = Xs.var(axis=0)
        vt = VarianceThreshold(threshold=np.percentile(vars_, self.percentile)).fit(Xs)  # tune threshold by inspecting var distribution
        cols[self.coltype] = sorted(Xs.columns[vt.get_support()].tolist())
        self.all_cols_ = []
        for k in cols.keys():
            self.all_cols_.extend(cols[k])
        print(f"FilterVariance - {len(Xs.columns)-len(cols[self.coltype])} {self.coltype} removed, {len(cols[self.coltype])} {self.coltype} remaining")
        return self
    def transform(self, X):
        check_is_fitted(self, attributes=['all_cols_'])
        return X[self.all_cols_].copy()

class WeightedUnivar(BaseEstimator, TransformerMixin):
    """ Univariate ANOVA (f_classif) on a target column group with rank-based prior for known genes and optional stability selection via bootstraps """
    def __init__(self, refcols, coltype, known_genes=(), score='f', ensemble_weights=None, k=100, known_rank_bonus=0, stability_rounds=0, stability_frac=0.7, random_state=None):
        self.refcols = refcols
        self.coltype = coltype
        self.known_genes = tuple(known_genes)
        self.score = score
        assert self.score in ['f', 'chi2', 'mi', 'auc', 'snr', 'mw', 'ensemble'], f'Invalid scoring mode entered: {self.score}'
        self.ensemble_weights = ensemble_weights
        self.k = int(k)
        self.known_rank_bonus = int(known_rank_bonus)
        self.stability_rounds = int(stability_rounds)
        self.stability_frac = float(stability_frac)
        self.random_state = random_state
    
    def _score_f(self, Xg, y):
        F, _ = f_classif(Xg.values, y.values if isinstance(y, pd.Series) else y)
        return pd.Series(np.clip(F, 0, None), index=Xg.columns) # guard: negative F occasionally from numerical issues -> clip at 0
    
    def _score_chi2(self, Xg, y):
        chi, _ = chi2(Xg, y.values if isinstance(y, pd.Series) else y)
        return pd.Series(np.nan_to_num(chi, nan=0.0), index=Xg.columns)

    def _score_mi(self, Xg, y):
        discf = True if self.coltype == 'mut' else False
        mi = mutual_info_classif(Xg.values, y.values if isinstance(y, pd.Series) else y, discrete_features=discf, random_state=self.random_state)
        return pd.Series(mi, index=Xg.columns)
    
    def _score_snr(self, Xg, y):
        yv = y.values if isinstance(y, pd.Series) else y
        x0 = Xg[yv==0]; x1 = Xg[yv==1]
        snr = (x1.mean() - x0.mean()) / (x1.std(ddof=1) + x0.std(ddof=1) + 1e-8)
        return snr.abs()
    
    def _score_mw(self, Xg, y):
        yv = y.values if isinstance(y, pd.Series) else y
        def mw(col):
            a, b = col[yv==0].values, col[yv==1].values
            if a.size<2 or b.size<2 or np.allclose(col, col.iloc[0]): return 0.0
            u, _ = mannwhitneyu(a, b, alternative='two-sided')
            m, n = len(a), len(b)
            return max(u, m*n - u) / (m*n) # normalize U to [0,1] by dividing by max possible
        return Xg.apply(mw, axis=0)
    
    def _score_auc(self, Xg, y):
        yv = y.values if isinstance(y, pd.Series) else y
        def au1(col):
            x = col.values
            if np.allclose(x, x[0]): return 0.5
            try:
                a = roc_auc_score(yv, x)
                return max(a, 1-a)  # direction-invariant
            except ValueError: return 0.5
        return Xg.apply(au1, axis=0)
    
    def _score_once(self, Xg, y):
        if self.score == 'f': return self._score_f(Xg, y), False
        if self.score == 'chi2': return self._score_chi2(Xg, y), False
        if self.score == 'mi': return self._score_mi(Xg, y), False
        if self.score == 'auc': return self._score_auc(Xg, y), False
        if self.score == 'snr': return self._score_snr(Xg, y), False
        if self.score == 'mw': return self._score_mw(Xg, y), False
        if self.score == 'ensemble':
            parts = {'f': self._score_f(Xg, y), 'auc': self._score_auc(Xg, y), 'snr': self._score_snr(Xg, y), 'mw': self._score_mw(Xg, y),}
            weights = self.ensemble_weights or {k:1.0 for k in parts} # rank-aggregate (lower rank = better)
            wrank = sum((weights.get(k, 0.0) * s.rank(ascending=False, method='average')) for k, s in parts.items() if weights.get(k, 0.0) > 0)
            wrank /= sum(w for w in weights.values() if w > 0)
            return wrank, True
    
    def _tie_break(self, scores, Xg, rng):
        # Deterministic tie-breaking: 1) If all equal, add an epsilon scaled by prevalence (column sums). 2) If prevalence is all zeros/equal, add tiny seeded jitter.
        if scores.nunique() > 1:
            return scores
        if self.coltype == 'expr':
            prev = Xg.var(axis=0).astype(float)
        else:
            prev = Xg.sum(axis=0).astype(float)
        if prev.max() > 0:
            eps = (prev / (prev.max() + 1e-12)) * 1e-6
            return scores + eps
        jitter = pd.Series(rng.normal(loc=0.0, scale=1e-6, size=len(scores)), index=scores.index) # final fallback: tiny seeded jitter
        return scores + jitter
    
    def _bootstrap_idx(self, y, m, rng): # Stratified, with-replacement bootstrap indices (keeps class mix and avoids single-class draws)
        yv = y.values if isinstance(y, pd.Series) else y
        n = len(yv)
        i1 = np.where(yv == 1)[0]
        i0 = np.where(yv == 0)[0]
        if i1.size == 0 or i0.size == 0: # Fallback: if a class is missing, just sample with replacement from all
            return rng.choice(n, size=m, replace=True)
        p1 = i1.size / n # Allocate per-class counts proportional to prevalence
        m1 = max(1, int(round(m * p1)))
        m0 = max(1, m - m1)
        b1 = rng.choice(i1, size=m1, replace=True)
        b0 = rng.choice(i0, size=m0, replace=True)
        idx = np.concatenate([b0, b1])
        rng.shuffle(idx)
        return idx

    def _stability(self, Xg, y): # returns a score (higher = better)
        if self.stability_rounds <= 0:
            vals, is_rank = self._score_once(Xg, y)
            if is_rank:
                max_rank = float(len(vals))
                return (max_rank + 1.0) - vals # convert rank→score once
            return vals  # already a score
        rng = check_random_state(self.random_state)
        n = len(y)
        m = max(2, int(np.floor(self.stability_frac * n)))
        agg_rank = pd.Series(0.0, index=Xg.columns) # Aggregate ranks across bootstraps (lower = better)
        for _ in range(self.stability_rounds):
            idx = self._bootstrap_idx(y, m, rng)  # stratified, with-replacement
            # idx = rng.choice(n, size=m, replace=False)
            Xb = Xg.iloc[idx]
            yb = y.iloc[idx] if isinstance(y, pd.Series) else y[idx]
            vals, is_rank = self._score_once(Xb, yb)
            rb = vals if is_rank else vals.rank(ascending=False, method='average')
            agg_rank += rb
        agg_rank /= float(self.stability_rounds)
        max_rank = float(len(agg_rank))
        return (max_rank + 1.0) - agg_rank  # single final rank→score

    def _apply_known_bonus(self, scores):
        if self.known_rank_bonus <= 0:
            return scores
        s = scores
        ranks = s.rank(ascending=False, method='average')
        kg = ranks.index.intersection(pd.Index(self.known_genes))
        if len(kg):
            r = ranks.copy()
            r.loc[kg] = np.maximum(1.0, r.loc[kg] - float(self.known_rank_bonus))
            max_rank = float(len(r))
            s = (max_rank + 1.0) - r  # back to score space
        return s

    def fit(self, X, y):
        cols = get_current_cols(X, self.refcols)
        Xg = X[cols[self.coltype]]
        scores = self._stability(Xg, y)
        scores = self._apply_known_bonus(scores)
        rng = check_random_state(self.random_state) # new
        scores = self._tie_break(scores, Xg, rng) # new
        self.scores = scores.sort_values(ascending=False)
        ranked = self.scores.index.tolist()
        selected = ranked[:self.k]
        cols[self.coltype] = sorted(selected)
        self.all_cols_ = []
        for k in cols.keys():
            self.all_cols_.extend(cols[k])
        print(f"WeightedUnivar - {len(Xg.columns)-len(cols[self.coltype])} {self.coltype} removed, {len(cols[self.coltype])} {self.coltype} remaining")
        return self

    def transform(self, X):
        check_is_fitted(self, attributes=['all_cols_'])
        return X[self.all_cols_].copy()

class SelectFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, refcols, coltype, clf, top_k=None, boost_known=None, boost_factor=2.0, **kwargs):
        self.refcols, self.coltype, self.clf, self.top_k, self.boost_factor = refcols, coltype, clf, top_k, boost_factor
        self.boost_known = set(boost_known or [])
        self.kwargs = kwargs
    def fit(self, X, y):
        cols = get_current_cols(X, self.refcols)
        Xblock = X[cols[self.coltype]]
        est = self.clf
        est.fit(Xblock, y)
        if hasattr(est, "feature_importances_"):
            imp = pd.Series(est.feature_importances_, index=Xblock.columns, dtype=float)
        elif hasattr(est, "coef_"):
            imp = pd.Series(np.abs(est.coef_).ravel(), index=Xblock.columns, dtype=float)
        else: raise ValueError("Estimator lacks feature importances/coefficients.")
        if self.coltype == 'mut' and self.boost_known: # apply boosting to mutation features that are in the known list
            mask = imp.index.isin(self.boost_known)
            imp.loc[mask] *= float(self.boost_factor)
        # selection: top_k if provided, else fallback to SelectFromModel threshold
        if self.top_k is not None:
            keep = imp.sort_values(ascending=False).head(self.top_k).index.tolist()
        else:
            sel = SelectFromModel(estimator=est, **self.kwargs).fit(Xblock, y)
            keep = [Xblock.columns[i] for i in sel.get_support(indices=True)]
        cols[self.coltype] = keep
        self.all_cols_ = []
        for k in cols.keys():
            self.all_cols_.extend(cols[k])
        print(f"SelectFeatures - {self.coltype} features selected: {len(cols[self.coltype])}")
        return self
    def transform(self, X):
        return X[self.all_cols_].copy()