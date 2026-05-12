import pandas as pd
import numpy as np
import random
import re
from itertools import product, combinations, cycle
import itertools
import os
import pickle
from collections import defaultdict, Counter
import warnings
import json
import copy
import math
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from scipy.linalg import fractional_matrix_power
from IPython.display import display
from pandas.api.types import is_numeric_dtype
import importlib

import sklearn
from sklearn import metrics, naive_bayes, set_config
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, average_precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, roc_curve, balanced_accuracy_score
from sklearn.feature_selection import SelectFromModel, SelectKBest, SelectPercentile, f_classif, mutual_info_classif, SelectFdr, SelectFpr, SelectFwe, VarianceThreshold
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler, PowerTransformer, FunctionTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator, AutoMinorLocator
import matplotlib.transforms as mtransforms

import jp_utils; importlib.reload(jp_utils)
from jp_utils import get_new_cols, drop_useless_cols, load_data, check_filename, write_file, get_hgnc26, set_idx, limit_df

import custom_transformers; importlib.reload(custom_transformers)
from custom_transformers import get_current_cols, RemoveNearlyConstant, MiniCorrelationAnalysis, FilterVariance, WeightedUnivar, SelectFeatures, RemoveRarelyMutated, RemoveLowExpression, CrossDatasetExpressionSurvival

# https://xenabrowser.net/datapages/?cohort=TCGA%20Lung%20Adenocarcinoma%20(LUAD)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443 
# https://gdc.cancer.gov/about-data/publications/pancanatlas

data_path = '../data/'

def get_data(filename, dp, lower=False, clean=True, **kwargs):
    df = load_data('raw txt', filename, dp, clean=clean, comment='#', **kwargs)
    if lower: df.columns = [col.lower() for col in df.columns]
    return df

def column_agreement(df, col1, col2):
    no_nulls = df.loc[df[col1].notna()&df[col2].notna(), [col1, col2]]
    def normalize(s):
        return s.map(lambda x: x.casefold() if isinstance(x, str) else x)
    return normalize(no_nulls[col1]).equals(normalize(no_nulls[col2]))

def fill_from_backup_cols(df, target, backups, drop_backups=True):
    column_pairs = list(combinations([target] + backups, 2))
    for pair in column_pairs:
        assert column_agreement(df, pair[0], pair[1]), f"Column Disagreement between {pair[0]} and {pair[1]}"
    work = df[[target] + backups].copy()
    filled_target = work.bfill(axis=1).iloc[:, 0] # Take first non-null across [target] + backups (row-wise)
    df[target] = filled_target
    return df.drop(columns=backups) if drop_backups else df

class TCGA:
    def __init__(self, keep, paths, hgnc, predetermined_cutoff=None):
        self.keep, self.paths, self.hgnc = keep, paths, hgnc
        self.explicit_cases = set()
        self.lost_follow_up = set()
        self.id_tracker = {}
        print('---- TCGA ----')
        self.get_clinical()
        self.get_genomic()
        self.finalize_data(predetermined_cutoff)

    def update_id_tracker(self, key, ids):
        if key is None:
            if isinstance(ids, dict): 
                norm = {k: (list(v) if not isinstance(v, (list, set)) else v) for k, v in ids.items()}
                self.id_tracker.update(norm)
            else: print("Error in call to self.update_id_tracker; 'ids' must be a dictionary when 'key' is None")
        else: self.id_tracker[key] = list(ids)

    def print_id_tracker(self, d, path=(), final_cohort_only=False):
        if isinstance(d, dict):
            for k, v in d.items(): self.print_id_tracker(v, path + (k,))
        elif isinstance(d, list) or isinstance(d, set):
            if final_cohort_only: print(" -> ".join(path), ":", len([i for i in d if i in self.clin.index]))
            else: print(" -> ".join(path), ":", len(d)) 
    
    def remove_subset(self, ids, key):
        self.update_id_tracker(key, ids)
        return self.clin[~self.clin.index.isin(ids)].copy()
    
    def compile_clinical_data(self):
        drop_cols = [*[f'gender_{i}' for i in [1,5,6,7]], *[f'sex_{i}' for i in [2,3]], 'vital_status_6', 'ajcc_pathologic_t_6', 'path_t_stage_2', *[f'pathologic_t_{i}' for i in [1,7]], 'ajcc_tumor_pathologic_pt_3', 'tobacco_smoking_history_7', 'tobacco_smoking_history_indicator_7', 'histological_type_4', 'type_4'] # confirmed agreement
        self.clin = self.clin.drop(columns=drop_cols)
        ttr_cols = [*[f'days_to_additional_surgery_locoregional_procedure_{i}' for i in [1,7]], *[f'days_to_additional_surgery_metastatic_procedure_{i}' for i in [1,7]], *[f'days_to_nte_after_initial_treatment_{i}' for i in [1,5,7]], 'dfi.time_4', *[f'dfs_months_{i}' for i in [2,3]], 'nte_dx_days_to_4']
        self.clin[ttr_cols] = self.clin[ttr_cols].apply(pd.to_numeric, errors='coerce')
        self.clin['ajcc_pathologic_tumor_stage_2'] = self.clin['ajcc_pathologic_tumor_stage_2'].str.replace('STAGE', 'Stage')
        # Disagreement between days_to_nte_after_initial_treatment_5 and _1/_7 (1 and 7 are both xena), defaulting to the lower value (_5)
        self.clin.loc['TCGA-78-7143-01', ['days_to_nte_after_initial_treatment_1', 'days_to_nte_after_initial_treatment_7']] = np.nan
        backfill_groupings = {
            'race_4': [f'race_{i}' for i in [3,5,6,2]],
            'ethnicity_3': [f'ethnicity_{i}' for i in [2,5,6]],
            'initial_pathologic_dx_year_4': ['initial_pathologic_dx_year_3', *[f'year_of_initial_pathologic_diagnosis_{i}' for i in [5,1,7]], 'year_of_diagnosis_6'],
            'residual_tumor_3': [f'residual_tumor_{i}' for i in [5,1,7]],
            'pathologic_stage_5': [*[f'ajcc_pathologic_tumor_stage_{i}' for i in [4,3]], 'ajcc_pathologic_stage_6', *[f'pathologic_stage_{i}' for i in [1,7]], 'ajcc_pathologic_tumor_stage_2'],
            'pathologic_m_5': ['ajcc_metastasis_pathologic_pm_3', 'ajcc_pathologic_m_6', *[f'pathologic_m_{i}' for i in [1,7]], 'path_m_stage_2'],
            'pathologic_n_5': ['ajcc_nodes_pathologic_pn_3', *[f'pathologic_n_{i}' for i in [1,7]], 'ajcc_pathologic_n_6', 'path_n_stage_2'],
            'number_pack_years_smoked_5': [*[f'number_pack_years_smoked_{i}' for i in [1,7]], 'pack_years_smoked_6', 'smoking_pack_years_3'],
            'system_version_5': ['ajcc_staging_edition_3', *[f'system_version_{i}' for i in [1,7]], 'system_version_1', 'system_version_7', 'ajcc_staging_system_edition_6', 'ajcc_staging_edition_2'],
            'tobacco_smoking_history_indicator_3': ['tobacco_smoking_history_1'],
            'smoking_year_started_3': [f'year_of_tobacco_smoking_onset_{i}' for i in [1,5,7]],
            'smoking_year_stopped_3': [f'stopped_smoking_year_{i}' for i in [1,5,7]],
            'anatomic_neoplasm_subdivision_5': [*[f'anatomic_neoplasm_subdivision_{i}' for i in [1,7]], 'primary_site_patient_3'],
            'days_to_additional_surgery_locoregional_procedure_1': ['days_to_additional_surgery_locoregional_procedure_7'],
            'days_to_additional_surgery_metastatic_procedure_1': ['days_to_additional_surgery_metastatic_procedure_7'],
            'days_to_nte_after_initial_treatment_5': [f'days_to_nte_after_initial_treatment_{i}' for i in [1,7]],
        }
        for target, backups in backfill_groupings.items():
            self.clin = fill_from_backup_cols(self.clin, target, backups, drop_backups=True)
        def assign_smoking_status(val):
            if pd.isna(val): return np.nan
            v = str(val).strip().lower()
            if v.startswith('current reformed smoker'): return 'former'
            elif v == 'current smoker': return 'current'
            elif v == 'lifelong non-smoker': return 'never'
            else: raise ValueError(f'Unknown value: {v}')
        self.clin['smoking_status'] = self.clin['tobacco_smoking_history_5'].apply(assign_smoking_status)
        smk_cols = ['tobacco_smoking_history_5', 'tobacco_smoking_history_indicator_1', 'tobacco_smoking_history_indicator_3', 'cigarettes_per_day_6', 'number_pack_years_smoked_5', 'smoking_year_started_3', 'smoking_year_stopped_3', 'years_smoked_6']
        self.clin = self.remove_subset(self.clin[self.clin[['smoking_status']+smk_cols].isna().all(axis=1)].index.tolist(), 'Missing smoking status *')
        self.clin = self.clin.drop(columns=smk_cols)
        self.clin_raw = self.clin.copy()

    def get_clinical(self):
        self.xena_clin = self._load_and_prep_xena('luad_xena', '_1')
        self.pc_clin = self._load_and_prep_cbio('luad_pc', '_2')
        self.fh_clin = self._load_and_prep_cbio('luad_fh', '_3')
        self.cdr = self._load_and_prep_gdc('cdr', '_4')
        self.cfu = self._load_and_prep_gdc('followup', '_5')
        self.gdc_clin = self._load_and_prep_gdc('luad_gdc_phenotypes', '_6')
        self.combined_xena = self._load_and_prep_xena('combined_xena', '_7')
        self.clin = pd.concat([self.xena_clin, self.pc_clin, self.fh_clin, self.cdr, self.cfu, self.gdc_clin, self.combined_xena], axis=1, join='outer').dropna(axis=1, how='all')
        self.clin = self.clin[sorted(self.clin.columns)]
        self.timeline_dx = get_data('data_timeline_status.txt', self.paths['luad_pc'], lower=True)[['patient_id', 'start_date', 'status']]
        self.timeline_rx = get_data('data_timeline_treatment.txt', self.paths['luad_pc'], lower=True)[['patient_id', 'start_date', 'anatomic_treatment_site', 'regimen_indication']]
        self.update_id_tracker('Has primary clin', self.clin.index.tolist())
        for k, df in zip(['rx', 'dx'], [self.timeline_rx, self.timeline_dx]):
            self.update_id_tracker(f'Primary clin with {k} timeline', [i for i in self.clin.index.tolist() if i.replace('-01','') in df['patient_id'].tolist()])
        age_columns = [c for c in self.clin.columns if c.startswith(('age', 'birth', 'days_to_birth'))]
        self.clin = self.remove_subset(self.clin[self.clin[age_columns].isna().all(axis=1)].index.tolist(), 'Missing age *')
        age_columns.remove('age_at_initial_pathologic_diagnosis_4')
        self.clin = self.clin.drop(columns=age_columns)
        race_ethn_columns = [c for c in self.clin.columns if c.startswith(('race', 'ethnicity'))]
        self.clin[race_ethn_columns] = self.clin[race_ethn_columns].apply(lambda s: s.astype('string').str.strip().str.lower())
        self.compile_clinical_data()
        self._build_outcomes_table()
        self.clin = self.remove_subset(self.clin[self.clin['pathologic_stage_5']=='Stage IV'].index.tolist(), 'Stage IV *')
        self.clin = self.remove_subset(self.clin[self.clin['pathologic_stage_5']=='Stage IIIB'].index.tolist(), 'Stage IIIB *')
        self.clin = self.remove_subset(self.clin[(self.clin['system_version_5']!='7th')&(self.clin['pathologic_t_5']=='T4')].index.tolist(), 'Pre-7th edition with T4 *')
        self.clin.loc[self.clin['pathologic_stage_5']=='Stage II', 'pathologic_stage_5'] = 'Stage IIB' # the 1 patient with Stage II can be classified as IIb based on TNM

    def _load_and_prep_xena(self, key, suffix=''):
        df = get_data('phenotypes.txt', dp=self.paths[key], lower=True)
        df = df[self.keep[key]['phenotypes']].copy()
        if key == 'combined_xena':
            df = df.loc[df['_primary_disease']=='lung adenocarcinoma'].copy()
        df = set_idx(df.dropna(subset=['sampleid']), idx_col='sampleid')
        self.explicit_cases.update(list(df[(df['sample_type']=='Recurrent Tumor')|(df.index.str.endswith('-02'))].index.to_series().str[:-2] + '01'))
        df = df.loc[(df['sample_type']=='Primary Tumor')|(df.index.str.endswith('-01'))].copy()
        self.lost_follow_up.update(df.loc[df['lost_follow_up']=='YES'].index.tolist())
        df = df.drop(columns=['sample_type', 'lost_follow_up'])
        df.columns = [f"{col.replace('new_tumor_event', 'nte')}{suffix}" for col in df.columns]
        return df

    def _load_and_prep_cbio(self, key, suffix=''):
        patient = get_data('data_clinical_patient.txt', dp=self.paths[key], lower=True)
        sample = get_data('data_clinical_sample.txt', dp=self.paths[key], lower=True, clean=False)
        df = patient[self.keep[key]['data_clinical_patient']].merge(sample[self.keep[key]['data_clinical_sample']], on='patient_id')
        df = set_idx(df.dropna(subset=['sample_id']), idx_col='sample_id', also_drop=['patient_id'])
        if 'sample_type' in df.columns:
            self.explicit_cases.update(list(df[(df['sample_type']=='Recurrence')|(df.index.str.endswith('-02'))].index.to_series().str[:-2] + '01'))
            df = df.loc[(df['sample_type']=='Primary')|(df.index.str.endswith('-01'))].drop(columns=['sample_type']).copy()
        else:
            self.explicit_cases.update(df[df.index.str.endswith('-02')].index.tolist())
            df = df.loc[df.index.str.endswith('-01')].copy()
        df.columns = [f"{col.replace('new_tumor_event', 'nte')}{suffix}" for col in df.columns]
        return df
    
    def _load_and_prep_gdc(self, key, suffix=''):
        dp = self.paths['gdc']
        if key == 'luad_gdc_phenotypes':
            df = load_data('tsv', key, data_path=dp)
            df.columns = [col.lower() for col in df.columns]
            df = df[self.keep['gdc'][key]].copy()
            df['sample_id'] = df['sample'].str[:-1] # all sample_ids have an additional letter at the end (A, B, C, etc.) that are unrelated
            df = df.dropna(subset=['sample_id'])
            self.explicit_cases.update(list(df[(df['sample_type.samples']=='Recurrent Tumor')|(df['tumor_descriptor.samples']=='Recurrence')|(df['sample_id'].str.endswith('-02'))]['sample_id'].str[:-2] + '01'))
            df = df[(df['sample_type.samples']=='Primary Tumor')|(df['tumor_descriptor.samples']=='Primary')|(df['sample_id'].str.endswith('-01'))].copy()
            df = set_idx(df.drop(columns = ['sample', 'sample_type.samples', 'tumor_descriptor.samples']).drop_duplicates(), idx_col='sample_id')
            df.columns = [col.split('.')[0] for col in df.columns]
        elif key in ['cdr', 'followup']: # these files are by patient, not sample (static to the patient), so we can tack on -01
            if key == 'cdr': 
                df = load_data('csv', 'tcga_cdr', data_path=dp, clean=True, index_col=0)
                df = df.loc[df['type']=='LUAD'].copy()
            else: 
                df = load_data('tsv', 'clinical_PANCAN_patient_with_followup', data_path=dp, clean=True, encoding='latin1', low_memory=False)
                df = df.loc[df['acronym']=='LUAD'].copy()
            df.columns = [col.lower() for col in df.columns]
            df = df[self.keep['gdc'][key]].copy()
            df['sample_id'] = df['bcr_patient_barcode'] + '-01'
            df = set_idx(df, idx_col='sample_id', also_drop=['bcr_patient_barcode'])
        df.columns = [f"{col.replace('new_tumor_event', 'nte')}{suffix}" for col in df.columns]
        return df

    def get_genomic(self):
        dp = self.paths['luad_pc']
        fns = {'expr': 'data_mrna_seq_v2_rsem.txt', 'mut': 'data_mutations.txt'}
        expr_gene_col, mut_gene_col, mut_id_col, mut_chr_col, mut_varclass_col = 'Hugo_Symbol', 'Hugo_Symbol', 'Tumor_Sample_Barcode', 'Chromosome', 'Variant_Classification'
        self.expr = self._process_expression_data(fns['expr'], dp, gene_col=expr_gene_col)
        self.mut = self._process_mutation_data(fns['mut'], dp, gene_col=mut_gene_col, id_col=mut_id_col, chr_str=mut_chr_col, varclass_str=mut_varclass_col)
        self.expr, self.mut = [df[df.index.str.endswith('-01')].copy() for df in [self.expr, self.mut]] # limit to primary samples only
        print(f"Expression shape: {self.expr.shape}, Mutation shape: {self.mut.shape}")
        
    def _process_expression_data(self, filename, dp, gene_col): 
        expr_df = get_data(filename, dp=dp)
        expr_df = self._update_gene_symbols(expr_df, gene_col)
        expr_df = expr_df.groupby('approved_symbol').sum()
        expr_df.index = list(expr_df.index) # removes "approved symbol" label
        expr_df = expr_df.T
        expr_df = expr_df.loc[:, (expr_df > 0).any(axis=0)] # drop all-zero columns
        return expr_df.add_suffix('_expr')

    def _process_mutation_data(self, filename, dp, gene_col, id_col, chr_str='Chromosome', varclass_str='Variant_Classification'):
        df = get_data(filename, dp=dp, low_memory=False)
        df = df[(~df[chr_str].isin(['X','Y'])) & (~df[varclass_str].isin(['Silent', 'RNA']))].copy()
        df = self._update_gene_symbols(df, gene_col)
        self.raw_mut = df.copy() # make sure to update this later
        mut_df = pd.crosstab(df[id_col], df['approved_symbol']).astype(float)
        mut_df.columns = [f"{col}_mut" for col in mut_df.columns]
        return mut_df

    def _update_gene_symbols(self, df, gene_col, drop_unmapped=True):
        df = df.copy()
        def _mapper_dict(key):
            d = self.hgnc.get(key)
            return {str(k).strip(): v for k, v in d.items()} # normalize keys to string and strip spaces
        def _norm_series(s):
            return s.astype('string').str.strip()
        approved_map = _mapper_dict('Symbol')
        df['approved_symbol'] = _norm_series(df[gene_col]).str.upper().map(approved_map)
        df['approved_symbol'] = df['approved_symbol'].infer_objects(copy=False)
        if drop_unmapped:
            df = df.dropna(subset=['approved_symbol'])
        return df.drop(columns=[gene_col])
    
    def analyze_data_availability(self, update_tracker=True):
        has_clin, has_expr, has_mut = self.clin.index, self.expr.index, self.mut.index
        expr_in_clin, mut_in_clin = has_expr.intersection(has_clin), has_mut.intersection(has_clin) # Expr and mut need clinical otherwise there's no recurrence label
        all3 = expr_in_clin.intersection(mut_in_clin) # clin + expr + mut
        expr_no_mut = expr_in_clin.difference(all3) # clin + expr only
        mut_no_expr = mut_in_clin.difference(all3) # clin + mut only
        clin_only = has_clin.difference(expr_in_clin.union(mut_in_clin)) # clin only (no expr/mut)
        if update_tracker: 
            self.update_id_tracker(None, {'Has all 3 types': all3, 'Missing expr only': expr_no_mut, 'Missing mut only': mut_no_expr, 'Missing expr AND mut': clin_only})
        print(f"All have clinical (otherwise no label):   Expr + Mut = {len(all3)}   Expr only = {len(expr_no_mut)}   Mut only = {len(mut_no_expr)}   Clin only = {len(clin_only)}")
        return {'all3': all3.tolist(), 'expr_only': expr_no_mut.tolist(), 'mut_only': mut_no_expr.tolist(), 'clin_only': clin_only.tolist()}

    def get_cohort_selection(self, include_expr=True, include_mut=True):
        idxs = self.clin.index # need clinical otherwise there's no recurrence label
        if include_expr: idxs = idxs.intersection(self.expr.index)
        if include_mut: idxs = idxs.intersection(self.mut.index)
        return idxs.tolist()

    def finalize_data(self, predetermined_cutoff):
        self.clin = self.clin.rename(columns={'age_at_initial_pathologic_diagnosis_4': 'age', 'gender_4': 'sex', 'pathologic_stage_5': 'stage', 'race_4': 'race', 'ethnicity_3': 'ethnicity'})
        self.clin = self.clin[['age', 'sex', 'stage', 'smoking_status', 'race', 'ethnicity', 'ttr']]
        self.analyze_data_availability()
        self.cohort = sorted(self.get_cohort_selection(include_expr=True, include_mut=True))
        for name in ['clin', 'expr', 'mut']:
            df = getattr(self, name)
            df = df.loc[self.cohort].copy()
            if name != 'clin': 
                assert not df.isna().values.any(), f"NaNs detected in {name}"
            df = df.loc[:, df.nunique() > 1].copy()
            setattr(self, name, df.reindex(self.cohort))
        self.raw_mut = self.raw_mut[self.raw_mut['Tumor_Sample_Barcode'].isin(self.cohort)].copy()
        self.clin_raw = self.clin_raw.loc[self.cohort].copy()
        if predetermined_cutoff is None:
            self.cutoff = self.clin['ttr'].median()
            print(f"Cutoff used: Median TTR ({self.cutoff})")
            self.clin['label'] = (self.clin['ttr'] < self.cutoff).astype(int)
        else:
            print(f"Cutoff used: Predetermined ({predetermined_cutoff})")
            self.clin['label'] = (self.clin['ttr'] < predetermined_cutoff).astype(int)
        early, late = [self.clin[self.clin['label']==i].index.tolist() for i in [1, 0]]
        print(f"{len(self.clin)} patients ({len(self.clin)-self.clin['label'].sum()} late, {self.clin['label'].sum()} early)")
        self.clin_for_table1 = self.clin.copy()
        self.clin['ever_smoker'] = self.clin['smoking_status'].isin(['former', 'current']).astype(int)
        self.clin['current_smoker'] = (self.clin['smoking_status'] == 'current').astype(int)
        self.clin['stage_I'] = self.clin['stage'].isin(['Stage IA', 'Stage IB']).astype(int)
        self.clin['stage_II'] = self.clin['stage'].isin(['Stage IIA', 'Stage IIB']).astype(int)
        self.clin['stage_III'] = (self.clin['stage'] == 'Stage IIIA').astype(int)
        self.clin['sex'] = (self.clin['sex'] == 'FEMALE').astype(int)
        self.clin = self.clin.drop(columns=['smoking_status', 'stage'])

    def _build_outcomes_table(self):
        # extract outcomes from the diagnosis and treatment timelines
        rx_treatment_site_map = {'Distant Recurrence': 'distant_recur_rx_start', 'Local Recurrence': 'local_recur_rx_start'}
        rx_regiment_ind_map = {'Recurrence': 'recurrence_reg_start'}
        rx = self._extract_timeline_outcomes(self.timeline_rx, 'anatomic_treatment_site', rx_treatment_site_map)
        rx = rx.merge(self._extract_timeline_outcomes(self.timeline_rx, 'regimen_indication', rx_regiment_ind_map), how='outer', left_index=True, right_index=True)
        dx_status_map = {'Distant Metastasis': 'distant_metas_dx_start', 'Locoregional Recurrence': 'locoreg_recur_dx_start', 'Locoregional Recurrence|Distant Metastasis': 'locoreg_distant_metas_dx_start', 'Distant Metastasis|New Primary Tumor': 'distant_metas_new_prim_dx_start'}
        dx = self._extract_timeline_outcomes(self.timeline_dx, 'status', dx_status_map)
        self.timelines = dx.merge(rx, how='outer', left_index=True, right_index=True)
        ttr_df = self.clin.merge(self.timelines, how='left', left_index=True, right_index=True)
        self.ttr_df = ttr_df.copy()
        # filter cases to recurrences, excluding new-primary-only cases and censored cases
        self.id_tracker['Has rec sample'] = list(self.explicit_cases)
        self.id_tracker['Exclusion workflow'] = defaultdict(list)
        ttr_df = ttr_df[ttr_df.apply(self._filter_recurrence_cases, axis=1)]
        # calculate TTR
        self.id_tracker['Get ttr'] = defaultdict(lambda: defaultdict(list))
        ttr_df['ttr'] = ttr_df.apply(self._get_ttr, axis=1)
        explicit_cases_missing_ttr = ttr_df.loc[ttr_df['ttr'].isna() & ttr_df.index.isin(self.explicit_cases)].index.tolist()
        if len(explicit_cases_missing_ttr) > 0: print(f'{len(explicit_cases_missing_ttr)} explicit cases missing TTR')
        self.id_tracker['Explicit cases'] = self.explicit_cases
        self.id_tracker['Get ttr']['Explicit w/o ttr'] = explicit_cases_missing_ttr
        # final cleanup
        self.id_tracker['No ttr'] = ttr_df[ttr_df['ttr'].isna()].index.tolist()
        ttr_df.dropna(subset=['ttr'], inplace=True)
        no_longer_needed = ['dfi.time_4', 'days_to_nte_after_initial_treatment_5', 'new_neoplasm_event_type_5', 'new_neoplasm_event_type_7', 'nte_dx_days_to_4', 'new_neoplasm_event_type_1', 'nte_after_initial_treatment_3', 'nte_type_4', 'days_to_additional_surgery_locoregional_procedure_1', 'days_to_additional_surgery_metastatic_procedure_1', 'dfi_4', 'dfs_months_3', 'dfs_status_3']
        self.clin = pd.concat([self.clin, ttr_df[['ttr']]], axis=1, join='inner').drop(columns=no_longer_needed)

    def _extract_timeline_outcomes(self, df, pivot_col, col_map):
        df_filtered = df[df[pivot_col].isin(col_map.keys())]
        self.explicit_cases.update(list(df_filtered['patient_id'] + '-01'))
        pivot_df = df_filtered.pivot_table(index='patient_id', columns=pivot_col, values='start_date', aggfunc='min')
        pivot_df.index = pivot_df.index + '-01' # this data is static to the patient
        pivot_df.rename(columns=col_map, inplace=True)
        return pivot_df

    def _filter_recurrence_cases(self, row):
        def track(reason, keep, add_to_explicit=False):
            self.id_tracker['Exclusion workflow'][reason].append(row.name)
            if add_to_explicit: self.explicit_cases.add(row.name)
            return keep
        if row.name in self.explicit_cases: 
            return True # 1. already identified as recurrence case
        has_recur_col = row[['distant_recur_rx_start', 'local_recur_rx_start', 'recurrence_reg_start', 'distant_metas_dx_start', 'locoreg_recur_dx_start', 'locoreg_distant_metas_dx_start', 'distant_metas_new_prim_dx_start']].notna().any() # 2. check for non-null recurrence column
        nte_vals = row[['nte_type_4', 'new_neoplasm_event_type_1', 'new_neoplasm_event_type_5', 'new_neoplasm_event_type_7']].dropna()
        has_recur_nte = nte_vals.isin({'Distant Metastasis', 'Locoregional Recurrence', 'Locoregional Recurrence|Distant Metastasis', 'Distant Metastasis|New Primary Tumor'}).any() # 3. check for NTE indicating recurrence
        if has_recur_col or has_recur_nte:
            return track('Explicit recurrence (col)' if has_recur_col else 'Explicit recurrence (NTE)', True, add_to_explicit=True)
        if row.get('dfi_4') == 1: 
            return track('Has DFI=1 (explicit)', True, add_to_explicit=True) # 4. DFI = 1, keep
        if row.get('dfi_4') == 0: 
            return track('Has DFI=0 *', False) # 5. DFI = 0 (censored), exclude
        if not nte_vals.empty:
            only_new_primary = nte_vals.isin({'New Primary Tumor'}).all()
            if only_new_primary:
                return track('Has new primary events only *', False) # 6. all non-na NTE indicate new primary
        return track('No recurrence clearly indicated *', False)
    
    def _get_ttr(self, row):
        tiers = [
            ['distant_recur_rx_start', 'local_recur_rx_start', 'recurrence_reg_start', 'locoreg_recur_dx_start'],
            ['locoreg_distant_metas_dx_start', 'distant_metas_dx_start', 'distant_metas_new_prim_dx_start'],
            ['days_to_nte_after_initial_treatment_5', 'nte_dx_days_to_4', 'dfi.time_4'],
            # days_to_additional_surgery ended up not used by any patients, so it's excluded from the manuscript
            ['days_to_additional_surgery_metastatic_procedure_1', 'days_to_additional_surgery_locoregional_procedure_1']
        ]
        for i, tier in enumerate(tiers):
            valid = [(col, row[col]) for col in tier if pd.notna(row[col])]
            if valid:
                col_used, val_used = min(valid, key=lambda x: x[1])
                self.id_tracker['Get ttr'][f'Tier {i+1}'][col_used].append(row.name)
                return val_used
        return np.nan

class TRACERx:
    def __init__(self, keep, paths, hgnc, predetermined_cutoff=None):
        self.keep = keep
        self.paths = paths
        self.hgnc = hgnc
        self.id_tracker = {}
        print('\n---- TracerX ----')
        self.get_clinical() # https://zenodo.org/records/10932811
        self.get_expression()
        self.get_mutation()
        self.finalize_data(predetermined_cutoff)
    
    def update_id_tracker(self, key, ids):
        if key is None:
            if isinstance(ids, dict): 
                norm = {k: (list(v) if not isinstance(v, (list, set)) else v) for k, v in ids.items()}
                self.id_tracker.update(norm)
            else: print("Error in call to self.update_id_tracker; 'ids' must be a dictionary when 'key' is None")
        else: self.id_tracker[key] = list(ids)

    def print_id_tracker(self, d, path=(), final_cohort_only=False):
        if isinstance(d, dict):
            for k, v in d.items(): self.print_id_tracker(v, path + (k,))
        elif isinstance(d, list) or isinstance(d, set):
            if final_cohort_only: print(" -> ".join(path), ":", len([i for i in d if i in self.clin.index]))
            else: print(" -> ".join(path), ":", len(d))
    
    def get_unique_tumors(self):
        return self.clin.drop(columns=['sample_name_cruk']).drop_duplicates()['tumour_id_muttable_cruk'].unique()

    def get_clinical(self):
        def quick_format(key):
            df = load_data('csv', key, data_path=self.paths['tracerx'], clean=False)
            df.columns = [col.lower() for col in df.columns]
            return df[self.keep['tracerx'][key]].copy() 
        clinhist = quick_format('clinicohistopathological_data') # no gained information
        self.clinhist = clinhist.loc[clinhist['is.primary.tumour.tx421']=='primary', ['sample_name_cruk', 'tumour_id_muttable_cruk']].copy()
        all_tm = quick_format('TRACERx421_all_tumour_df')
        self.all_tm = all_tm[all_tm['histology_3']=='LUAD'].copy()
        all_pt = quick_format('TRACERx421_all_patient_df')
        self.all_pt = all_pt.loc[all_pt['recurrence_time_use'].notna(), ['tumour_id_muttable_cruk', 'margin_status_per_patient', 'recurrence_time_use']].copy()
        self.clin = self.clinhist.merge(self.all_tm, on='tumour_id_muttable_cruk', how='inner') # inner bc need both sample_type == primary and histology == LUAD or LUSC
        self.update_id_tracker('Unique primary LUAD (+LUSC if appl.) tumors', self.get_unique_tumors())
        self.clin = self.clin.merge(self.all_pt, on='tumour_id_muttable_cruk', how='inner') # inner bc need recurrence_time_use to be notna
        self.update_id_tracker('Primary tumors with non-null recurrence_time_use', self.get_unique_tumors())
        self.update_id_tracker('Stage IIIB *', self.clin.loc[self.clin['pathologytnm']=='IIIB', 'tumour_id_muttable_cruk'].tolist())
        self.clin = self.clin[self.clin['pathologytnm']!='IIIB'].copy()
        def assign_smoking_status(val): # there are no nulls
            v = str(val).strip().lower()
            if v == 'ex-smoker': return 'former'
            elif v == 'smoker': return 'current'
            elif v == 'never smoked': return 'never'
            else: raise ValueError(f'Unknown value: {v}')
        self.clin['smoking_status'] = self.clin['smoking_status_merged'].apply(assign_smoking_status)
        self.clin = self.clin.drop(columns=['smoking_status_merged'])
    
    def get_expression(self):
        expr = load_data('csv', 'rsem_counts_mat', data_path=self.paths['tracerx'])
        expr = self._update_gene_symbols(expr, 'gene_id')
        expr = expr.groupby('approved_symbol').sum()
        expr.index = list(expr.index) # removes "approved symbol" label
        expr = expr.T
        # expr regions are pre-filtered for QC (as described in the paper - confirmed with Carlos)
        self.clin = self.clin[self.clin['sample_name_cruk'].isin(expr.index.tolist())].copy()
        self.update_id_tracker('Primary tumors with with expression regions passing QC', self.get_unique_tumors())
        # now limit regions in expr to those in self.clin, which are primary
        temp = self.clin[['sample_name_cruk', 'tumour_id_muttable_cruk']].drop_duplicates().set_index('sample_name_cruk')
        expr = pd.concat([expr, temp], axis=1, join='inner')
        self.expr = expr.groupby('tumour_id_muttable_cruk').median() # median across regions (Carlos's suggestion)
        self.expr.index = list(self.expr.index)
        self.expr = self.expr.loc[:, (self.expr > 0).any(axis=0)].add_suffix('_expr') # drop all-zero columns
    
    def get_mutation(self):
        # Mut format looks like ANNOVAR (https://annovar.openbioinformatics.org/en/latest/articles/wANNOVAR/)
        mut = load_data('csv', 'TRACERx421_mutation_table', data_path=self.paths['tracerx'], clean=False)
        is_silent = mut['exonic.func'].astype('string').str.lower().isin(['synonymous snv', 'synonymous']) # exonic.func = Exonic variant function (non-synonymous, synonymous, etc)
        is_rna = mut['func'].astype('string').str.lower().str.startswith('ncrna_', na=False) # func = Variant function (exonic, intronic, intergenic, UTR, etc)
        is_xy = mut['chr'].astype('string').isin(['chrX','chrY'])
        mut = mut[~is_silent & ~is_rna & ~is_xy].copy()
        mut = self._update_gene_symbols(mut, 'Hugo_Symbol')
        mut = mut[mut['tumour_id'].isin(self.clin['tumour_id_muttable_cruk'])]
        self.raw_mut = mut.copy()
        self.mut = pd.crosstab(mut['tumour_id'], mut['approved_symbol']).astype(float)
        self.mut.columns = [f"{col}_mut" for col in self.mut.columns]
        self.qc = load_data('tsv', '20221109_TRACERx421_manual_qc_sheet', data_path=self.paths['tracerx'])
    
    def get_cohort_selection(self, include_expr=True, include_mut=True):
        idxs = self.clin.index # need clinical otherwise there's no recurrence label
        if include_expr: idxs = idxs.intersection(self.expr.index)
        if include_mut: idxs = idxs.intersection(self.mut.index)
        return idxs.tolist()
        
    def finalize_data(self, predetermined_cutoff):
        self.clin = set_idx(self.clin.drop(columns=['sample_name_cruk']).drop_duplicates(), idx_col='tumour_id_muttable_cruk')
        if self.clin['cruk_id'].nunique() != len(self.clin): print('Multiple tumors for a single patient')
        self.clin = self.clin.rename(columns={'clinical_sex': 'sex', 'pathologytnm': 'stage', 'recurrence_time_use': 'ttr'})
        self.clin_raw = self.clin.copy()
        self.clin = self.clin[['age', 'sex', 'stage', 'smoking_status', 'ethnicity', 'ttr']].copy()
        self.cohort = sorted(self.get_cohort_selection(include_expr=True, include_mut=True))
        for name in ['clin', 'expr', 'mut']:
            df = getattr(self, name)
            df = df.loc[self.cohort].copy()
            assert not df.isna().values.any(), f"NaNs detected in {name}"
            df = df.loc[:, df.nunique() > 1].copy()
            setattr(self, name, df.reindex(self.cohort))
        self.raw_mut = self.raw_mut[self.raw_mut['tumour_id'].isin(self.cohort)].copy()
        self.clin_raw = self.clin_raw.loc[self.cohort].copy()
        print(f"Cutoff used: Predetermined ({predetermined_cutoff})")
        self.clin['label'] = (self.clin['ttr'] < predetermined_cutoff).astype(int)
        early, late = [self.clin[self.clin['label']==i].index.tolist() for i in [1, 0]]
        print(f"{len(self.clin)} patients ({len(self.clin)-self.clin['label'].sum()} late, {self.clin['label'].sum()} early)")
        self.clin_for_table1 = self.clin.copy()
        self.clin['ever_smoker'] = self.clin['smoking_status'].isin(['former', 'current']).astype(int)
        self.clin['current_smoker'] = (self.clin['smoking_status'] == 'current').astype(int)
        self.clin['stage_I'] = self.clin['stage'].isin(['IA', 'IB']).astype(int)
        self.clin['stage_II'] = self.clin['stage'].isin(['IIA', 'IIB']).astype(int)
        self.clin['stage_III'] = (self.clin['stage'] == 'IIIA').astype(int)
        self.clin['sex'] = (self.clin['sex'] == 'Female').astype(int)
        self.clin = self.clin.drop(columns=['smoking_status', 'stage'])
        
    def _update_gene_symbols(self, df, gene_col, drop_unmapped=True):
        df = df.copy()
        def _mapper_dict(key):
            d = self.hgnc.get(key)
            return {str(k).strip(): v for k, v in d.items()} # normalize keys to string and strip spaces
        def _norm_series(s):
            return s.astype('string').str.strip()
        approved_map = _mapper_dict('Symbol')
        df['approved_symbol'] = _norm_series(df[gene_col]).str.upper().map(approved_map)        
        df['approved_symbol'] = df['approved_symbol'].infer_objects(copy=False)
        if drop_unmapped:
            df = df.dropna(subset=['approved_symbol'])
        return df.drop(columns=[gene_col])

class LUNG:
    def __init__(self, tcga, tracerx, dp=data_path):
        self.tcga = tcga
        self.trcr = tracerx
        self.dp = dp
        self.og_cols = {'clin': tcga.clin.columns.tolist(), 'expr': tcga.expr.columns.tolist(), 'mut': tcga.mut.columns.tolist()}
        self.og_cols['clin'] = [c for c in self.og_cols['clin'] if c not in ['label']]

    def preprocess(self, presplit=True, fn='', test_size=0.2, valid_size=0.2, rs=42):
        # TCGA
        tcga_all = self.tcga.clin.join(self.tcga.expr, how='inner').join(self.tcga.mut, how='inner')
        self.tcga_x = tcga_all.drop(columns=['label']).copy()
        self.tcga_y = tcga_all['label'].copy()
        if presplit:
            idxs = load_data('json', fn, self.dp)
            self.tcga_x_train, self.tcga_x_test, self.tcga_x_valid = [self.tcga_x.loc[idxs[split]].copy() for split in ['train', 'test', 'valid']]
            self.tcga_y_train, self.tcga_y_test, self.tcga_y_valid = [self.tcga_y.loc[idxs[split]].copy() for split in ['train', 'test', 'valid']]
        else:
            x_temp, self.tcga_x_test, y_temp, self.tcga_y_test = train_test_split(self.tcga_x, self.tcga_y, test_size=test_size, stratify=self.tcga_y, random_state=rs)
            self.tcga_x_train, self.tcga_x_valid, self.tcga_y_train, self.tcga_y_valid = train_test_split(x_temp, y_temp, test_size=valid_size/(1-test_size), stratify=y_temp, random_state=rs)
            out_dict = {'train': self.tcga_x_train.index.tolist(), 'valid': self.tcga_x_valid.index.tolist(), 'test': self.tcga_x_test.index.tolist()}
            if fn != '': write_file('json', fn, out_dict, self.dp)
        print(f"\nTCGA splits: {len(self.tcga_x_train)} train, {len(self.tcga_x_valid)} valid, {len(self.tcga_x_test)} test")
        # TracerX
        trcr_all = self.trcr.clin.join(self.trcr.expr, how='inner').join(self.trcr.mut, how='inner')
        self.trcr_x = trcr_all.drop(columns=['label']).copy()
        self.trcr_y = trcr_all['label'].copy()

    def finalize(self):
        keep = self.tcga_x_train.columns[self.tcga_x_train.nunique() > 1]
        self.tcga_x_train, self.tcga_x_test, self.tcga_x_valid, self.trcr_x = [df[keep].copy() for df in [self.tcga_x_train, self.tcga_x_test, self.tcga_x_valid, self.trcr_x]]
        self.cols = {kind: get_new_cols(keep, kind_og, sort=True) for kind, kind_og in self.og_cols.items()}
        outstr = f"\nFeatures removed:"
        for kind in self.cols.keys():
            outstr += f" {len(self.og_cols[kind])-len(self.cols[kind])} {kind} ({len(self.cols[kind])} remaining)"
        print(outstr)

# TRACERx expression: Upper-quartile normalize each sample/tumor (row) to 1000 using 75th percentile of nonzero values (TCGA data arrives this way)
def uqx1000(tracerx_expr):
    X = tracerx_expr.to_numpy(dtype=np.float32)
    out = np.empty_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        v = X[i]
        nz = v[v > 0]
        s = np.percentile(nz, 75) if nz.size else 1.0
        if not np.isfinite(s) or s <= 0:
            s = 1.0
        out[i] = (v / s) * 1000.0
    return pd.DataFrame(out, index=tracerx_expr.index, columns=tracerx_expr.columns)

def transform_expr(tcga_expr, tracerx_expr):
    if not tcga_expr.columns.equals(tracerx_expr.columns):
        raise ValueError('Expression columns do not match (names/order)')
    edfs = [tcga_expr.copy(), uqx1000(tracerx_expr)]
    outs = [np.log2(edf.to_numpy(dtype=np.float32) + 1.0) for edf in edfs]
    return [pd.DataFrame(o, index=edf.index, columns=edf.columns) for o, edf in zip(outs, edfs)]

def transform_mut(kind, tcga_mut, tracerx_mut):
    if not tcga_mut.columns.equals(tracerx_mut.columns):
        raise ValueError('Mutation columns do not match (names/order)')
    mdfs = [tcga_mut.copy(), tracerx_mut.copy()]
    if kind is None:
        return mdfs
    if kind == 'binarize':
        return [(mdf > 0).astype(np.float32) for mdf in mdfs]
    if kind == 'log1p':
        outs = [np.log1p(mdf.to_numpy(dtype=np.float32)) for mdf in mdfs]
        return [pd.DataFrame(o, index=mdf.index, columns=mdf.columns) for o, mdf in zip(outs, mdfs)]
    raise ValueError(f'Invalid transformation type: {kind}')

def prepare_data(paths, keep, hgnc, mut_transform=None):
    tcga = TCGA(keep, paths, hgnc)
    tracerx = TRACERx(keep, paths, hgnc, predetermined_cutoff=tcga.cutoff)
    shared_clin_cols = ['age', 'sex', 'stage_I', 'stage_II', 'stage_III', 'ever_smoker', 'current_smoker', 'label']
    tcga.clin = tcga.clin[shared_clin_cols].copy()
    tracerx.clin = tracerx.clin[shared_clin_cols].copy()
    # align and transform expression features (Upper-quartile (75th percentile) normalize each tumor to 1000, then log2(x+1))
    overlapping_expr = sorted(list(set(tcga.expr.columns.tolist()) & set(tracerx.expr.columns.tolist())))
    tcga.expr, tracerx.expr = [df.reindex(columns=overlapping_expr).copy() for df in [tcga.expr, tracerx.expr]]  
    tcga.expr, tracerx.expr = transform_expr(tcga.expr, tracerx.expr)
    # align and transform mutation features
    overlapping_mut = sorted(list(set(tcga.mut.columns.tolist()) & set(tracerx.mut.columns.tolist())))
    tcga.mut, tracerx.mut = [df.reindex(columns=overlapping_mut).copy() for df in [tcga.mut, tracerx.mut]]  
    tcga.mut, tracerx.mut = transform_mut(mut_transform, tcga.mut, tracerx.mut)
    lung = LUNG(tcga, tracerx)
    return lung, tcga, tracerx

class Table1:
    def __init__(self, tcga, tracerx, decpts=2):
        self.tcga = tcga
        self.trcr = tracerx
        self.decpts = decpts

    def get_idxs(self, df): # return indices for All, Early (label==1), Late (label==0)
        return df.index, df.index[df['label'] == 1], df.index[df['label'] == 0]

    def is_continuous(self, s, max_cats=10): # numeric with >max_cats unique non-nan values → continuous; else categorical
        if not is_numeric_dtype(s): return False
        return s.dropna().nunique() > max_cats

    def median(self, idxs, s, as_str=True):
        med = float(np.nanmedian(s.loc[idxs])) if len(idxs) else np.nan
        med_out = np.round(med, self.decpts)
        return f"{med_out:.{self.decpts}f}" if as_str else med_out
    
    def iqr(self, x, as_str=True):
        x = x.dropna().to_numpy()
        if x.size == 0: return np.nan
        q1, q3 = np.percentile(x, [25, 75])
        return f"{q1:.{self.decpts}f}-{q3:.{self.decpts}f}"
        
    def count(self, idxs, lvl, sc, add_pct=True):
        ct = int((sc.loc[idxs] == lvl).sum())
        pct = round(ct/len(idxs)*100)
        return f"{ct} ({pct})" if add_pct else ct

    def table1(self, dataset='tcga', fn=''):
        assert dataset in ['tcga', 'trcr'], "Dataset must be 'tcga' or 'trcr'"
        df = self.tcga.clin_for_table1.copy() if dataset == 'tcga' else self.trcr.clin_for_table1.copy()
        all_idx, early_idx, late_idx = self.get_idxs(df)
        rows = []
        rows.append(("N (patients)", len(all_idx), len(early_idx), len(late_idx))) # First row: N patients
        features = [c for c in df.columns if c != 'label']
        for col in features:
            s = df[col]
            if self.is_continuous(s):
                rows.append((f"{col} — median", self.median(all_idx, s), self.median(early_idx, s), self.median(late_idx, s))) # median
                rows.append((f"{col} — IQR", self.iqr(s.loc[all_idx]), self.iqr(s.loc[early_idx]), self.iqr(s.loc[late_idx]))) # IQR
            else:
                sc = s.astype('object').fillna('Missing')
                levels = [lvl for lvl in sorted(sc.unique(), key=lambda x: str(x))]
                if 'Missing' in levels:
                    levels = [l for l in levels if l != 'Missing'] + ['Missing'] # make sure 'Missing' (if present) is last for readability
                for lvl in levels:
                    rows.append((f"{col} = {lvl}", self.count(all_idx, lvl, sc), self.count(early_idx, lvl, sc), self.count(late_idx, lvl, sc)))
        out = pd.DataFrame(rows, columns=['Feature', 'All', 'Early', 'Late'], dtype=object)
        if fn != '': write_file('csv', fn, out, '../results/')
        return out

    def get_tables(self, tcga_fn='', trcr_fn=''):
        self.tcga_table1 = self.table1('tcga', tcga_fn)
        self.trcr_table1 = self.table1('trcr', trcr_fn)

class MutationFrequencies:
    def __init__(self, lung, step_pct=0.25):
        self.mfreq = (lung.tcga_x_train[lung.cols['mut']] > 0).mean()
        self.pct_steps = np.arange(0.0, 10.0 + step_pct, step_pct)
        self.thresholds = self.pct_steps / 100.0
        self.n_genes = [(self.mfreq >= t).sum() for t in self.thresholds]

    def print_retention(self, cutoff=None):
        if cutoff is None:
            for i in range(len(self.thresholds)):
                print(f"#{i}: {self.thresholds[i]:.5f} -> {self.n_genes[i]} genes")
        elif isinstance(cutoff, float): 
            print((self.mfreq >= cutoff).sum())
    
    def plot_retention(self, p=None, y_major=1000, y_minor=500, figsize=(6.5, 3.5)):
        fig, ax = plt.subplots(figsize=figsize, dpi=600)
        ax.plot(self.pct_steps, self.n_genes, marker='.')
        if p is not None:
            ax.axvline(p*100, color='red', linestyle='--', label=f'f = {p*100}%')
            ax.legend()
        ax.set_xticks(np.arange(0, 11, 1))
        ax.set_xlabel('Mutation Frequency Threshold (%)'); ax.set_ylabel('Number of Genes Retained')
        ax.set_title('Impact of Mutation Frequency Cutoff on Gene Retention', weight='bold')
        ax.yaxis.set_major_locator(MultipleLocator(y_major))  # e.g., every 500
        ax.yaxis.set_minor_locator(MultipleLocator(y_minor))  # e.g., every 100
        ax.grid(True, which='both', alpha=0.5)
        plt.show()

class AlignTRACERx:
    """ Apply unsupervised domain alignment between TCGA and TRACERx (expression and, optionally, age) """
    def __init__(self, columns, method='quantile', align_age=False, n_q=100):
        self.aligncols = ['age', *columns['expr']] if align_age else columns['expr']
        self.method = method
        self.n_q = int(n_q)
    def align_continuous(self, xtr, trcr_x):
        aligned_trcr = trcr_x.copy()
        if self.method == 'quantile': # align each feature's distribution
            quantiles = np.linspace(0, 1, self.n_q)
            for col in self.aligncols:
                tcga_vals = xtr[col].to_numpy()
                trcr_vals = trcr_x[col].to_numpy()
                tcga_q = np.quantile(tcga_vals, quantiles)
                trcr_q = np.quantile(trcr_vals, quantiles)
                aligned_trcr[col] = np.interp(trcr_vals, trcr_q, tcga_q)
        return aligned_trcr

class SelectScalers:
    def __init__(self, x_train, columns, to_scale=['clin', 'expr'], skew_symmetric=0.5, skew_heavy=1.0, outlier_heavy=0.01, many_zeros=0.9):
        self.columns = columns
        self.to_scale = to_scale
        self.skew_symm, self.skew_heavy = skew_symmetric, skew_heavy
        self.outlier_heavy, self.many_zeros = outlier_heavy, many_zeros
        self.recommend_feature_scalers(x_train)

    def return_empty(self, reason):
        stats_dict = {k: np.nan for k in ['outlier_f', 'zero_f', 'skew', 'vmin', 'vmax', 'uniq', 'bound01', 'binary']}
        return (stats_dict, 'none', reason)
    
    def recommend_feature_scalers(self, x_train):
        records = []
        for cat in ['clin', 'expr', 'mut']:
            for col in self.columns[cat]:
                if cat == 'mut':
                    stats, scaler, reasoning = self.return_empty('scaled separately')
                elif cat not in self.to_scale:
                    stats, scaler, reasoning = self.return_empty('not selected for scaling')
                elif cat == 'clin' and col != 'age':
                    stats, scaler, reasoning = self.return_empty('categorical')
                else: 
                    stats = self._series_stats(x_train[col])
                    scaler, reasoning = self._decide_scaler(stats, kind=cat)
                records.append({'category': cat, 'feature': col, 'scaler': scaler, 'reasoning': reasoning, **stats})
        self.recs = pd.DataFrame.from_records(records)
        if len(self.recs[(self.recs['category']=='expr')&(self.recs['reasoning']=='binary')]) > 0: print('Binary expression features detected')

    def _series_stats(self, x): # uses Fisher-Pearson sample skewness
        v = x.to_numpy()
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1 if q3 > q1 else 0.0
        outlier_f = float(((v < (q1 - 1.5 * iqr))|(v > (q3 + 1.5 * iqr))).mean()) if iqr > 0 else 0.0
        s = {'outlier_f': outlier_f, 'zero_f': float((v == 0).mean()), 'skew': float(pd.Series(v).skew()), 'vmin': float(v.min()), 'vmax': float(v.max()), 'uniq': int(np.unique(v).size)}
        s.update({'bound01': (s['vmin'] >= 0.0) and (s['vmax'] <= 1.0), 'binary': (s['vmin'] == 0) and (s['vmax'] == 1) and (s['uniq'] == 2)})
        if s['vmin'] == s['vmax']: raise ValueError('Constant feature present')
        return s

    def _decide_scaler(self, stats, kind):
        many_outliers = stats['outlier_f'] > self.outlier_heavy
        skew_type = 'mild' if abs(stats['skew']) <= self.skew_symm else 'heavy' if abs(stats['skew']) > self.skew_heavy else 'moderate'
        if kind == 'expr':
            if stats['binary']: raise ValueError(f'Binary feature in expr data')
            return ('robust', 'many outliers') if many_outliers else ('standard', 'few outliers') # already log-transformed so we don't need a second normality like YJ
        if kind == 'clin':
            if stats['bound01']: return ('none', 'already normalized to [0,1] range')
            if skew_type == 'mild': return ('robust', 'many outliers, mild skew') if many_outliers else ('standard', 'few outliers, mild skew')
            if skew_type == 'heavy': return ('yj + robust', 'many outliers, heavy skew') if many_outliers else ('yj + standard', 'few outliers, heavy skew') # Works for neg/zero vals
            return ('yj + robust', 'many outliers, moderate skew') if many_outliers else ('yj + standard', 'few outliers, moderate skew')

    def _build_transformer_from_recs(self):
        def tag_to_transformer(s_tag):
            tag = s_tag.strip().lower()
            if tag == 'none': return 'passthrough'
            if tag == 'standard': return StandardScaler()
            if tag == 'robust': return RobustScaler()
            if tag == 'yj + standard': return Pipeline([('yj', PowerTransformer(method='yeo-johnson', standardize=False)), ('std', StandardScaler())])
            if tag == 'yj + robust': return Pipeline([('yj', PowerTransformer(method='yeo-johnson', standardize=False)), ('rob', RobustScaler())])
            raise ValueError(f'Unknown scaler tag: {s_tag}')
        def slug(name):
            return name.lower().replace('+', 'plus').replace(' ', '_').replace('/', '_').replace('-', '_')
        # group features by the same scaler recipe to minimize the number of transformers (creates dict where keys = scaler tags and vals = list of features per tag)
        self.recs['scaler'] = self.recs['scaler'].fillna('none')
        recs_ct = self.recs[self.recs['category'].isin(['clin', 'expr'])].copy()
        groups = recs_ct[['feature', 'scaler']].groupby('scaler')['feature'].apply(list).to_dict()
        transformers = [(slug(tag), tag_to_transformer(tag), cols) for tag, cols in groups.items()]
        return ColumnTransformer(transformers=transformers, remainder='drop', verbose_feature_names_out=False).set_output(transform='pandas')
    
    def format_mut_data(self, xtrain, xvalid, xtest, trcr_x, trcr_x_aligned=None):
        mut_cols = self.columns['mut']
        dfs = [df[mut_cols].copy() if df is not None else None for df in [xtrain, xvalid, xtest, trcr_x, trcr_x_aligned]]
        return dfs
        
    def apply_scalers_from_recs(self, xtrain, xvalid, xtest, trcr_x, trcr_x_aligned=None):
        ct = self._build_transformer_from_recs()
        # transform only clin/expr (don't need to slice because of drop above)
        xtr_ce = ct.fit_transform(xtrain)
        xva_ce, xte_ce, xtrcr_ce = [ct.transform(ds) for ds in [xvalid, xtest, trcr_x]]
        xtrcr_aligned_ce = ct.transform(trcr_x_aligned) if trcr_x_aligned is not None else None
        # fallback for older scikit-learn: strip any '<step>__' prefixes if present
        for df in [xtr_ce, xva_ce, xte_ce, xtrcr_ce, xtrcr_aligned_ce]:
            if df is None: continue
            if isinstance(df.columns[0], str) and '__' in df.columns[0]:
                df.columns = [c.split('__', 1)[-1] for c in df.columns]
        # mutation passthrough
        xtr_m, xva_m, xte_m, xtrcr_m, xtrcr_aligned_m = self.format_mut_data(xtrain, xvalid, xtest, trcr_x, trcr_x_aligned)
        for a, b, name in [(xtr_ce, xtr_m, 'train'), (xva_ce, xva_m, 'val'), (xte_ce, xte_m, 'test'), (xtrcr_ce, xtrcr_m, 'tracerx')]:
            if not a.index.equals(b.index):
                raise RuntimeError(f'Index mismatch between CE and MUT for split={name}')
        if trcr_x_aligned is not None:
            if not xtrcr_aligned_ce.index.equals(xtrcr_aligned_m.index):
                raise RuntimeError(f'Index mismatch between CE and MUT for split=tracerx aligned')
        xtr = pd.concat([xtr_ce, xtr_m], axis=1)
        xva = pd.concat([xva_ce, xva_m], axis=1)
        xte = pd.concat([xte_ce, xte_m], axis=1)
        x_trcr = pd.concat([xtrcr_ce, xtrcr_m], axis=1)
        x_trcr_aligned = pd.concat([xtrcr_aligned_ce, xtrcr_aligned_m], axis=1) if trcr_x_aligned is not None else None
        expected = self.columns['clin'] + self.columns['expr'] + self.columns['mut']
        duplicated = pd.Series(xtr.columns).duplicated().any()
        missing = set(expected) - set(xtr.columns)
        if duplicated: raise RuntimeError('Duplicate column names after transform; check recs for duplicates across scalers.')
        if missing: print(f'[warn] Missing columns after transform: {sorted(missing)[:5]} ...')
        xtr, xva, xte, x_trcr, x_trcr_aligned = (df.reindex(columns=expected) if df is not None else None for df in [xtr, xva, xte, x_trcr, x_trcr_aligned])
        return (xtr, xva, xte, x_trcr) if x_trcr_aligned is None else (xtr, xva, xte, x_trcr, x_trcr_aligned)