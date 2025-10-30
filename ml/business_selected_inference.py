#!/usr/bin/env python3
from pathlib import Path
from datetime import datetime
from typing import List
import json, re, pickle
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / 'models' / 'business_selected'
FEATURES_JSON = MODEL_DIR / 'business_selected_features.json'

def _parse_naics4(x) -> str:
    if pd.isna(x):
        return '0000'
    s = re.sub(r'\D', '', str(x))
    return (s + '0000')[:4] if s else '0000'

class BusinessSelectedPredictor:
    def __init__(self):
        with open(FEATURES_JSON, 'r') as f:
            self.features: List[str] = json.load(f)['features']
        self.model_rev = pickle.load(open(MODEL_DIR / 'rf_log_revenue_final.pkl', 'rb'))
        self.model_emp = pickle.load(open(MODEL_DIR / 'rf_log_employees_final.pkl', 'rb'))

    def _build_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        yr = pd.to_numeric(d.get('year_established'), errors='coerce')
        current_year = datetime.now().year
        d['age_years'] = (current_year - yr).clip(lower=0)
        d['innovation_score'] = pd.to_numeric(d.get('innovation_score'), errors='coerce')
        d['county'] = d.get('county').fillna('Unknown')
        d['county_class'] = d.get('county_class').fillna('Other')
        d['in_focus_county'] = d.get('in_focus_county').fillna(0).astype(int)
        d['naics4'] = d.get('naics_code').map(_parse_naics4)
        d['cluster_type'] = d.get('cluster_type').fillna('Other')
        na = pd.get_dummies(d['naics4'], prefix='naics4')
        co = pd.get_dummies(d['county'], prefix='county')
        cl = pd.get_dummies(d['cluster_type'], prefix='cluster')
        cc = pd.get_dummies(d['county_class'], prefix='county_class')
        base = d[['age_years','innovation_score','in_focus_county']]
        X_full = pd.concat([base, na, co, cl, cc], axis=1).astype(float)
        for col in self.features:
            if col not in X_full.columns:
                X_full[col] = 0.0
        return X_full[self.features].fillna(0.0)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._build_matrix(df)
        pred_lr = self.model_rev.predict(X)
        pred_le = self.model_emp.predict(X)
        pred_rev = np.maximum(0.0, np.expm1(pred_lr))
        pred_emp = np.maximum(0.0, np.expm1(pred_le))
        out = df.copy()
        out['pred_log_revenue'] = pred_lr
        out['pred_log_employees'] = pred_le
        out['pred_revenue'] = pred_rev
        out['pred_employees'] = pred_emp
        out['revenue_estimate'] = pred_rev
        out['employees'] = np.round(pred_emp).astype(int)
        return out

_singleton = None

def apply_business_predictions(df: pd.DataFrame) -> pd.DataFrame:
    global _singleton
    if _singleton is None:
        _singleton = BusinessSelectedPredictor()
    return _singleton.predict(df)
