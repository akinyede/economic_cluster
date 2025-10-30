#!/usr/bin/env python3
"""Data cleaning utilities for KC Cluster Prediction Tool"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataCleaner:
    """Clean and preprocess business data"""
    
    def __init__(self):
        self.revenue_cap = 50_000_000_000  # $50B - only cap truly impossible values
        self.min_revenue = 50_000  # $50K minimum
        
    def clean_business_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean business data and handle outliers using conservative, robust methods"""
        logger.info(f"Starting data cleaning for {len(df)} businesses")
        
        # Create a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # 1. Handle revenue outliers with IQR winsorization (with hard bounds as safety)
        if 'revenue_estimate' in cleaned_df.columns:
            original_revenue = cleaned_df['revenue_estimate'].copy()
            try:
                q1 = cleaned_df['revenue_estimate'].quantile(0.25)
                q3 = cleaned_df['revenue_estimate'].quantile(0.75)
                iqr = q3 - q1
                lower = max(self.min_revenue, q1 - 1.5 * iqr)
                upper = min(self.revenue_cap, q3 + 1.5 * iqr)
                cleaned_df['revenue_estimate'] = cleaned_df['revenue_estimate'].clip(lower=lower, upper=upper)
                cleaned_df.loc[cleaned_df['revenue_estimate'] < self.min_revenue, 'revenue_estimate'] = self.min_revenue
                cleaned_df.loc[cleaned_df['revenue_estimate'] > self.revenue_cap, 'revenue_estimate'] = self.revenue_cap
                revenue_capped = (original_revenue != cleaned_df['revenue_estimate']).sum()
                if revenue_capped > 0:
                    logger.info(f"  Winsorized revenue for {revenue_capped} businesses (IQR)")
            except Exception as e:
                logger.warning(f"IQR revenue winsorization failed ({e}); applying hard caps")
                cleaned_df.loc[cleaned_df['revenue_estimate'] > self.revenue_cap, 'revenue_estimate'] = self.revenue_cap
                cleaned_df.loc[cleaned_df['revenue_estimate'] < self.min_revenue, 'revenue_estimate'] = self.min_revenue
        
        # 2. Remove duplicate businesses
        # Identify duplicates based on name and address
        duplicate_cols = ['name', 'address'] if 'address' in cleaned_df.columns else ['name']
        duplicates = cleaned_df.duplicated(subset=duplicate_cols, keep='first')
        
        if duplicates.sum() > 0:
            logger.info(f"  Removing {duplicates.sum()} duplicate businesses")
            cleaned_df = cleaned_df[~duplicates]
        
        # 3. Fix NAICS codes
        # Ensure NAICS codes are strings and properly formatted (6-digit canonical)
        cleaned_df['naics_code'] = cleaned_df['naics_code'].astype(str).str.replace(".0$", "", regex=True).str.strip()
        cleaned_df['naics_code'] = cleaned_df['naics_code'].apply(self._canonicalize_naics)
        # Add 4-digit view for reporting convenience
        cleaned_df['naics4'] = cleaned_df['naics_code'].str[:4]
        
        # Remove invalid NAICS codes
        invalid_naics = cleaned_df['naics_code'].isin(['nan', 'None', '', '0', '00', '000'])
        if invalid_naics.sum() > 0:
            logger.info(f"  Found {invalid_naics.sum()} businesses with invalid NAICS codes")
            # Don't remove, but flag them
            cleaned_df.loc[invalid_naics, 'naics_code'] = '999999'  # Unknown category
        
        # 4. Clean employee counts (IQR winsorization + conservative caps)
        if 'employees' in cleaned_df.columns:
            employee_cap = 50_000
            try:
                q1e = cleaned_df['employees'].quantile(0.25)
                q3e = cleaned_df['employees'].quantile(0.75)
                iqre = q3e - q1e
                lower_e = max(1, int(q1e - 1.5 * iqre))
                upper_e = min(employee_cap, int(q3e + 1.5 * iqre))
                cleaned_df['employees'] = cleaned_df['employees'].clip(lower=lower_e, upper=upper_e)
            except Exception as e:
                logger.warning(f"IQR employee winsorization failed ({e}); applying hard caps")
                extreme_employees = cleaned_df['employees'] > employee_cap
                if extreme_employees.sum() > 0:
                    logger.info(f"  Capping employees for {extreme_employees.sum()} businesses")
                    cleaned_df.loc[extreme_employees, 'employees'] = employee_cap
            # Ensure minimum of 1 employee
            cleaned_df.loc[cleaned_df['employees'] < 1, 'employees'] = 1
        
        # 5. Fix year established
        current_year = datetime.now().year
        
        # Future years
        future_years = cleaned_df['year_established'] > current_year
        if future_years.sum() > 0:
            logger.info(f"  Fixing {future_years.sum()} businesses with future establishment years")
            cleaned_df.loc[future_years, 'year_established'] = current_year
        
        # Very old businesses (before 1800)
        very_old = cleaned_df['year_established'] < 1800
        if very_old.sum() > 0:
            logger.info(f"  Fixing {very_old.sum()} businesses with establishment year < 1800")
            cleaned_df.loc[very_old, 'year_established'] = 1900
        
        # 6. Ensure data consistency
        # Revenue per employee sanity check
        cleaned_df['revenue_per_employee'] = cleaned_df['revenue_estimate'] / cleaned_df['employees']
        
        # Flag suspicious revenue per employee (> $5M per employee is very high)
        suspicious_rpe = cleaned_df['revenue_per_employee'] > 5_000_000
        if suspicious_rpe.sum() > 0:
            logger.info(f"  Found {suspicious_rpe.sum()} businesses with revenue > $5M per employee")
            # Flag but don't adjust - let the analysis decide
            logger.warning(f"  ⚠️  {suspicious_rpe.sum()} businesses have very high revenue per employee")
            # Only adjust if it's truly impossible (>$10M per employee)
            extreme_rpe = cleaned_df['revenue_per_employee'] > 10_000_000
            if extreme_rpe.sum() > 0:
                cleaned_df.loc[extreme_rpe, 'revenue_estimate'] = (
                    cleaned_df.loc[extreme_rpe, 'employees'] * 2_000_000  # $2M per employee for high-value businesses
                )
        
        # Remove temporary column
        cleaned_df.drop('revenue_per_employee', axis=1, inplace=True)

        logger.info(f"Data cleaning complete: {len(cleaned_df)} businesses remain")

        return cleaned_df
    
    def deduplicate_by_priority(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates keeping the best record"""
        # Sort by revenue (descending) to keep the largest version of duplicates
        df_sorted = df.sort_values('revenue_estimate', ascending=False)
        
        # Keep first occurrence of each business name
        deduplicated = df_sorted.drop_duplicates(subset=['name'], keep='first')
        
        removed = len(df) - len(deduplicated)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate businesses by name")
        
        return deduplicated

    def _canonicalize_naics(self, code: str) -> str:
        """Return a 6-digit NAICS string when possible (pad/truncate), or '999999' for unknown."""
        if not code or code.lower() in {'nan', 'none', 'null', '0', '00', '000'}:
            return '999999'
        digits = ''.join(ch for ch in str(code) if ch.isdigit())
        if not digits:
            return '999999'
        if len(digits) >= 6:
            return digits[:6]
        return digits.ljust(6, '0')

    def normalize_address(self, s: str) -> str:
        """Basic address normalization: trim, title case, common suffix abbreviations."""
        if not s:
            return ''
        x = ' '.join(str(s).strip().split())
        x = x.title()
        # Common suffixes
        repl = {
            ' Street': ' St', ' Avenue': ' Ave', ' Road': ' Rd', ' Boulevard': ' Blvd',
            ' Drive': ' Dr', ' Lane': ' Ln', ' Court': ' Ct', ' Circle': ' Cir'
        }
        for k, v in repl.items():
            if x.endswith(k):
                x = x[: -len(k)] + v
        return x

    def deduplicate_entities(self, df: pd.DataFrame, name_threshold: int = 90) -> pd.DataFrame:
        """Fuzzy deduplicate businesses by name within the same county.

        - Uses fuzzywuzzy ratio if available; otherwise returns df unchanged.
        - Prefers the row with higher revenue_estimate when merging duplicates.
        """
        # Prefer RapidFuzz; fallback to fuzzywuzzy; otherwise skip
        fuzz_ratio = None
        try:
            from rapidfuzz import fuzz as rfuzz  # type: ignore
            fuzz_ratio = rfuzz.ratio
        except Exception:
            try:
                from fuzzywuzzy import fuzz as fw_fuzz  # type: ignore
                fuzz_ratio = fw_fuzz.ratio
            except Exception:
                logger.warning("No fuzzy matcher available; skipping fuzzy deduplication")
                return df

        if 'name' not in df.columns or 'county' not in df.columns:
            return df

        # Normalize names for comparison
        work = df.copy()
        work['__name_norm'] = work['name'].astype(str).str.strip().str.lower()
        work['__keep'] = True

        # Group within counties to limit comparisons
        dedup_rows = []
        for county, group in work.groupby('county'):
            group = group.copy()
            used = set()
            indices = list(group.index)
            for i in range(len(indices)):
                idx_i = indices[i]
                if idx_i in used:
                    continue
                cand_i = group.loc[idx_i]
                cluster = [idx_i]
                for j in range(i+1, len(indices)):
                    idx_j = indices[j]
                    if idx_j in used:
                        continue
                    cand_j = group.loc[idx_j]
                    score = fuzz_ratio(str(cand_i['__name_norm']), str(cand_j['__name_norm']))
                    if score >= name_threshold:
                        cluster.append(idx_j)
                        used.add(idx_j)
                # Keep best by revenue_estimate
                if len(cluster) == 1:
                    dedup_rows.append(group.loc[cluster[0]])
                else:
                    best_idx = max(cluster, key=lambda k: group.loc[k].get('revenue_estimate', 0))
                    dedup_rows.append(group.loc[best_idx])
                    used.update(cluster)
        result = pd.DataFrame(dedup_rows).drop(columns=['__name_norm', '__keep'], errors='ignore')
        logger.info(f"Fuzzy-deduplicated businesses: {len(df)} -> {len(result)}")
        return result
    
    def balance_industries(self, df: pd.DataFrame, target_pct: dict) -> pd.DataFrame:
        """Balance industry representation to avoid over-dominance"""
        # Example target_pct: {'541': 0.15, '484': 0.05, ...}
        
        balanced_dfs = []
        total_target = len(df)
        
        for naics_prefix, max_pct in target_pct.items():
            industry_df = df[df['naics_code'].str.startswith(naics_prefix)]
            max_count = int(total_target * max_pct)
            
            if len(industry_df) > max_count:
                # Sample the best businesses from this industry
                industry_df = industry_df.nlargest(max_count, 'revenue_estimate')
                logger.info(f"  Limited NAICS {naics_prefix} from {len(industry_df)} to {max_count} businesses")
            
            balanced_dfs.append(industry_df)
        
        # Add remaining businesses not in target industries
        targeted_naics = list(target_pct.keys())
        remaining_df = df[~df['naics_code'].str[:3].isin(targeted_naics)]
        balanced_dfs.append(remaining_df)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        
        return balanced_df

    def deduplicate_geospatial(self, df: pd.DataFrame, max_miles: float = 0.5, name_threshold: int = 85) -> pd.DataFrame:
        """Deduplicate by proximity (<= max_miles) and fuzzy name similarity within same county.

        Requires columns: ['name','county','latitude','longitude']
        """
        required = {'name', 'county', 'latitude', 'longitude'}
        if not required.issubset(set(df.columns)):
            logger.info("Geospatial dedup skipped (missing lat/lon)")
            return df
        # Prefer RapidFuzz; fallback to fuzzywuzzy; otherwise skip
        fuzz_ratio = None
        try:
            from rapidfuzz import fuzz as rfuzz  # type: ignore
            fuzz_ratio = rfuzz.ratio
        except Exception:
            try:
                from fuzzywuzzy import fuzz as fw_fuzz  # type: ignore
                fuzz_ratio = fw_fuzz.ratio
            except Exception:
                logger.warning("No fuzzy matcher available; skipping geospatial fuzzy match")
                return df

        def haversine_miles(lat1, lon1, lat2, lon2):
            from math import radians, cos, sin, asin, sqrt
            R = 3958.8  # Earth radius in miles
            dlat = radians(lat2 - lat1)
            dlon = radians(lon2 - lon1)
            a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            return R * c

        work = df.copy()
        work['__name_norm'] = work['name'].astype(str).str.strip().str.lower()
        kept = []
        used = set()

        # Group within county for efficiency
        for county, group in work.groupby('county'):
            idxs = list(group.index)
            for i in range(len(idxs)):
                ii = idxs[i]
                if ii in used:
                    continue
                can_i = group.loc[ii]
                cluster = [ii]
                for j in range(i+1, len(idxs)):
                    jj = idxs[j]
                    if jj in used:
                        continue
                    can_j = group.loc[jj]
                    try:
                        d = haversine_miles(float(can_i['latitude']), float(can_i['longitude']),
                                            float(can_j['latitude']), float(can_j['longitude']))
                    except Exception:
                        continue
                    if d <= max_miles:
                        score = fuzz_ratio(str(can_i['__name_norm']), str(can_j['__name_norm']))
                        if score >= name_threshold:
                            cluster.append(jj)
                # keep best by revenue_estimate
                best = max(cluster, key=lambda k: work.loc[k].get('revenue_estimate', 0))
                kept.append(work.loc[best])
                used.update(cluster)
        result = pd.DataFrame(kept).drop(columns=['__name_norm'], errors='ignore')
        logger.info(f"Geospatial-deduplicated businesses: {len(df)} -> {len(result)}")
        return result
