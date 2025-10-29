#!/usr/bin/env python3
"""Selective enhancement of business data with KC-specific features"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class SelectiveEnhancer:
    """Selectively enhance businesses with KC Open Data features"""
    
    def __init__(self):
        self.feature_columns = [
            'kc_crime_safety',
            'kc_development_activity', 
            'kc_demographic_strength',
            'kc_infrastructure_score',
            'kc_market_access'
        ]
        self.enhancement_stats = {}
        
    def enhance_businesses(self, 
                          business_df: pd.DataFrame,
                          kc_features: pd.DataFrame,
                          license_matches: List = None) -> pd.DataFrame:
        """
        Enhance business data with KC features
        
        Strategy:
        1. Direct matches get actual KC data
        2. Same zip code businesses get zip-level averages
        3. Others get county-level defaults
        """
        logger.info(f"Starting selective enhancement for {len(business_df)} businesses")
        
        # Initialize KC feature columns
        for col in self.feature_columns:
            business_df[col] = None
        
        # Track enhancement sources
        business_df['kc_enhancement_source'] = 'none'
        
        # Normalize ZIPs to 5-digit strings on both frames
        try:
            business_df['_zip5'] = business_df.get('zip', '').astype(str).str.extract(r'(\d{5})')[0]
            if 'zip_code' in kc_features.columns:
                kc_features['zip_code'] = kc_features['zip_code'].astype(str).str.extract(r'(\d{5})')[0]
        except Exception:
            business_df['_zip5'] = business_df.get('zip', '')

        # 1. Apply direct matches from business licenses
        if license_matches:
            direct_count = self._apply_direct_matches(business_df, kc_features, license_matches)
            self.enhancement_stats['direct_matches'] = direct_count
        
        # 2. Apply zip-level features
        zip_count = self._apply_zip_features(business_df, kc_features)
        self.enhancement_stats['zip_matches'] = zip_count
        
        # 3. Apply county defaults
        default_count = self._apply_county_defaults(business_df)
        self.enhancement_stats['defaults'] = default_count
        
        # 4. Calculate derived features
        business_df = self._calculate_derived_features(business_df)
        
        # Log enhancement statistics
        self._log_enhancement_stats(business_df)
        
        return business_df
    
    def _apply_direct_matches(self, 
                            business_df: pd.DataFrame,
                            kc_features: pd.DataFrame,
                            matches: List) -> int:
        """Apply KC features for directly matched businesses"""
        logger.info("Applying direct matches...")
        
        enhanced_count = 0
        for match in matches:
            source_idx = match.source_idx
            
            # Get zip code from matched business
            zip_code = business_df.loc[source_idx, '_zip5'] if '_zip5' in business_df.columns else business_df.loc[source_idx, 'zip']
            
            if zip_code in kc_features['zip_code'].values:
                # Get features for this zip
                zip_features = kc_features[kc_features['zip_code'] == zip_code].iloc[0]
                
                # Apply features with confidence boost for direct matches
                for col in self.feature_columns:
                    if col in zip_features:
                        # Direct matches get a 10% boost for being verified businesses
                        value = zip_features[col] * 1.1
                        business_df.loc[source_idx, col] = min(100, value)
                
                business_df.loc[source_idx, 'kc_enhancement_source'] = 'direct'
                enhanced_count += 1
        
        logger.info(f"  Enhanced {enhanced_count} directly matched businesses")
        return enhanced_count
    
    def _apply_zip_features(self, 
                           business_df: pd.DataFrame,
                           kc_features: pd.DataFrame) -> int:
        """Apply zip-level average features"""
        logger.info("Applying zip-level features...")
        
        enhanced_count = 0
        
        # Group by zip code
        for zip_code in kc_features['zip_code'].dropna().unique():
            # Find businesses in this zip without direct enhancement
            if '_zip5' in business_df.columns:
                mask = (business_df['_zip5'] == zip_code) & \
                       (business_df['kc_enhancement_source'] == 'none')
            else:
                mask = (business_df['zip'] == zip_code) & \
                       (business_df['kc_enhancement_source'] == 'none')
            
            if mask.sum() > 0:
                # Get features for this zip
                zip_features = kc_features[kc_features['zip_code'] == zip_code].iloc[0]
                
                # Apply features
                for col in self.feature_columns:
                    if col in zip_features:
                        business_df.loc[mask, col] = zip_features[col]
                
                business_df.loc[mask, 'kc_enhancement_source'] = 'zip'
                enhanced_count += mask.sum()
        
        logger.info(f"  Enhanced {enhanced_count} businesses with zip-level features")
        return enhanced_count
    
    def _apply_county_defaults(self, business_df: pd.DataFrame) -> int:
        """Apply county-level defaults for remaining businesses"""
        logger.info("Applying county defaults...")
        
        # County-level defaults based on KC MSA characteristics
        county_defaults = {
            'Jackson': {
                'kc_crime_safety': 75,
                'kc_development_activity': 65,
                'kc_demographic_strength': 70,
                'kc_infrastructure_score': 75,
                'kc_market_access': 72
            },
            'Johnson': {
                'kc_crime_safety': 85,
                'kc_development_activity': 75,
                'kc_demographic_strength': 85,
                'kc_infrastructure_score': 85,
                'kc_market_access': 83
            },
            'Wyandotte': {
                'kc_crime_safety': 65,
                'kc_development_activity': 55,
                'kc_demographic_strength': 60,
                'kc_infrastructure_score': 65,
                'kc_market_access': 61
            },
            'Clay': {
                'kc_crime_safety': 80,
                'kc_development_activity': 60,
                'kc_demographic_strength': 75,
                'kc_infrastructure_score': 70,
                'kc_market_access': 71
            },
            'Platte': {
                'kc_crime_safety': 82,
                'kc_development_activity': 70,
                'kc_demographic_strength': 78,
                'kc_infrastructure_score': 72,
                'kc_market_access': 75
            },
            'Cass': {
                'kc_crime_safety': 78,
                'kc_development_activity': 50,
                'kc_demographic_strength': 65,
                'kc_infrastructure_score': 60,
                'kc_market_access': 63
            },
        }
        
        # Default for unknown counties
        default_values = {
            'kc_crime_safety': 70,
            'kc_development_activity': 50,
            'kc_demographic_strength': 60,
            'kc_infrastructure_score': 65,
            'kc_market_access': 61
        }
        
        enhanced_count = 0
        
        # Apply county-specific defaults
        for county, defaults in county_defaults.items():
            mask = (business_df['county'].str.contains(county, case=False, na=False)) & \
                   (business_df['kc_enhancement_source'] == 'none')
            
            if mask.sum() > 0:
                for col, value in defaults.items():
                    business_df.loc[mask, col] = value
                
                business_df.loc[mask, 'kc_enhancement_source'] = 'county'
                enhanced_count += mask.sum()
        
        # Apply general defaults to remaining
        remaining = business_df['kc_enhancement_source'] == 'none'
        if remaining.sum() > 0:
            for col, value in default_values.items():
                business_df.loc[remaining, col] = value
            
            business_df.loc[remaining, 'kc_enhancement_source'] = 'default'
            enhanced_count += remaining.sum()
        
        logger.info(f"  Enhanced {enhanced_count} businesses with county/default features")
        return enhanced_count
    
    def _calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional derived features from KC data"""
        logger.info("Calculating derived features...")
        
        # 1. Location Quality Index (combination of safety and infrastructure)
        df['kc_location_quality'] = (
            0.4 * df['kc_crime_safety'] +
            0.3 * df['kc_infrastructure_score'] +
            0.3 * df['kc_market_access']
        )
        
        # 2. Growth Potential (development + demographics)
        df['kc_growth_potential'] = (
            0.5 * df['kc_development_activity'] +
            0.5 * df['kc_demographic_strength']
        )
        
        # 3. Business Environment Score (overall KC attractiveness)
        df['kc_business_environment'] = (
            0.25 * df['kc_crime_safety'] +
            0.25 * df['kc_development_activity'] +
            0.25 * df['kc_demographic_strength'] +
            0.25 * df['kc_infrastructure_score']
        )
        
        # 4. Risk-adjusted opportunity (high growth with good safety)
        df['kc_risk_adjusted_opportunity'] = (
            df['kc_growth_potential'] * (df['kc_crime_safety'] / 100)
        )
        
        # 5. Cluster readiness (how ready is this location for clustering)
        df['kc_cluster_readiness'] = (
            0.3 * df['kc_infrastructure_score'] +
            0.3 * df['kc_market_access'] +
            0.2 * df['kc_demographic_strength'] +
            0.2 * df['kc_business_environment']
        )
        
        logger.info(f"  Added 5 derived features")
        
        return df
    
    def _log_enhancement_stats(self, df: pd.DataFrame):
        """Log detailed enhancement statistics"""
        logger.info("\n=== Enhancement Statistics ===")
        
        # Source distribution
        source_counts = df['kc_enhancement_source'].value_counts()
        logger.info("Enhancement sources:")
        for source, count in source_counts.items():
            pct = count / len(df) * 100
            logger.info(f"  {source}: {count:,} ({pct:.1f}%)")
        
        # Feature statistics
        logger.info("\nFeature statistics:")
        for col in self.feature_columns:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                min_val = df[col].min()
                max_val = df[col].max()
                logger.info(f"  {col}:")
                logger.info(f"    Mean: {mean_val:.1f}, Std: {std_val:.1f}")
                logger.info(f"    Range: {min_val:.1f} - {max_val:.1f}")
        
        # Quality metrics
        high_quality = (df['kc_business_environment'] > 75).sum()
        logger.info(f"\nHigh-quality locations (environment > 75): {high_quality:,} ({high_quality/len(df)*100:.1f}%)")
        
        direct_enhanced = (df['kc_enhancement_source'] == 'direct').sum()
        logger.info(f"Directly verified businesses: {direct_enhanced:,} ({direct_enhanced/len(df)*100:.1f}%)")
    
    def create_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create final ML-ready features from enhanced data"""
        logger.info("Creating ML-ready features...")
        
        # Ensure all KC features are present and normalized
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 60  # Neutral default
            else:
                # Normalize to 0-100 range
                df[col] = df[col].clip(0, 100)
        
        # Create interaction features
        df['kc_safety_x_infrastructure'] = df['kc_crime_safety'] * df['kc_infrastructure_score'] / 100
        df['kc_growth_x_access'] = df['kc_development_activity'] * df['kc_market_access'] / 100
        
        # Create categorical features
        df['kc_tier'] = pd.cut(df['kc_business_environment'], 
                               bins=[0, 60, 75, 90, 100],
                               labels=['low', 'medium', 'high', 'premium'])
        
        # One-hot encode tier
        tier_dummies = pd.get_dummies(df['kc_tier'], prefix='kc_tier')
        df = pd.concat([df, tier_dummies], axis=1)
        
        logger.info(f"  Created {len([c for c in df.columns if c.startswith('kc_')])} KC-related features")
        
        return df
    
    def validate_enhancement(self, df: pd.DataFrame) -> Dict:
        """Validate enhancement quality and completeness"""
        validation = {}
        
        # Check completeness
        validation['total_businesses'] = len(df)
        validation['enhanced'] = (df['kc_enhancement_source'] != 'none').sum()
        validation['enhancement_rate'] = validation['enhanced'] / validation['total_businesses']
        
        # Check feature completeness
        missing_features = {}
        for col in self.feature_columns:
            missing = df[col].isna().sum()
            if missing > 0:
                missing_features[col] = missing
        validation['missing_features'] = missing_features
        
        # Check data quality
        validation['direct_matches'] = (df['kc_enhancement_source'] == 'direct').sum()
        validation['zip_matches'] = (df['kc_enhancement_source'] == 'zip').sum()
        validation['quality_score'] = (
            validation['direct_matches'] * 1.0 +
            validation['zip_matches'] * 0.7
        ) / validation['total_businesses']
        
        # Check value distributions
        validation['feature_ranges'] = {}
        for col in self.feature_columns:
            if col in df.columns:
                validation['feature_ranges'][col] = {
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'mean': df[col].mean(),
                    'std': df[col].std()
                }
        
        return validation
