#!/usr/bin/env python3
"""Extract and process data from data.kcmo.org API"""

import pandas as pd
import numpy as np
import requests
import logging
from typing import Dict, List, Optional
import json
from pathlib import Path
import time
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)

from config import Config

class KCDataExtractor:
    """Extract business-relevant data from Kansas City Open Data Portal"""
    
    def __init__(self, cache_dir: str = 'cache/kc_data'):
        self.base_url = 'https://data.kcmo.org/resource'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(days=7)  # Cache for 7 days
        # Optional Socrata app token to reduce throttling
        self.config = Config()
        self.headers = {}
        if getattr(self.config, 'KCMO_APP_TOKEN', None):
            self.headers['X-App-Token'] = self.config.KCMO_APP_TOKEN
        
        # Dataset endpoints from data.kcmo.org
        # Allow override via env vars; fall back to known IDs; include alternates where some are retired
        self.datasets = {
            'business_licenses': os.getenv('KCMO_DATASET_BUSINESS_LICENSES', 'e5aw-jx7h'),  # Active business licenses (alt: pnm4-68wg)
            'crime_2024': os.getenv('KCMO_DATASET_CRIME', 'c44e-5qd5'),         # 2024 crime data (override allowed)
            'development_activity': 'rdqb-na8f', # Building permits
            'demographics': 'demographic_api',   # Census data endpoint
            'property_data': 'property_api',     # Property information
        }
        
    def extract_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Extract all relevant KC datasets"""
        logger.info("Starting Kansas City data extraction...")
        
        datasets = {}
        
        # 1. Business Licenses
        datasets['licenses'] = self.extract_business_licenses()
        
        # 2. Crime Data (for safety scoring)
        datasets['crime'] = self.extract_crime_data()
        
        # 3. Development Activity (building permits)
        datasets['development'] = self.extract_development_activity()
        
        # 4. Demographics (simulated - would use Census API in production)
        datasets['demographics'] = self.extract_demographics()
        
        # 5. Property/Infrastructure (simulated)
        datasets['infrastructure'] = self.extract_infrastructure()
        
        logger.info(f"Extraction complete. Retrieved {len(datasets)} datasets")
        
        return datasets
    
    def extract_business_licenses(self, limit: int = 50000) -> pd.DataFrame:
        """Extract active business licenses from KC"""
        cache_file = self.cache_dir / 'business_licenses.csv'
        
        # Check cache
        if self._is_cache_valid(cache_file):
            logger.info("Loading business licenses from cache")
            return pd.read_csv(cache_file)
        
        logger.info("Fetching business licenses from data.kcmo.org...")
        
        # Build query with pagination
        all_records = []
        offset = 0
        batch_size = 1000
        
        while offset < limit:
            params = {
                '$limit': batch_size,
                '$offset': offset,
                '$where': "status='Active'",
                '$select': 'business_name,business_address,license_type,issue_date,expiration_date,naics_code'
            }
            
            try:
                dsid = self.datasets['business_licenses']
                url = f"{self.base_url}/{dsid}.json"
                response = requests.get(url, params=params, headers=self.headers, timeout=30)
                if response.status_code != 200:
                    # Try alternate known dataset id
                    alt = 'pnm4-68wg'
                    if dsid != alt:
                        logger.warning(f"API returned status {response.status_code} for {dsid}; retrying {alt}")
                        response = requests.get(f"{self.base_url}/{alt}.json", params=params, headers=self.headers, timeout=30)
                if response.status_code == 200:
                    batch = response.json()
                    if not batch:
                        break
                    all_records.extend(batch)
                    offset += batch_size
                    logger.info(f"  Fetched {len(all_records)} licenses...")
                    time.sleep(0.1)  # Rate limiting
                else:
                    logger.warning(f"License API returned status {response.status_code}; stopping")
                    break
            except Exception as e:
                logger.error(f"Error fetching business licenses: {e}")
                break
        
        if all_records:
            df = pd.DataFrame(all_records)
            
            # Clean and process
            df['business_name'] = df['business_name'].str.strip().str.title()
            df['license_active'] = 1
            df['years_in_business'] = pd.to_datetime('today').year - pd.to_datetime(df['issue_date']).dt.year
            
            # Save to cache
            df.to_csv(cache_file, index=False)
            logger.info(f"  Retrieved {len(df)} active business licenses")
            
            return df
        else:
            logger.warning("No business license data retrieved")
            return pd.DataFrame()
    
    def extract_crime_data(self, limit: int = 10000) -> pd.DataFrame:
        """Extract recent crime data for safety scoring"""
        cache_file = self.cache_dir / 'crime_data.csv'
        
        if self._is_cache_valid(cache_file):
            logger.info("Loading crime data from cache")
            return pd.read_csv(cache_file)
        
        logger.info("Fetching crime data from data.kcmo.org...")
        
        # Get crimes from last 90 days
        date_90_days_ago = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        
        params = {
            '$limit': limit,
            '$where': f"report_date >= '{date_90_days_ago}'",
            '$select': 'report_date,offense,category,location,zip_code,latitude,longitude'
        }
        
        try:
            dsid = self.datasets['crime_2024']
            response = requests.get(f"{self.base_url}/{dsid}.json", params=params, headers=self.headers, timeout=30)
            
            if response.status_code == 200:
                crimes = response.json()
                df = pd.DataFrame(crimes)
                
                # Process crime data
                df['crime_severity'] = df['category'].map({
                    'Violent Crime': 3,
                    'Property Crime': 2,
                    'Quality of Life': 1
                }).fillna(1)
                
                # Aggregate by zip code
                crime_by_zip = df.groupby('zip_code').agg({
                    'offense': 'count',
                    'crime_severity': 'mean'
                }).rename(columns={'offense': 'crime_count'})
                
                crime_by_zip.to_csv(cache_file)
                logger.info(f"  Retrieved crime data for {len(crime_by_zip)} zip codes")
                
                return crime_by_zip
            else:
                # Try alternative dataset id if provided via env ALT
                alt = os.getenv('KCMO_DATASET_CRIME_ALT')
                if alt and alt != dsid:
                    logger.warning(f"Crime API returned status {response.status_code} for {dsid}; retrying {alt}")
                    response = requests.get(f"{self.base_url}/{alt}.json", params=params, headers=self.headers, timeout=30)
                    if response.status_code == 200:
                        crimes = response.json()
                        df = pd.DataFrame(crimes)
                        df['crime_severity'] = df['category'].map({
                            'Violent Crime': 3,
                            'Property Crime': 2,
                            'Quality of Life': 1
                        }).fillna(1)
                        crime_by_zip = df.groupby('zip_code').agg({'offense': 'count','crime_severity': 'mean'}).rename(columns={'offense': 'crime_count'})
                        crime_by_zip.to_csv(cache_file)
                        logger.info(f"  Retrieved crime data for {len(crime_by_zip)} zip codes (alt dataset)")
                        return crime_by_zip
                logger.warning(f"Crime API returned status {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching crime data: {e}")
            return pd.DataFrame()
    
    def extract_development_activity(self, limit: int = 5000) -> pd.DataFrame:
        """Extract building permits and development activity"""
        cache_file = self.cache_dir / 'development_activity.csv'
        
        if self._is_cache_valid(cache_file):
            logger.info("Loading development data from cache")
            return pd.read_csv(cache_file)
        
        logger.info("Fetching development activity...")
        
        # Simulate development data (in production, would use real API)
        # Generate realistic KC development patterns
        np.random.seed(42)
        
        zip_codes = ['64111', '64112', '64113', '64114', '64105', '64106', '64108', 
                    '64109', '64110', '64127', '64128', '64129', '64130', '64131']
        
        data = []
        for zip_code in zip_codes:
            # Downtown and Crossroads have more development
            if zip_code in ['64105', '64106', '64108']:
                permit_count = np.random.poisson(50)
                avg_value = np.random.normal(500000, 200000)
            else:
                permit_count = np.random.poisson(20)
                avg_value = np.random.normal(200000, 100000)
            
            data.append({
                'zip_code': zip_code,
                'permit_count': permit_count,
                'avg_permit_value': max(50000, avg_value),
                'development_index': permit_count * avg_value / 1000000  # Development activity index
            })
        
        df = pd.DataFrame(data)
        df.to_csv(cache_file, index=False)
        logger.info(f"  Generated development data for {len(df)} zip codes")
        
        return df
    
    def extract_demographics(self) -> pd.DataFrame:
        """Extract demographic data (simulated - would use Census API)"""
        cache_file = self.cache_dir / 'demographics.csv'
        
        if self._is_cache_valid(cache_file):
            logger.info("Loading demographics from cache")
            return pd.read_csv(cache_file)
        
        logger.info("Generating demographic profiles...")
        
        # Simulate realistic KC demographics
        np.random.seed(42)
        
        zip_data = {
            '64111': {'median_income': 85000, 'population': 15000, 'education_index': 0.8},  # Plaza
            '64112': {'median_income': 75000, 'population': 20000, 'education_index': 0.75}, # Waldo
            '64113': {'median_income': 95000, 'population': 25000, 'education_index': 0.85}, # Leawood area
            '64114': {'median_income': 90000, 'population': 22000, 'education_index': 0.82}, # Prairie Village
            '64105': {'median_income': 45000, 'population': 8000, 'education_index': 0.7},   # Downtown
            '64106': {'median_income': 48000, 'population': 6000, 'education_index': 0.72},  # River Market
            '64108': {'median_income': 42000, 'population': 10000, 'education_index': 0.65}, # Crossroads
            '64127': {'median_income': 35000, 'population': 18000, 'education_index': 0.55}, # Northeast
            '64128': {'median_income': 38000, 'population': 16000, 'education_index': 0.58}, # Historic Northeast
            '64130': {'median_income': 32000, 'population': 20000, 'education_index': 0.52}, # Southeast
        }
        
        data = []
        for zip_code, demo in zip_data.items():
            data.append({
                'zip_code': zip_code,
                'median_income': demo['median_income'],
                'population': demo['population'],
                'education_index': demo['education_index'],
                'workforce_score': demo['education_index'] * 100,
                'market_size': demo['population'] * demo['median_income'] / 1000000
            })
        
        df = pd.DataFrame(data)
        df.to_csv(cache_file, index=False)
        logger.info(f"  Generated demographics for {len(df)} zip codes")
        
        return df
    
    def extract_infrastructure(self) -> pd.DataFrame:
        """Extract infrastructure quality metrics"""
        cache_file = self.cache_dir / 'infrastructure.csv'
        
        if self._is_cache_valid(cache_file):
            logger.info("Loading infrastructure data from cache")
            return pd.read_csv(cache_file)
        
        logger.info("Generating infrastructure scores...")
        
        # Infrastructure quality by area
        infra_data = {
            '64111': {'transit_score': 75, 'highway_access': 90, 'utilities_reliability': 95},
            '64112': {'transit_score': 70, 'highway_access': 85, 'utilities_reliability': 92},
            '64113': {'transit_score': 60, 'highway_access': 95, 'utilities_reliability': 98},
            '64114': {'transit_score': 65, 'highway_access': 90, 'utilities_reliability': 96},
            '64105': {'transit_score': 90, 'highway_access': 95, 'utilities_reliability': 94},
            '64106': {'transit_score': 85, 'highway_access': 90, 'utilities_reliability': 93},
            '64108': {'transit_score': 80, 'highway_access': 92, 'utilities_reliability': 91},
            '64127': {'transit_score': 55, 'highway_access': 70, 'utilities_reliability': 85},
            '64128': {'transit_score': 50, 'highway_access': 75, 'utilities_reliability': 83},
            '64130': {'transit_score': 45, 'highway_access': 80, 'utilities_reliability': 82},
        }
        
        data = []
        for zip_code, scores in infra_data.items():
            overall_score = (scores['transit_score'] + scores['highway_access'] + scores['utilities_reliability']) / 3
            data.append({
                'zip_code': zip_code,
                **scores,
                'infrastructure_score': overall_score
            })
        
        df = pd.DataFrame(data)
        df.to_csv(cache_file, index=False)
        logger.info(f"  Generated infrastructure data for {len(df)} zip codes")
        
        return df
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cache file exists and is recent"""
        if not cache_file.exists():
            return False
        
        # Check age
        file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age = datetime.now() - file_time
        
        return age < self.cache_ttl
    
    def calculate_composite_features(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate composite KC features from raw datasets"""
        logger.info("Calculating composite KC features...")
        
        # Start with zip codes
        zip_codes = set()
        for df in datasets.values():
            if not df.empty and 'zip_code' in df.columns:
                zip_codes.update(df['zip_code'].unique())
        
        features = pd.DataFrame({'zip_code': list(zip_codes)})
        
        # 1. Crime Safety Score (0-100, higher is safer)
        if not datasets['crime'].empty:
            crime = datasets['crime'].reset_index()
            # Invert and normalize crime counts
            max_crime = crime['crime_count'].max()
            crime['safety_score'] = 100 * (1 - crime['crime_count'] / max_crime)
            features = features.merge(crime[['zip_code', 'safety_score']], 
                                    on='zip_code', how='left')
            features['kc_crime_safety'] = features['safety_score'].fillna(80)  # Default medium safety
        else:
            features['kc_crime_safety'] = 80
        
        # 2. Development Activity Score
        if not datasets['development'].empty:
            dev = datasets['development']
            features = features.merge(dev[['zip_code', 'development_index']], 
                                    on='zip_code', how='left')
            # Normalize to 0-100
            max_dev = dev['development_index'].max()
            features['kc_development_activity'] = 100 * features['development_index'] / max_dev
            features['kc_development_activity'] = features['kc_development_activity'].fillna(50)
        else:
            features['kc_development_activity'] = 50
        
        # 3. Demographic Strength
        if not datasets['demographics'].empty:
            demo = datasets['demographics']
            features = features.merge(demo[['zip_code', 'workforce_score', 'market_size']], 
                                    on='zip_code', how='left')
            features['kc_demographic_strength'] = (
                0.6 * features['workforce_score'].fillna(60) + 
                0.4 * features['market_size'].fillna(30)
            )
        else:
            features['kc_demographic_strength'] = 60
        
        # 4. Infrastructure Score
        if not datasets['infrastructure'].empty:
            infra = datasets['infrastructure']
            features = features.merge(infra[['zip_code', 'infrastructure_score']], 
                                    on='zip_code', how='left')
            features['kc_infrastructure_score'] = features['infrastructure_score'].fillna(70)
        else:
            features['kc_infrastructure_score'] = 70
        
        # 5. Market Access (composite of location and infrastructure)
        features['kc_market_access'] = (
            0.5 * features['kc_infrastructure_score'] + 
            0.3 * features['kc_demographic_strength'] +
            0.2 * features['kc_development_activity']
        )
        
        # Clean up temporary columns
        features = features[[col for col in features.columns if col.startswith('kc_') or col == 'zip_code']]
        
        logger.info(f"  Calculated KC features for {len(features)} zip codes")
        logger.info(f"  Feature ranges:")
        for col in features.columns:
            if col.startswith('kc_'):
                logger.info(f"    {col}: {features[col].min():.1f} - {features[col].max():.1f}")
        
        return features
