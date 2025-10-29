#!/usr/bin/env python3
"""
Main pipeline for enhancing business data with Kansas City Open Data features
This adds 5 KC-specific features to our existing 13-feature ML model
"""

import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
import json
import time
from datetime import datetime

# Import our custom modules
from data_preparation.smart_geocoder import SmartGeocoder
from data_preparation.business_indexer import BusinessIndexer
from data_collection.kc_data_extractor import KCDataExtractor
from enhancement.selective_enhancer import SelectiveEnhancer

# Setup logging
# Ensure logs directory exists and configure robust logging
_handlers = [logging.StreamHandler()]
try:
    _log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(_log_dir, exist_ok=True)
    _handlers.insert(0, logging.FileHandler(os.path.join(_log_dir, 'kc_enhancement.log')))
except Exception:
    # Fall back to console-only logging if file handler cannot be created
    pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=_handlers
)

logger = logging.getLogger(__name__)

class KCEnhancementPipeline:
    """
    Main orchestrator for KC data enhancement pipeline
    
    This pipeline:
    1. Loads our 525K cleaned businesses
    2. Geocodes them efficiently (smart strategy)
    3. Extracts KC Open Data (licenses, crime, development, etc.)
    4. Matches KC businesses to our data
    5. Selectively enhances with KC features
    6. Outputs enhanced dataset ready for ML retraining
    """
    
    def __init__(self, config: dict = None):
        self.config = config or self._default_config()
        
        # Initialize components
        self.geocoder = SmartGeocoder(cache_dir=self.config['cache_dir'] + '/geocode')
        self.indexer = BusinessIndexer()
        self.extractor = KCDataExtractor(cache_dir=self.config['cache_dir'] + '/kc_data')
        self.enhancer = SelectiveEnhancer()
        
        # Track pipeline metrics
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'businesses_processed': 0,
            'features_added': 0,
            'enhancement_rate': 0.0
        }
    
    def _default_config(self) -> dict:
        """Default pipeline configuration"""
        return {
            'input_file': 'data/processed/kc_businesses_cleaned.csv',
            'output_file': 'data/processed/kc_businesses_enhanced.parquet',
            'cache_dir': 'cache',
            'sample_size': None,  # None for full dataset, or number for testing
            'geocoding': {
                'enabled': True,
                'precise_quota': 1000
            },
            'matching': {
                'threshold': 0.8,
                'methods': ['exact', 'phone', 'address', 'fuzzy']
            },
            'features': {
                'kc_crime_safety': True,
                'kc_development_activity': True,
                'kc_demographic_strength': True,
                'kc_infrastructure_score': True,
                'kc_market_access': True
            }
        }
    
    def run(self) -> pd.DataFrame:
        """Execute the full enhancement pipeline"""
        logger.info("=" * 80)
        logger.info("KC ENHANCEMENT PIPELINE STARTING")
        logger.info("=" * 80)
        
        self.metrics['start_time'] = datetime.now()
        
        try:
            # Step 1: Load business data
            businesses = self._load_businesses()
            
            # Step 2: Geocode businesses (if enabled)
            if self.config['geocoding']['enabled']:
                businesses = self._geocode_businesses(businesses)
            
            # Step 3: Extract KC Open Data
            kc_datasets = self._extract_kc_data()
            
            # Step 4: Match businesses
            matches = self._match_businesses(businesses, kc_datasets)
            
            # Step 5: Enhance with KC features
            enhanced_businesses = self._enhance_businesses(businesses, kc_datasets, matches)
            
            # Step 6: Validate and save
            self._save_results(enhanced_businesses)
            
            self.metrics['end_time'] = datetime.now()
            self._report_metrics()
            
            return enhanced_businesses
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _load_businesses(self) -> pd.DataFrame:
        """Load cleaned business data"""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: Loading Business Data")
        logger.info("=" * 60)
        
        input_path = Path(self.config['input_file'])
        
        # Check if file exists
        if not input_path.exists():
            # Try to find it in common locations
            alt_paths = [
                Path('data/kc_businesses_525k.csv'),
                Path('data/processed/businesses_deduped.csv'),
                Path('data/businesses_cleaned.csv')
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    input_path = alt_path
                    logger.info(f"Using alternative path: {input_path}")
                    break
            else:
                raise FileNotFoundError(f"Could not find business data file. Tried: {input_path}, {alt_paths}")
        
        # Load data
        if input_path.suffix == '.parquet':
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)
        
        logger.info(f"Loaded {len(df):,} businesses from {input_path}")
        
        # Apply sample size if configured (for testing)
        if self.config['sample_size']:
            df = df.sample(n=min(self.config['sample_size'], len(df)), random_state=42)
            logger.info(f"Sampled {len(df):,} businesses for testing")
        
        self.metrics['businesses_processed'] = len(df)
        
        # Log data summary
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {', '.join(df.columns[:10])}...")
        logger.info(f"Counties: {df['county'].value_counts().head(5).to_dict()}")
        
        return df
    
    def _geocode_businesses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add geocoding to businesses"""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Geocoding Businesses")
        logger.info("=" * 60)
        
        # Pass through existing coordinates when present (lat/lon or latitude/longitude)
        had_existing = False
        if 'lat' in df.columns and 'lon' in df.columns:
            try:
                lat = pd.to_numeric(df['lat'], errors='coerce')
                lon = pd.to_numeric(df['lon'], errors='coerce')
                use = lat.notna() & lon.notna()
                if use.any():
                    df.loc[use, 'latitude'] = lat[use]
                    df.loc[use, 'longitude'] = lon[use]
                    df.loc[use, 'geo_source'] = 'existing'
                    df.loc[use, 'geo_confidence'] = 0.85
                    had_existing = True
            except Exception:
                pass
        if 'latitude' in df.columns and 'longitude' in df.columns:
            existing_share = (pd.to_numeric(df['latitude'], errors='coerce').notna() &
                              pd.to_numeric(df['longitude'], errors='coerce').notna()).mean()
            if existing_share > 0.9:
                logger.info("Businesses already geocoded, skipping...")
                return df
        
        # Perform smart geocoding for rows still missing coordinates
        df = self.geocoder.geocode_businesses(df)
        
        # Add spatial features
        df = self.geocoder.add_spatial_features(df)
        
        return df
    
    def _extract_kc_data(self) -> dict:
        """Extract data from KC Open Data portal"""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Extracting KC Open Data")
        logger.info("=" * 60)
        
        # Extract all datasets
        datasets = self.extractor.extract_all_datasets()
        
        # Calculate composite features
        kc_features = self.extractor.calculate_composite_features(datasets)
        datasets['composite_features'] = kc_features
        
        # Log extraction summary
        for name, data in datasets.items():
            if isinstance(data, pd.DataFrame):
                logger.info(f"  {name}: {len(data)} records")
        
        return datasets
    
    def _match_businesses(self, businesses: pd.DataFrame, kc_datasets: dict) -> list:
        """Match KC businesses to our dataset"""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Matching Businesses")
        logger.info("=" * 60)
        
        # Build index from our businesses
        logger.info("Building business index...")
        self.indexer.build_indices(businesses)
        
        # Match KC licensed businesses
        matches = []
        if 'licenses' in kc_datasets and not kc_datasets['licenses'].empty:
            kc_licenses = kc_datasets['licenses']
            
            # Prepare KC data for matching
            kc_licenses['name'] = kc_licenses.get('business_name', '')
            kc_licenses['address'] = kc_licenses.get('business_address', '')
            
            # Find matches
            logger.info(f"Matching {len(kc_licenses)} KC licensed businesses...")
            matches = self.indexer.find_matches(
                kc_licenses, 
                threshold=self.config['matching']['threshold']
            )
        
        return matches
    
    def _enhance_businesses(self, businesses: pd.DataFrame, 
                          kc_datasets: dict, matches: list) -> pd.DataFrame:
        """Enhance businesses with KC features"""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 5: Enhancing with KC Features")
        logger.info("=" * 60)
        
        # Get composite KC features
        kc_features = kc_datasets.get('composite_features', pd.DataFrame())
        
        # Perform selective enhancement
        enhanced = self.enhancer.enhance_businesses(
            businesses,
            kc_features,
            matches
        )
        
        # Create ML-ready features
        enhanced = self.enhancer.create_ml_features(enhanced)
        
        # Track features added
        kc_columns = [col for col in enhanced.columns if col.startswith('kc_')]
        self.metrics['features_added'] = len(kc_columns)
        
        # Validate enhancement
        validation = self.enhancer.validate_enhancement(enhanced)
        self.metrics['enhancement_rate'] = validation['enhancement_rate']
        
        logger.info(f"Enhancement validation:")
        logger.info(f"  Total businesses: {validation['total_businesses']:,}")
        logger.info(f"  Enhanced: {validation['enhanced']:,}")
        logger.info(f"  Enhancement rate: {validation['enhancement_rate']*100:.1f}%")
        logger.info(f"  Quality score: {validation['quality_score']:.3f}")
        
        return enhanced
    
    def _save_results(self, df: pd.DataFrame):
        """Save enhanced dataset"""
        logger.info("\n" + "=" * 60)
        logger.info("STEP 6: Saving Results")
        logger.info("=" * 60)
        
        output_path = Path(self.config['output_file'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV (or parquet if available)
        try:
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved enhanced data to {output_path} (parquet)")
        except ImportError:
            # Fallback to CSV if parquet not available
            csv_path = output_path.with_suffix('.csv')
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved enhanced data to {csv_path} (CSV)")
        
        # Also save a sample as CSV for inspection
        sample_path = output_path.with_suffix('.sample.csv')
        df.head(1000).to_csv(sample_path, index=False)
        logger.info(f"Saved sample (1000 rows) to {sample_path}")
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'total_businesses': len(df),
            'features_added': self.metrics['features_added'],
            'enhancement_rate': self.metrics['enhancement_rate'],
            'kc_columns': [col for col in df.columns if col.startswith('kc_')],
            'config': self.config
        }
        
        metadata_path = output_path.with_suffix('.metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
    
    def _report_metrics(self):
        """Report pipeline metrics"""
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)
        
        duration = (self.metrics['end_time'] - self.metrics['start_time']).total_seconds()
        
        logger.info(f"Total time: {duration:.1f} seconds")
        logger.info(f"Businesses processed: {self.metrics['businesses_processed']:,}")
        logger.info(f"Features added: {self.metrics['features_added']}")
        logger.info(f"Enhancement rate: {self.metrics['enhancement_rate']*100:.1f}%")
        logger.info(f"Processing rate: {self.metrics['businesses_processed']/duration:.0f} businesses/second")


def main():
    """Main entry point"""
    
    # Configure pipeline
    config = {
        'input_file': 'data/processed/kc_businesses_cleaned.csv',
        'output_file': 'data/processed/kc_businesses_enhanced.parquet',
        'cache_dir': 'cache',
        'sample_size': None,  # Set to number for testing (e.g., 10000)
        'geocoding': {
            'enabled': True,
            'precise_quota': 1000
        },
        'matching': {
            'threshold': 0.8,
            'methods': ['exact', 'phone', 'address', 'fuzzy']
        },
        'features': {
            'kc_crime_safety': True,
            'kc_development_activity': True,
            'kc_demographic_strength': True,
            'kc_infrastructure_score': True,
            'kc_market_access': True
        }
    }
    
    # Run pipeline
    pipeline = KCEnhancementPipeline(config)
    enhanced_data = pipeline.run()
    
    logger.info(f"\nEnhanced dataset ready with {enhanced_data.shape[0]} businesses and {enhanced_data.shape[1]} features")
    
    # Display sample of KC features
    kc_cols = [col for col in enhanced_data.columns if col.startswith('kc_')][:5]
    if kc_cols:
        logger.info("\nSample of KC features added:")
        logger.info(enhanced_data[kc_cols].describe())


if __name__ == "__main__":
    main()
