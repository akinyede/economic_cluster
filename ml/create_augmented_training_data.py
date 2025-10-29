#!/usr/bin/env python3
"""
Generate augmented training data with KC features for ML model retraining
This creates training data with 18 features (13 original + 5 KC features)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from typing import Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AugmentedDataGenerator:
    """Generate training data with KC features for enhanced ML models"""
    
    def __init__(self):
        # Original 13 features from current model
        self.original_features = [
            'business_count',
            'total_employees', 
            'total_revenue',
            'avg_business_age',
            'strategic_score',
            'innovation_score',
            'critical_mass',
            'supply_chain_completeness',
            'geographic_density',
            'workforce_diversity',
            'natural_community_score',
            'cluster_synergy',
            'market_position'
        ]
        
        # 5 new KC features
        self.kc_features = [
            'kc_crime_safety',
            'kc_development_activity',
            'kc_demographic_strength',
            'kc_infrastructure_score',
            'kc_market_access'
        ]
        
        # All 18 features
        self.all_features = self.original_features + self.kc_features
        
        # Target variables
        self.targets = ['gdp_impact', 'job_creation', 'roi_percentage']
        
    def generate_training_data(self, n_samples: int = 3000) -> pd.DataFrame:
        """Generate synthetic training data with all 18 features"""
        logger.info(f"Generating {n_samples} training samples with 18 features...")
        
        np.random.seed(42)
        data = []
        
        # Define cluster types with realistic characteristics
        cluster_types = [
            'technology', 'biosciences', 'logistics', 
            'manufacturing', 'professional_services', 'mixed'
        ]
        
        for i in range(n_samples):
            cluster_type = np.random.choice(cluster_types)
            sample = self._generate_cluster_sample(cluster_type)
            data.append(sample)
        
        df = pd.DataFrame(data)
        
        # Calculate targets based on features
        df = self._calculate_targets(df)
        
        logger.info(f"Generated {len(df)} samples with {len(self.all_features)} features")
        
        return df
    
    def _generate_cluster_sample(self, cluster_type: str) -> Dict:
        """Generate a single cluster sample with realistic feature values"""
        
        sample = {}
        
        # Base characteristics by cluster type
        if cluster_type == 'technology':
            # Tech clusters: high innovation, good infrastructure
            sample['business_count'] = np.random.randint(20, 80)
            sample['total_employees'] = sample['business_count'] * np.random.randint(50, 200)
            sample['total_revenue'] = sample['total_employees'] * np.random.uniform(200000, 500000)
            sample['avg_business_age'] = np.random.uniform(5, 15)
            sample['strategic_score'] = np.random.uniform(70, 95)
            sample['innovation_score'] = np.random.uniform(75, 95)
            sample['critical_mass'] = np.random.uniform(65, 90)
            sample['supply_chain_completeness'] = np.random.uniform(0.6, 0.9)
            sample['geographic_density'] = np.random.uniform(0.7, 0.95)
            sample['workforce_diversity'] = np.random.uniform(0.65, 0.9)
            sample['natural_community_score'] = np.random.uniform(0.6, 0.85)
            sample['cluster_synergy'] = np.random.uniform(70, 90)
            sample['market_position'] = np.random.uniform(65, 85)
            # KC features - tech clusters in good areas
            sample['kc_crime_safety'] = np.random.uniform(75, 90)
            sample['kc_development_activity'] = np.random.uniform(70, 90)
            sample['kc_demographic_strength'] = np.random.uniform(75, 90)
            sample['kc_infrastructure_score'] = np.random.uniform(80, 95)
            sample['kc_market_access'] = np.random.uniform(75, 90)
            
        elif cluster_type == 'biosciences':
            # Bio clusters: high innovation, critical mass important
            sample['business_count'] = np.random.randint(15, 60)
            sample['total_employees'] = sample['business_count'] * np.random.randint(80, 250)
            sample['total_revenue'] = sample['total_employees'] * np.random.uniform(250000, 600000)
            sample['avg_business_age'] = np.random.uniform(8, 20)
            sample['strategic_score'] = np.random.uniform(75, 95)
            sample['innovation_score'] = np.random.uniform(80, 98)
            sample['critical_mass'] = np.random.uniform(70, 95)
            sample['supply_chain_completeness'] = np.random.uniform(0.7, 0.95)
            sample['geographic_density'] = np.random.uniform(0.6, 0.85)
            sample['workforce_diversity'] = np.random.uniform(0.7, 0.95)
            sample['natural_community_score'] = np.random.uniform(0.65, 0.9)
            sample['cluster_synergy'] = np.random.uniform(75, 95)
            sample['market_position'] = np.random.uniform(70, 90)
            # KC features - bio clusters need good infrastructure
            sample['kc_crime_safety'] = np.random.uniform(70, 85)
            sample['kc_development_activity'] = np.random.uniform(65, 85)
            sample['kc_demographic_strength'] = np.random.uniform(80, 95)
            sample['kc_infrastructure_score'] = np.random.uniform(75, 90)
            sample['kc_market_access'] = np.random.uniform(70, 85)
            
        elif cluster_type == 'logistics':
            # Logistics: location and infrastructure critical
            sample['business_count'] = np.random.randint(30, 100)
            sample['total_employees'] = sample['business_count'] * np.random.randint(100, 300)
            sample['total_revenue'] = sample['total_employees'] * np.random.uniform(150000, 350000)
            sample['avg_business_age'] = np.random.uniform(10, 25)
            sample['strategic_score'] = np.random.uniform(65, 85)
            sample['innovation_score'] = np.random.uniform(50, 70)
            sample['critical_mass'] = np.random.uniform(60, 80)
            sample['supply_chain_completeness'] = np.random.uniform(0.75, 0.95)
            sample['geographic_density'] = np.random.uniform(0.5, 0.75)
            sample['workforce_diversity'] = np.random.uniform(0.5, 0.7)
            sample['natural_community_score'] = np.random.uniform(0.7, 0.9)
            sample['cluster_synergy'] = np.random.uniform(65, 80)
            sample['market_position'] = np.random.uniform(60, 80)
            # KC features - logistics needs highway access
            sample['kc_crime_safety'] = np.random.uniform(60, 75)
            sample['kc_development_activity'] = np.random.uniform(60, 80)
            sample['kc_demographic_strength'] = np.random.uniform(60, 75)
            sample['kc_infrastructure_score'] = np.random.uniform(85, 98)
            sample['kc_market_access'] = np.random.uniform(80, 95)
            
        elif cluster_type == 'manufacturing':
            # Manufacturing: workforce and infrastructure important
            sample['business_count'] = np.random.randint(20, 70)
            sample['total_employees'] = sample['business_count'] * np.random.randint(150, 400)
            sample['total_revenue'] = sample['total_employees'] * np.random.uniform(180000, 400000)
            sample['avg_business_age'] = np.random.uniform(15, 35)
            sample['strategic_score'] = np.random.uniform(60, 80)
            sample['innovation_score'] = np.random.uniform(55, 75)
            sample['critical_mass'] = np.random.uniform(55, 75)
            sample['supply_chain_completeness'] = np.random.uniform(0.65, 0.85)
            sample['geographic_density'] = np.random.uniform(0.4, 0.65)
            sample['workforce_diversity'] = np.random.uniform(0.45, 0.65)
            sample['natural_community_score'] = np.random.uniform(0.6, 0.8)
            sample['cluster_synergy'] = np.random.uniform(60, 75)
            sample['market_position'] = np.random.uniform(55, 75)
            # KC features - manufacturing in industrial areas
            sample['kc_crime_safety'] = np.random.uniform(55, 70)
            sample['kc_development_activity'] = np.random.uniform(50, 70)
            sample['kc_demographic_strength'] = np.random.uniform(55, 70)
            sample['kc_infrastructure_score'] = np.random.uniform(70, 85)
            sample['kc_market_access'] = np.random.uniform(65, 80)
            
        elif cluster_type == 'professional_services':
            # Professional services: downtown locations, high workforce quality
            sample['business_count'] = np.random.randint(40, 120)
            sample['total_employees'] = sample['business_count'] * np.random.randint(30, 100)
            sample['total_revenue'] = sample['total_employees'] * np.random.uniform(180000, 400000)
            sample['avg_business_age'] = np.random.uniform(8, 20)
            sample['strategic_score'] = np.random.uniform(65, 85)
            sample['innovation_score'] = np.random.uniform(60, 80)
            sample['critical_mass'] = np.random.uniform(50, 70)
            sample['supply_chain_completeness'] = np.random.uniform(0.5, 0.7)
            sample['geographic_density'] = np.random.uniform(0.8, 0.95)
            sample['workforce_diversity'] = np.random.uniform(0.7, 0.9)
            sample['natural_community_score'] = np.random.uniform(0.5, 0.7)
            sample['cluster_synergy'] = np.random.uniform(60, 75)
            sample['market_position'] = np.random.uniform(60, 75)
            # KC features - downtown/plaza locations
            sample['kc_crime_safety'] = np.random.uniform(70, 85)
            sample['kc_development_activity'] = np.random.uniform(75, 90)
            sample['kc_demographic_strength'] = np.random.uniform(75, 90)
            sample['kc_infrastructure_score'] = np.random.uniform(75, 90)
            sample['kc_market_access'] = np.random.uniform(80, 95)
            
        else:  # mixed
            # Mixed clusters: average across all dimensions
            sample['business_count'] = np.random.randint(25, 80)
            sample['total_employees'] = sample['business_count'] * np.random.randint(50, 150)
            sample['total_revenue'] = sample['total_employees'] * np.random.uniform(150000, 300000)
            sample['avg_business_age'] = np.random.uniform(10, 20)
            sample['strategic_score'] = np.random.uniform(55, 75)
            sample['innovation_score'] = np.random.uniform(50, 70)
            sample['critical_mass'] = np.random.uniform(45, 65)
            sample['supply_chain_completeness'] = np.random.uniform(0.4, 0.65)
            sample['geographic_density'] = np.random.uniform(0.5, 0.75)
            sample['workforce_diversity'] = np.random.uniform(0.5, 0.7)
            sample['natural_community_score'] = np.random.uniform(0.4, 0.6)
            sample['cluster_synergy'] = np.random.uniform(50, 65)
            sample['market_position'] = np.random.uniform(50, 65)
            # KC features - mixed locations
            sample['kc_crime_safety'] = np.random.uniform(60, 80)
            sample['kc_development_activity'] = np.random.uniform(50, 75)
            sample['kc_demographic_strength'] = np.random.uniform(60, 80)
            sample['kc_infrastructure_score'] = np.random.uniform(65, 80)
            sample['kc_market_access'] = np.random.uniform(60, 80)
        
        sample['cluster_type'] = cluster_type
        
        # Add some noise to make data more realistic
        for feature in self.all_features:
            if feature in sample:
                sample[feature] = sample[feature] * np.random.uniform(0.95, 1.05)
        
        return sample
    
    def _calculate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate target variables based on features with KC influence"""
        
        # GDP Impact - influenced by KC market access and infrastructure
        base_gdp = df['total_revenue'] * 1.85  # Base multiplier from paper
        
        # KC features boost GDP
        kc_boost = (
            0.15 * (df['kc_market_access'] / 100) +
            0.10 * (df['kc_infrastructure_score'] / 100) +
            0.05 * (df['kc_demographic_strength'] / 100)
        )
        
        # Critical mass and innovation boost
        innovation_boost = 0.2 * (df['innovation_score'] / 100) * (df['critical_mass'] / 100)
        
        # Apply conservative friction (0.75-0.85)
        friction = 0.75 + 0.1 * (df['strategic_score'] / 100)
        
        df['gdp_impact'] = base_gdp * (1 + kc_boost + innovation_boost) * friction
        
        # Job Creation - influenced by KC demographic strength and development
        base_jobs = df['total_employees'] * 2.2  # Base multiplier from paper
        
        # KC features influence job growth
        kc_job_boost = (
            0.12 * (df['kc_demographic_strength'] / 100) +
            0.08 * (df['kc_development_activity'] / 100) +
            0.05 * (df['kc_crime_safety'] / 100)
        )
        
        # Critical mass affects job creation
        mass_factor = 0.95 + 0.25 * (df['critical_mass'] / 100)
        
        df['job_creation'] = (base_jobs * (1 + kc_job_boost) * mass_factor * friction).astype(int)
        
        # ROI Percentage - influenced by all KC features
        base_roi = 5 + 20 * (df['strategic_score'] / 100)  # 5-25% base
        
        # KC features affect ROI
        kc_roi_boost = (
            2 * (df['kc_market_access'] / 100) +
            1.5 * (df['kc_infrastructure_score'] / 100) +
            1 * (df['kc_development_activity'] / 100) +
            0.5 * (df['kc_crime_safety'] / 100)
        )
        
        # Sustainability affects long-term ROI
        sustainability_factor = 1 + 0.3 * (df['critical_mass'] / 100)
        
        df['roi_percentage'] = np.clip(
            base_roi + kc_roi_boost * sustainability_factor,
            5, 30  # Realistic bounds from paper
        )
        
        return df
    
    def save_training_data(self, df: pd.DataFrame, output_dir: str = 'data/ml_training'):
        """Save training data in multiple formats"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save full dataset
        full_path = output_path / 'training_data_18features.csv'
        df.to_csv(full_path, index=False)
        logger.info(f"Saved full training data to {full_path}")
        
        # Save feature matrix and targets separately
        X = df[self.all_features]
        y = df[self.targets]
        
        X.to_csv(output_path / 'X_train_18features.csv', index=False)
        y.to_csv(output_path / 'y_train_18features.csv', index=False)
        
        # Save metadata
        metadata = {
            'n_samples': len(df),
            'n_features': len(self.all_features),
            'features': self.all_features,
            'kc_features': self.kc_features,
            'targets': self.targets,
            'feature_stats': {
                col: {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
                for col in self.all_features
            },
            'target_stats': {
                col: {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
                for col in self.targets
            }
        }
        
        with open(output_path / 'training_metadata_18features.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved training metadata")
        
        # Create versions for comparison
        self._create_comparison_datasets(df, output_path)
        
        return full_path
    
    def _create_comparison_datasets(self, df: pd.DataFrame, output_path: Path):
        """Create datasets for comparing 13 vs 18 feature models"""
        
        # 13-feature version (original)
        df_13 = df[self.original_features + self.targets]
        df_13.to_csv(output_path / 'training_data_13features.csv', index=False)
        logger.info("Created 13-feature comparison dataset")
        
        # KC-only version (5 KC features + original basics)
        basic_features = ['business_count', 'total_employees', 'total_revenue']
        df_kc = df[basic_features + self.kc_features + self.targets]
        df_kc.to_csv(output_path / 'training_data_kc_only.csv', index=False)
        logger.info("Created KC-only feature dataset")
        
    def analyze_feature_importance(self, df: pd.DataFrame):
        """Analyze correlation between features and targets"""
        logger.info("\n" + "="*60)
        logger.info("FEATURE IMPORTANCE ANALYSIS")
        logger.info("="*60)
        
        # Calculate correlations
        for target in self.targets:
            logger.info(f"\nCorrelations with {target}:")
            
            correlations = {}
            for feature in self.all_features:
                corr = df[feature].corr(df[target])
                correlations[feature] = corr
            
            # Sort by absolute correlation
            sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Show top 10
            for feature, corr in sorted_corr[:10]:
                logger.info(f"  {feature:30s}: {corr:+.3f}")
            
            # Highlight KC features
            logger.info("\n  KC Feature Correlations:")
            for feature in self.kc_features:
                logger.info(f"    {feature:30s}: {correlations[feature]:+.3f}")


def main():
    """Generate augmented training data"""
    logger.info("="*80)
    logger.info("GENERATING AUGMENTED TRAINING DATA WITH KC FEATURES")
    logger.info("="*80)
    
    # Initialize generator
    generator = AugmentedDataGenerator()
    
    # Generate training data
    df = generator.generate_training_data(n_samples=3000)
    
    # Analyze feature importance
    generator.analyze_feature_importance(df)
    
    # Save training data
    output_path = generator.save_training_data(df)
    
    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("TRAINING DATA SUMMARY")
    logger.info("="*60)
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Total features: {len(generator.all_features)}")
    logger.info(f"Original features: {len(generator.original_features)}")
    logger.info(f"KC features: {len(generator.kc_features)}")
    logger.info(f"\nTarget ranges:")
    logger.info(f"  GDP Impact: ${df['gdp_impact'].min()/1e6:.1f}M - ${df['gdp_impact'].max()/1e6:.1f}M")
    logger.info(f"  Job Creation: {df['job_creation'].min():,} - {df['job_creation'].max():,}")
    logger.info(f"  ROI: {df['roi_percentage'].min():.1f}% - {df['roi_percentage'].max():.1f}%")
    
    logger.info(f"\nData saved to: {output_path}")
    logger.info("\nNext step: Run train_enhanced_models.py to train the 18-feature models")


if __name__ == "__main__":
    main()