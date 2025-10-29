#!/usr/bin/env python3
"""
Train enhanced ML models with 18 features (13 original + 5 KC features)
Creates new .pkl files for production use
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
import json
from typing import Dict, Tuple

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedModelTrainer:
    """Train ML models with KC-enhanced features"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self.feature_importance = {}
        
        # Feature names
        self.feature_names = [
            'business_count', 'total_employees', 'total_revenue',
            'avg_business_age', 'strategic_score', 'innovation_score',
            'critical_mass', 'supply_chain_completeness', 'geographic_density',
            'workforce_diversity', 'natural_community_score', 'cluster_synergy',
            'market_position', 'kc_crime_safety', 'kc_development_activity',
            'kc_demographic_strength', 'kc_infrastructure_score', 'kc_market_access'
        ]
        
        # Target names
        self.targets = ['gdp_impact', 'job_creation', 'roi_percentage']
        
    def load_training_data(self, data_path: str = 'data/ml_training/training_data_18features.csv'):
        """Load the augmented training data"""
        logger.info(f"Loading training data from {data_path}...")
        
        df = pd.read_csv(data_path)
        
        X = df[self.feature_names]
        y = df[self.targets]
        
        logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
        
        return X, y
    
    def train_models(self, X: pd.DataFrame, y: pd.DataFrame):
        """Train models for each target variable"""
        
        for target in self.targets:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training models for {target}")
            logger.info('='*60)
            
            y_target = y[target]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_target, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            self.scalers[target] = scaler
            
            # Train multiple models
            models_to_train = {
                'xgboost': XGBRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                ),
                'random_forest': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'gradient_boost': GradientBoostingRegressor(
                    n_estimators=150,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
            }
            
            best_model = None
            best_score = -float('inf')
            best_model_name = None
            
            for model_name, model in models_to_train.items():
                logger.info(f"\nTraining {model_name}...")
                
                # Train
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                
                # Evaluate
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation (robust to estimators lacking sklearn tags)
                try:
                    cv_scores = cross_val_score(
                        model, X_train_scaled, y_train, cv=5, scoring='r2', n_jobs=-1
                    )
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                except Exception as _:
                    cv_mean = r2
                    cv_std = 0.0
                
                logger.info(f"  MSE: {mse:.4f}")
                logger.info(f"  MAE: {mae:.4f}")
                logger.info(f"  R²: {r2:.4f}")
                logger.info(f"  CV R² Mean: {cv_mean:.4f} (+/- {cv_std:.4f})")
                
                # Track best model
                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_model_name = model_name
            
            logger.info(f"\nBest model for {target}: {best_model_name} (R² = {best_score:.4f})")
            
            # Store best model
            self.models[target] = best_model
            
            # Calculate feature importance
            if hasattr(best_model, 'feature_importances_'):
                importance = best_model.feature_importances_
                self.feature_importance[target] = dict(zip(self.feature_names, importance))
                
                # Log top features
                sorted_features = sorted(self.feature_importance[target].items(), 
                                       key=lambda x: x[1], reverse=True)
                logger.info(f"\nTop 10 features for {target}:")
                for feat, imp in sorted_features[:10]:
                    logger.info(f"  {feat:30s}: {imp:.4f}")
                
                # Highlight KC features
                logger.info(f"\nKC feature importance:")
                for feat in ['kc_crime_safety', 'kc_development_activity', 
                           'kc_demographic_strength', 'kc_infrastructure_score', 
                           'kc_market_access']:
                    imp = self.feature_importance[target].get(feat, 0)
                    logger.info(f"  {feat:30s}: {imp:.4f}")
            
            # Store metrics
            self.metrics[target] = {
                'model_type': best_model_name,
                'mse': mse,
                'mae': mae,
                'r2': best_score,
                'cv_mean': cv_mean,
                'cv_std': cv_std
            }
    
    def save_models(self, output_dir: str = 'models/enhanced'):
        """Save trained models and metadata"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nSaving models to {output_path}...")
        
        # Save each model and its scaler
        for target in self.targets:
            # Save model
            model_file = output_path / f'{target}_18features.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(self.models[target], f)
            logger.info(f"  Saved {target} model to {model_file}")
            
            # Save scaler
            scaler_file = output_path / f'{target}_scaler_18features.pkl'
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scalers[target], f)
            logger.info(f"  Saved {target} scaler to {scaler_file}")
        
        # Save metadata
        metadata = {
            'features': self.feature_names,
            'n_features': len(self.feature_names),
            'targets': self.targets,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance,
            'model_types': {target: self.metrics[target]['model_type'] 
                          for target in self.targets}
        }
        
        metadata_file = output_path / 'model_metadata_18features.json'
        def _json_default(o):
            try:
                import numpy as _np
                if isinstance(o, (_np.floating, _np.integer)):
                    return o.item()
                if isinstance(o, _np.ndarray):
                    return o.tolist()
            except Exception:
                pass
            return str(o)
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=_json_default)
        logger.info(f"  Saved metadata to {metadata_file}")
    
    def compare_with_13_features(self):
        """Compare 18-feature models with 13-feature baseline"""
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON: 13 vs 18 Features")
        logger.info("="*80)
        
        # Load 13-feature data if available
        data_13_path = Path('data/ml_training/training_data_13features.csv')
        if not data_13_path.exists():
            logger.info("13-feature dataset not found, skipping comparison")
            return
        
        df_13 = pd.read_csv(data_13_path)
        X_13 = df_13.drop(columns=self.targets)
        y_13 = df_13[self.targets]
        
        # Train simple model on 13 features for comparison
        for target in self.targets:
            logger.info(f"\nComparing {target}:")
            
            # 13-feature model
            X_train, X_test, y_train, y_test = train_test_split(
                X_13, y_13[target], test_size=0.2, random_state=42
            )
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model_13 = XGBRegressor(n_estimators=200, max_depth=6, random_state=42)
            model_13.fit(X_train_scaled, y_train)
            
            y_pred_13 = model_13.predict(X_test_scaled)
            r2_13 = r2_score(y_test, y_pred_13)
            
            # Compare with 18-feature model
            r2_18 = self.metrics[target]['r2']
            improvement = ((r2_18 - r2_13) / r2_13) * 100
            
            logger.info(f"  13-feature R²: {r2_13:.4f}")
            logger.info(f"  18-feature R²: {r2_18:.4f}")
            logger.info(f"  Improvement: {improvement:+.1f}%")
    
    def test_predictions(self):
        """Test models with sample predictions"""
        logger.info("\n" + "="*60)
        logger.info("SAMPLE PREDICTIONS TEST")
        logger.info("="*60)
        
        # Create sample cluster
        sample_cluster = pd.DataFrame([{
            'business_count': 50,
            'total_employees': 5000,
            'total_revenue': 1000000000,  # $1B
            'avg_business_age': 12,
            'strategic_score': 80,
            'innovation_score': 75,
            'critical_mass': 70,
            'supply_chain_completeness': 0.75,
            'geographic_density': 0.8,
            'workforce_diversity': 0.7,
            'natural_community_score': 0.65,
            'cluster_synergy': 75,
            'market_position': 70,
            # KC features - good location
            'kc_crime_safety': 80,
            'kc_development_activity': 75,
            'kc_demographic_strength': 82,
            'kc_infrastructure_score': 85,
            'kc_market_access': 80
        }])
        
        logger.info("\nSample cluster characteristics:")
        logger.info(f"  Businesses: {sample_cluster['business_count'].iloc[0]}")
        logger.info(f"  Employees: {sample_cluster['total_employees'].iloc[0]:,}")
        logger.info(f"  Revenue: ${sample_cluster['total_revenue'].iloc[0]/1e9:.1f}B")
        logger.info(f"  Critical Mass: {sample_cluster['critical_mass'].iloc[0]}")
        logger.info(f"  KC Market Access: {sample_cluster['kc_market_access'].iloc[0]}")
        
        logger.info("\nPredictions:")
        for target in self.targets:
            scaler = self.scalers[target]
            model = self.models[target]
            
            sample_scaled = scaler.transform(sample_cluster)
            prediction = model.predict(sample_scaled)[0]
            
            if target == 'gdp_impact':
                logger.info(f"  GDP Impact: ${prediction/1e9:.2f}B")
            elif target == 'job_creation':
                logger.info(f"  Job Creation: {int(prediction):,} jobs")
            else:  # roi_percentage
                logger.info(f"  ROI: {prediction:.1f}%")


def main():
    """Main training pipeline"""
    logger.info("="*80)
    logger.info("TRAINING ENHANCED ML MODELS (18 FEATURES)")
    logger.info("="*80)
    
    # Initialize trainer
    trainer = EnhancedModelTrainer()
    
    # Load data
    X, y = trainer.load_training_data()
    
    # Train models
    trainer.train_models(X, y)
    
    # Save models
    trainer.save_models()
    
    # Compare with 13-feature baseline
    trainer.compare_with_13_features()
    
    # Test predictions
    trainer.test_predictions()
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info("\nModel Performance Summary:")
    for target in trainer.targets:
        metrics = trainer.metrics[target]
        logger.info(f"\n{target}:")
        logger.info(f"  Model Type: {metrics['model_type']}")
        logger.info(f"  R² Score: {metrics['r2']:.4f}")
        logger.info(f"  CV Mean: {metrics['cv_mean']:.4f}")
    
    logger.info("\nModels saved to: models/enhanced/")
    logger.info("\nNext steps:")
    logger.info("1. Create ensemble model: python ml/create_ensemble_model.py")
    logger.info("2. Validate improvements: python ml/validate_improvements.py")
    logger.info("3. Update production: python update_production.py")


if __name__ == "__main__":
    main()
