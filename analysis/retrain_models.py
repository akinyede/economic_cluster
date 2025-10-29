"""
Retrain ML models with correct 8-feature structure matching ML enhancer expectations.

This script creates synthetic training data and trains models with the exact features
that the ML cluster enhancer uses for predictions.
"""
import pandas as pd
import numpy as np
import os
import sys
import pickle
import logging
from datetime import datetime
from typing import List, Dict, Tuple
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data_collection.scraper import BusinessDataScraper
from analysis.business_scorer import BusinessScorer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
NUM_TRAINING_CLUSTERS = 5000
MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '../models')
TRAINING_DATA_PATH = os.path.join(os.path.dirname(__file__), 'retrained_cluster_data.csv')

class ModelRetrainer:
    def __init__(self):
        self.config = Config()
        self.scraper = BusinessDataScraper()
        self.scorer = BusinessScorer()
        
    def generate_synthetic_clusters(self) -> List[Dict]:
        """Generate synthetic cluster data matching the 8-feature structure"""
        logger.info(f"Generating {NUM_TRAINING_CLUSTERS} synthetic clusters...")
        
        # Get real business data as basis
        logger.info("Loading real business data...")
        businesses = self._load_real_businesses()
        
        if businesses.empty:
            logger.warning("No real business data found, generating purely synthetic data")
            businesses = self._generate_synthetic_businesses()
        
        clusters = []
        cluster_types = ['logistics', 'manufacturing', 'technology', 'biosciences', 'animal_health', 'mixed']
        
        for i in range(NUM_TRAINING_CLUSTERS):
            if i % 100 == 0:
                logger.info(f"Generated {i}/{NUM_TRAINING_CLUSTERS} clusters...")
            
            # Randomly select cluster type
            cluster_type = np.random.choice(cluster_types)
            
            # Select businesses for this cluster
            cluster_size = np.random.randint(10, 100)
            
            # Bias selection based on cluster type
            if cluster_type != 'mixed' and not businesses.empty:
                # Try to select businesses matching the cluster type
                type_businesses = self._filter_by_cluster_type(businesses, cluster_type)
                if len(type_businesses) >= cluster_size:
                    cluster_businesses = type_businesses.sample(n=cluster_size)
                else:
                    # Mix with other businesses
                    n_type = len(type_businesses)
                    n_other = cluster_size - n_type
                    other_businesses = businesses[~businesses.index.isin(type_businesses.index)].sample(n=n_other)
                    cluster_businesses = pd.concat([type_businesses, other_businesses])
            else:
                # Random selection for mixed clusters
                cluster_businesses = businesses.sample(n=min(cluster_size, len(businesses)), replace=True)
            
            # Calculate the 8 features matching ML enhancer
            features = self._calculate_cluster_features(cluster_businesses, cluster_type)
            
            # Generate realistic outcomes
            outcomes = self._synthesize_outcomes(features, cluster_type)
            
            # Combine features and outcomes
            cluster_data = {**features, **outcomes}
            clusters.append(cluster_data)
        
        logger.info(f"Generated {len(clusters)} synthetic clusters")
        return clusters
    
    def _parse_employee_range(self, value) -> int:
        """Parse employee range string to numeric value"""
        if pd.isna(value) or value == '':
            return 10
        
        value = str(value).replace(',', '')
        
        # Handle ranges like "1-10", "50-100"
        if '-' in value:
            parts = value.split('-')
            try:
                # Return midpoint of range
                low = int(parts[0])
                high = int(parts[1]) if len(parts) > 1 else low
                return (low + high) // 2
            except:
                return 10
        
        # Try to extract first number
        import re
        numbers = re.findall(r'\d+', value)
        if numbers:
            return int(numbers[0])
        
        return 10  # Default
    
    def _parse_revenue(self, value) -> float:
        """Parse revenue string to numeric value"""
        if pd.isna(value) or value == '':
            return 500000  # Default $500K
        
        value = str(value).upper().replace('$', '').replace(',', '')
        
        # Handle millions/billions
        multiplier = 1
        if 'B' in value or 'BILLION' in value:
            multiplier = 1e9
        elif 'M' in value or 'MILLION' in value:
            multiplier = 1e6
        elif 'K' in value or 'THOUSAND' in value:
            multiplier = 1e3
        
        # Extract numeric part
        import re
        numbers = re.findall(r'\d+\.?\d*', value)
        if numbers:
            try:
                return float(numbers[0]) * multiplier
            except:
                pass
        
        return 500000  # Default $500K
    
    def _load_real_businesses(self) -> pd.DataFrame:
        """Load real business data if available"""
        try:
            # Try multiple potential data sources
            paths = [
                'data/kc_businesses_all_filtered.csv',
                'data/businesses_sample.csv',
                'data/kc_businesses.csv'
            ]
            
            for path in paths:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    logger.info(f"Loaded {len(df)} businesses from {path}")
                    
                    # Map column names
                    if 'SIC Code' in df.columns:
                        df['naics_code'] = df['SIC Code']
                    if 'Employee Range' in df.columns:
                        df['employees'] = df['Employee Range'].apply(self._parse_employee_range)
                    if 'Annual Sales' in df.columns:
                        df['revenue_estimate'] = df['Annual Sales'].apply(self._parse_revenue)
                    
                    # Ensure required columns
                    required_cols = ['name', 'naics_code', 'employees', 'revenue_estimate', 
                                   'year_established', 'patent_count', 'sbir_awards']
                    
                    # Add missing columns with defaults
                    for col in required_cols:
                        if col not in df.columns:
                            if col == 'year_established':
                                df[col] = np.random.randint(1990, 2020, size=len(df))
                            elif col in ['patent_count', 'sbir_awards']:
                                df[col] = 0
                            elif col == 'employees':
                                df[col] = np.random.randint(5, 500, size=len(df))
                            elif col == 'revenue_estimate':
                                df[col] = df.get('employees', 50) * np.random.randint(50000, 200000, size=len(df))
                    
                    return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error loading business data: {e}")
            return pd.DataFrame()
    
    def _generate_synthetic_businesses(self) -> pd.DataFrame:
        """Generate synthetic business data if no real data available"""
        n_businesses = 1000
        
        naics_codes = {
            'logistics': ['484', '488', '492', '493'],
            'manufacturing': ['332', '333', '334', '336'],
            'technology': ['511', '518', '519', '5415'],
            'biosciences': ['3254', '3391', '621', '5417'],
            'animal_health': ['3253', '3259', '5419']
        }
        
        all_naics = []
        for codes in naics_codes.values():
            all_naics.extend(codes)
        
        data = {
            'name': [f'Business_{i}' for i in range(n_businesses)],
            'naics_code': np.random.choice(all_naics, n_businesses),
            'employees': np.random.lognormal(3, 1.5, n_businesses).astype(int).clip(5, 5000),
            'year_established': np.random.randint(1980, 2020, n_businesses),
            'patent_count': np.random.poisson(0.5, n_businesses),
            'sbir_awards': np.random.poisson(0.1, n_businesses)
        }
        
        df = pd.DataFrame(data)
        df['revenue_estimate'] = df['employees'] * np.random.uniform(80000, 150000, n_businesses)
        
        return df
    
    def _filter_by_cluster_type(self, businesses: pd.DataFrame, cluster_type: str) -> pd.DataFrame:
        """Filter businesses by cluster type based on NAICS codes"""
        naics_mapping = {
            'logistics': ['484', '488', '492', '493'],
            'manufacturing': ['311', '312', '332', '333', '336'],
            'technology': ['511', '518', '519', '5415'],
            'biosciences': ['3254', '3391', '621', '5417'],
            'animal_health': ['3253', '3259', '5419']
        }
        
        if cluster_type in naics_mapping and 'naics_code' in businesses.columns:
            codes = naics_mapping[cluster_type]
            mask = businesses['naics_code'].astype(str).str[:3].isin([c[:3] for c in codes])
            return businesses[mask]
        
        return businesses
    
    def _calculate_cluster_features(self, businesses: pd.DataFrame, cluster_type: str) -> Dict:
        """Calculate the 8 features expected by ML enhancer"""
        
        # Map cluster type to numeric value
        type_values = {
            'logistics': 1.0,
            'biosciences': 2.0,
            'technology': 3.0,
            'manufacturing': 4.0,
            'animal_health': 5.0,
            'mixed': 0.0
        }
        
        # Calculate features matching _extract_cluster_features in ml_cluster_enhancer.py
        features = {
            'business_count': len(businesses),
            'total_employees': businesses['employees'].sum(),
            'total_revenue': businesses['revenue_estimate'].sum(),
            'avg_business_age': datetime.now().year - businesses['year_established'].mean(),
            'strategic_score': np.random.uniform(60, 90),  # Simulated strategic score
            'innovation_score': (businesses['patent_count'].sum() * 10 + 
                               businesses['sbir_awards'].sum() * 20) / len(businesses),
            'cluster_type': type_values.get(cluster_type, 0.0),
            'avg_patents': businesses['patent_count'].mean()
        }
        
        return features
    
    def _synthesize_outcomes(self, features: Dict, cluster_type: str) -> Dict:
        """Generate realistic outcomes based on features and economic principles"""
        
        # GDP impact calculation following our corrected methodology
        gdp_output_ratios = {
            'logistics': 0.45,
            'manufacturing': 0.35,
            'technology': 0.65,
            'biosciences': 0.55,
            'animal_health': 0.50,
            'mixed': 0.45
        }
        
        # Output multipliers
        output_multipliers = {
            'logistics': 2.8,
            'manufacturing': 2.5,
            'technology': 2.0,
            'biosciences': 2.3,
            'animal_health': 2.2,
            'mixed': 2.1
        }
        
        # Employment multipliers
        employment_multipliers = {
            'logistics': 1.8,
            'manufacturing': 1.6,
            'technology': 2.2,
            'biosciences': 2.0,
            'animal_health': 1.9,
            'mixed': 1.7
        }
        
        # Get cluster type name from numeric value
        type_map = {1.0: 'logistics', 2.0: 'biosciences', 3.0: 'technology', 
                   4.0: 'manufacturing', 5.0: 'animal_health', 0.0: 'mixed'}
        cluster_type_name = type_map.get(features['cluster_type'], 'mixed')
        
        # Calculate GDP impact
        gdp_ratio = gdp_output_ratios.get(cluster_type_name, 0.45)
        output_mult = output_multipliers.get(cluster_type_name, 2.1)
        base_gdp = features['total_revenue'] * output_mult * gdp_ratio
        
        # Add variability based on innovation and strategic scores
        innovation_factor = 1 + (features['innovation_score'] / 100) * 0.2
        strategic_factor = 1 + (features['strategic_score'] / 100) * 0.1
        
        # Add noise
        gdp_noise = np.random.normal(1.0, 0.1)
        actual_gdp = base_gdp * innovation_factor * strategic_factor * gdp_noise
        
        # Calculate job creation
        emp_mult = employment_multipliers.get(cluster_type_name, 1.7)
        direct_jobs = features['total_employees']
        indirect_jobs = direct_jobs * (emp_mult - 1)
        total_jobs = direct_jobs + indirect_jobs
        
        # Add growth factor
        job_growth_factor = 1 + (features['avg_business_age'] / 50) * 0.3  # Older clusters grow slower
        job_noise = np.random.normal(1.0, 0.08)
        actual_jobs = total_jobs * job_growth_factor * job_noise
        
        # Calculate ROI
        # Investment needed (following our corrected methodology)
        avg_emp_per_business = features['total_employees'] / max(features['business_count'], 1)
        if avg_emp_per_business < 10:
            investment_per_business = 50000
        elif avg_emp_per_business < 50:
            investment_per_business = 150000
        elif avg_emp_per_business < 200:
            investment_per_business = 500000
        else:
            investment_per_business = 1000000
        
        total_investment = features['business_count'] * investment_per_business
        
        # ROI calculation
        returns = actual_gdp * 0.3  # Assume 30% of GDP impact translates to returns over 5 years
        roi = (returns - total_investment) / total_investment if total_investment > 0 else 0
        
        # Add noise and ensure reasonable bounds
        roi_noise = np.random.normal(1.0, 0.15)
        actual_roi = max(-0.2, min(1.5, roi * roi_noise))  # ROI between -20% and 150%
        
        return {
            'actual_gdp_impact': max(0, actual_gdp),
            'actual_job_creation': max(0, int(actual_jobs)),
            'actual_roi': actual_roi
        }
    
    def train_models(self, training_data: pd.DataFrame) -> Dict:
        """Train the three models with proper feature scaling"""
        logger.info("Training models with 8-feature structure...")
        
        # Ensure output directory exists
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        
        # Define feature columns (must match _get_feature_names in ml_cluster_enhancer.py)
        feature_cols = [
            'business_count', 'total_employees', 'total_revenue',
            'avg_business_age', 'strategic_score', 'innovation_score',
            'cluster_type', 'avg_patents'
        ]
        
        # Prepare features and targets
        X = training_data[feature_cols]
        y_gdp = training_data['actual_gdp_impact']
        y_jobs = training_data['actual_job_creation']
        y_roi = training_data['actual_roi']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save scaler
        scaler_path = os.path.join(MODEL_OUTPUT_DIR, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Saved feature scaler to {scaler_path}")
        
        # XGBoost parameters
        xgb_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        results = {}
        
        # Train GDP model
        logger.info("Training GDP impact model...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_gdp, test_size=0.2, random_state=42
        )
        gdp_model = xgb.XGBRegressor(**xgb_params)
        gdp_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = gdp_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        logger.info(f"GDP Model - R²: {r2:.3f}, MAE: ${mae:,.0f}")
        results['gdp'] = {'r2': r2, 'mae': mae}
        
        # Save model
        with open(os.path.join(MODEL_OUTPUT_DIR, 'gdp_model.pkl'), 'wb') as f:
            pickle.dump(gdp_model, f)
        
        # Train Jobs model
        logger.info("Training job creation model...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_jobs, test_size=0.2, random_state=42
        )
        job_model = xgb.XGBRegressor(**xgb_params)
        job_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = job_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        logger.info(f"Job Model - R²: {r2:.3f}, MAE: {mae:.0f} jobs")
        results['jobs'] = {'r2': r2, 'mae': mae}
        
        # Save model
        with open(os.path.join(MODEL_OUTPUT_DIR, 'job_model.pkl'), 'wb') as f:
            pickle.dump(job_model, f)
        
        # Train ROI model
        logger.info("Training ROI model...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_roi, test_size=0.2, random_state=42
        )
        roi_model = xgb.XGBRegressor(**xgb_params)
        roi_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = roi_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        logger.info(f"ROI Model - R²: {r2:.3f}, MAE: {mae:.3f}")
        results['roi'] = {'r2': r2, 'mae': mae}
        
        # Save model
        with open(os.path.join(MODEL_OUTPUT_DIR, 'roi_model.pkl'), 'wb') as f:
            pickle.dump(roi_model, f)
        
        logger.info(f"All models saved to {MODEL_OUTPUT_DIR}")
        return results
    
    def run(self):
        """Main execution method"""
        logger.info("Starting model retraining process...")
        
        # Generate synthetic training data
        clusters = self.generate_synthetic_clusters()
        
        # Convert to DataFrame
        training_df = pd.DataFrame(clusters)
        
        # Save training data
        training_df.to_csv(TRAINING_DATA_PATH, index=False)
        logger.info(f"Saved {len(training_df)} training samples to {TRAINING_DATA_PATH}")
        
        # Display sample
        logger.info("\nSample of generated training data:")
        print(training_df.head())
        print(f"\nFeature statistics:")
        print(training_df.describe())
        
        # Train models
        results = self.train_models(training_df)
        
        logger.info("\n" + "="*50)
        logger.info("MODEL RETRAINING COMPLETE")
        logger.info("="*50)
        logger.info(f"\nModel Performance Summary:")
        for model_name, metrics in results.items():
            logger.info(f"{model_name.upper()} Model: R² = {metrics['r2']:.3f}, MAE = {metrics['mae']:,.0f}")
        
        logger.info(f"\nModels saved to: {MODEL_OUTPUT_DIR}")
        logger.info("The ML enhancer will now use these properly trained models.")
        
        return results


def main():
    """Main entry point"""
    retrainer = ModelRetrainer()
    retrainer.run()


if __name__ == "__main__":
    main()