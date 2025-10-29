"""
Analyze existing business data to understand characteristics and distributions
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BusinessDataAnalyzer:
    def __init__(self):
        self.data_paths = {
            'kc_filtered': 'data/kc_businesses_all_filtered.csv',
            'ks_email': '../KS_Business_Email_Data/KS_Business_Email.csv',
            'mo_email_1': '../MO_Business_Email_Data/MO_Business_Email_1.csv',
            'mo_email_2': '../MO_Business_Email_Data/MO_Business_Email_2.csv'
        }
        
    def load_and_analyze_all_data(self):
        """Load and analyze all available business data"""
        logger.info("="*60)
        logger.info("BUSINESS DATA ANALYSIS")
        logger.info("="*60)
        
        for name, path in self.data_paths.items():
            if Path(path).exists():
                logger.info(f"\nAnalyzing {name} from {path}")
                self._analyze_dataset(name, path)
            else:
                logger.warning(f"File not found: {path}")
    
    def _analyze_dataset(self, name: str, path: str):
        """Analyze a single dataset"""
        try:
            # Load data
            df = pd.read_csv(path, low_memory=False)
            logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
            
            # Basic info
            logger.info(f"\nColumns: {list(df.columns)}")
            
            # Analyze key fields
            self._analyze_naics_codes(df, name)
            self._analyze_geography(df, name)
            self._analyze_business_size(df, name)
            self._analyze_data_quality(df, name)
            
        except Exception as e:
            logger.error(f"Error analyzing {name}: {str(e)}")
    
    def _analyze_naics_codes(self, df: pd.DataFrame, dataset_name: str):
        """Analyze NAICS/SIC code distribution"""
        logger.info(f"\n--- NAICS/SIC Code Analysis for {dataset_name} ---")
        
        # Look for NAICS or SIC columns
        naics_col = None
        for col in df.columns:
            if 'NAICS' in col.upper() or 'SIC' in col.upper():
                naics_col = col
                break
        
        if naics_col:
            # Get top codes
            codes = df[naics_col].astype(str).str[:3]  # First 3 digits
            top_codes = codes.value_counts().head(10)
            
            logger.info(f"Top 10 industry codes:")
            for code, count in top_codes.items():
                pct = (count / len(df)) * 100
                logger.info(f"  {code}: {count} ({pct:.1f}%)")
            
            # Industry diversity
            unique_codes = codes.nunique()
            logger.info(f"\nIndustry diversity: {unique_codes} unique 3-digit codes")
            
            # Map to cluster types
            self._map_to_clusters(codes)
        else:
            logger.warning(f"No NAICS/SIC column found in {dataset_name}")
    
    def _map_to_clusters(self, codes):
        """Map NAICS codes to potential clusters"""
        cluster_mapping = {
            'Logistics': ['484', '488', '492', '493'],
            'Manufacturing': ['311', '312', '321', '322', '323', '324', '325', '326', '327', '331', '332', '333', '334', '335', '336', '337', '339'],
            'Technology': ['511', '517', '518', '519', '541'],
            'Biosciences': ['325', '339', '541', '621', '622'],
            'Finance': ['522', '523', '524', '525'],
            'Retail': ['441', '442', '443', '444', '445', '446', '447', '448', '451', '452', '453', '454']
        }
        
        cluster_counts = {}
        for cluster, cluster_codes in cluster_mapping.items():
            count = codes.isin(cluster_codes).sum()
            cluster_counts[cluster] = count
        
        logger.info("\nPotential cluster distribution:")
        total = sum(cluster_counts.values())
        for cluster, count in sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                pct = (count / total) * 100 if total > 0 else 0
                logger.info(f"  {cluster}: {count} ({pct:.1f}%)")
    
    def _analyze_geography(self, df: pd.DataFrame, dataset_name: str):
        """Analyze geographic distribution"""
        logger.info(f"\n--- Geographic Analysis for {dataset_name} ---")
        
        # County analysis
        if 'County' in df.columns or 'county' in df.columns:
            county_col = 'County' if 'County' in df.columns else 'county'
            top_counties = df[county_col].value_counts().head(10)
            
            logger.info("Top 10 counties:")
            for county, count in top_counties.items():
                pct = (count / len(df)) * 100
                logger.info(f"  {county}: {count} ({pct:.1f}%)")
        
        # State analysis
        if 'State' in df.columns or 'state' in df.columns:
            state_col = 'State' if 'State' in df.columns else 'state'
            states = df[state_col].value_counts()
            logger.info(f"\nState distribution: {dict(states)}")
    
    def _analyze_business_size(self, df: pd.DataFrame, dataset_name: str):
        """Analyze business size metrics"""
        logger.info(f"\n--- Business Size Analysis for {dataset_name} ---")
        
        # Employee analysis
        emp_cols = [col for col in df.columns if 'Employee' in col or 'employee' in col]
        if emp_cols:
            emp_col = emp_cols[0]
            logger.info(f"Employee column: {emp_col}")
            
            # Try to extract numeric values
            if df[emp_col].dtype in ['int64', 'float64']:
                stats = df[emp_col].describe()
                logger.info(f"Employee statistics:\n{stats}")
            else:
                # Sample values to understand format
                sample = df[emp_col].dropna().head(10).tolist()
                logger.info(f"Sample employee values: {sample}")
        
        # Revenue analysis
        rev_cols = [col for col in df.columns if 'Revenue' in col or 'Sales' in col]
        if rev_cols:
            rev_col = rev_cols[0]
            logger.info(f"\nRevenue column: {rev_col}")
            sample = df[rev_col].dropna().head(10).tolist()
            logger.info(f"Sample revenue values: {sample}")
        
        # Year established
        year_cols = [col for col in df.columns if 'Year' in col or 'Established' in col]
        if year_cols:
            year_col = year_cols[0]
            logger.info(f"\nYear established column: {year_col}")
            try:
                years = pd.to_numeric(df[year_col], errors='coerce')
                valid_years = years.dropna()
                if len(valid_years) > 0:
                    logger.info(f"Business age distribution:")
                    logger.info(f"  Oldest: {int(valid_years.min())}")
                    logger.info(f"  Newest: {int(valid_years.max())}")
                    logger.info(f"  Median year: {int(valid_years.median())}")
            except:
                pass
    
    def _analyze_data_quality(self, df: pd.DataFrame, dataset_name: str):
        """Analyze data quality and completeness"""
        logger.info(f"\n--- Data Quality Analysis for {dataset_name} ---")
        
        # Missing data
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        logger.info("Missing data by column:")
        for col, pct in missing_pct.items():
            if pct > 0:
                logger.info(f"  {col}: {pct:.1f}%")
        
        # Duplicate analysis
        if 'Company Name' in df.columns:
            duplicates = df['Company Name'].duplicated().sum()
            logger.info(f"\nDuplicate company names: {duplicates} ({(duplicates/len(df))*100:.1f}%)")
    
    def create_unified_dataset(self):
        """Create a unified dataset from all sources"""
        logger.info("\n" + "="*60)
        logger.info("CREATING UNIFIED DATASET")
        logger.info("="*60)
        
        all_data = []
        
        # Load KC filtered data
        if Path(self.data_paths['kc_filtered']).exists():
            df = pd.read_csv(self.data_paths['kc_filtered'], low_memory=False)
            df['source'] = 'kc_filtered'
            all_data.append(df)
            logger.info(f"Loaded {len(df)} records from KC filtered data")
        
        # Combine all data
        if all_data:
            unified = pd.concat(all_data, ignore_index=True)
            logger.info(f"\nUnified dataset: {len(unified)} total records")
            
            # Save unified dataset
            unified.to_csv('data/unified_business_data.csv', index=False)
            logger.info("Saved unified dataset to data/unified_business_data.csv")
            
            return unified
        else:
            logger.error("No data loaded!")
            return pd.DataFrame()
    
    def recommend_features(self):
        """Recommend features based on data analysis"""
        logger.info("\n" + "="*60)
        logger.info("FEATURE RECOMMENDATIONS FOR ML MODELS")
        logger.info("="*60)
        
        logger.info("\nBased on the data analysis, here are the recommended features:")
        
        logger.info("\n1. CORE BUSINESS FEATURES (Available):")
        logger.info("   - company_name")
        logger.info("   - naics_code / sic_code (industry classification)")
        logger.info("   - county (geographic location)")
        logger.info("   - state")
        logger.info("   - employees (needs parsing from ranges)")
        logger.info("   - revenue_estimate (needs parsing from ranges)")
        
        logger.info("\n2. DERIVED FEATURES (Can Calculate):")
        logger.info("   - business_age (from year_established)")
        logger.info("   - industry_cluster_type (from NAICS mapping)")
        logger.info("   - county_business_density (businesses per county)")
        logger.info("   - industry_concentration (% of businesses in same NAICS)")
        logger.info("   - size_category (micro/small/medium/large)")
        
        logger.info("\n3. EXTERNAL FEATURES (From APIs):")
        logger.info("   - patent_count (USPTO API)")
        logger.info("   - sbir_awards (SBIR API)")
        logger.info("   - industry_employment_trend (BLS API)")
        logger.info("   - county_unemployment_rate (BLS API)")
        logger.info("   - infrastructure_score (composite from multiple sources)")
        
        logger.info("\n4. CLUSTER-LEVEL FEATURES:")
        logger.info("   - business_count (number of businesses in cluster)")
        logger.info("   - total_employees (sum of all employees)")
        logger.info("   - total_revenue (sum of all revenue)")
        logger.info("   - avg_business_age (average age of businesses)")
        logger.info("   - industry_diversity (number of unique NAICS codes)")
        logger.info("   - geographic_concentration (spread across counties)")
        logger.info("   - cluster_type (logistics/manufacturing/tech/etc)")
        logger.info("   - innovation_score (patents + SBIR per business)")
        
        logger.info("\n5. TARGET VARIABLES FOR TRAINING:")
        logger.info("   - gdp_impact (economic output)")
        logger.info("   - job_creation (direct + indirect jobs)")
        logger.info("   - roi (return on investment)")
        logger.info("   - success_probability (composite score)")


def main():
    analyzer = BusinessDataAnalyzer()
    
    # Analyze all datasets
    analyzer.load_and_analyze_all_data()
    
    # Create unified dataset
    unified_data = analyzer.create_unified_dataset()
    
    # Provide feature recommendations
    analyzer.recommend_features()


if __name__ == "__main__":
    main()