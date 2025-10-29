"""Configuration settings for KC Cluster Prediction Tool"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://localhost/kc_clusters')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    # API Keys
    # IMPORTANT: API keys must be supplied via environment variables in production.
    # Defaults are intentionally blank to avoid accidental key leakage.
    BLS_API_KEY = os.getenv('BLS_API_KEY')
    USPTO_API_KEY = os.getenv('USPTO_API_KEY')  # PatentsView API key
    
    # Market Monitoring APIs
    FRED_API_KEY = os.getenv('FRED_API_KEY')
    EIA_API_KEY = os.getenv('EIA_API_KEY')
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    # Geopolitical APIs
    CENSUS_API_KEY = os.getenv('CENSUS_API_KEY')

    # KC Open Data / Socrata App Token (optional but recommended to avoid throttling)
    KCMO_APP_TOKEN = os.getenv('KCMO_APP_TOKEN') or os.getenv('SOCRATA_APP_TOKEN')
    
    # Data Collection
    SCRAPING_DURATION_DAYS = 7
    CONFIDENCE_THRESHOLD = 0.95
    RETRY_ATTEMPTS = 3
    REQUEST_TIMEOUT = 30
    
    # API Timeout configuration (in seconds)
    API_TIMEOUT_SHORT = 30   # For simple queries
    API_TIMEOUT_MEDIUM = 60  # For moderate data fetches
    API_TIMEOUT_LONG = 120   # For large data fetches
    API_TIMEOUT_EXTENDED = 300  # For very large operations (5 minutes)
    
    # Business Scoring Weights
    INNOVATION_WEIGHT = 0.30
    MARKET_POTENTIAL_WEIGHT = 0.40
    COMPETITION_WEIGHT = 0.30
    
    # Cluster Scoring Weights
    NATURAL_ASSETS_WEIGHT = 0.20
    INFRASTRUCTURE_WEIGHT = 0.20
    WORKFORCE_WEIGHT = 0.15
    INNOVATION_CAPACITY_WEIGHT = 0.15
    MARKET_ACCESS_WEIGHT = 0.15
    GEOPOLITICAL_WEIGHT = 0.10
    RESILIENCE_WEIGHT = 0.05
    
    # Economic Targets
    GDP_GROWTH_TARGET = 2872450000  # $2.87245 billion
    DIRECT_JOBS_TARGET = 1000
    INDIRECT_JOBS_TARGET = 2000
    WAGE_GROWTH_TARGET = 0.10  # 10% wage growth target
    TIME_HORIZON_YEARS = 5  # 5-year planning horizon
    MIN_ROI_THRESHOLD = 0.15  # 15% minimum ROI

    # Derived annual impact thresholds for validation/progress messaging
    # Defaults are computed from targets if env vars are not provided
    MIN_ANNUAL_GDP_IMPACT = float(os.getenv(
        'MIN_ANNUAL_GDP_IMPACT',
        str(GDP_GROWTH_TARGET / max(TIME_HORIZON_YEARS, 1) * 0.20)
    ))  # 20% of annualized GDP target

    # Conservative calibration parameters (paper Section 4.3)
    CALIBRATION_ENABLED = True
    MGDP_MULTIPLIER = float(os.getenv('MGDP_MULTIPLIER', '1.85'))
    MJOBS_MULTIPLIER = float(os.getenv('MJOBS_MULTIPLIER', '2.2'))
    CALIBRATION_FRICTION = float(os.getenv('CALIBRATION_FRICTION', '0.80'))  # behavioral friction
    SUCCESS_CAP = float(os.getenv('SUCCESS_CAP', '0.75'))  # cap on success probabilities when used
    # Macro triggers scale (optional): when adverse conditions detected, scale projections
    TRIGGER_SCALE = float(os.getenv('TRIGGER_SCALE', '0.90'))
    
    # Data Sources
    KANSAS_SOS_URL = "https://www.sos.ks.gov/businesses/"
    MISSOURI_SOS_URL = "https://www.sos.mo.gov/business/"
    BLS_API_URL = "https://api.bls.gov/publicAPI/v2/"
    USPTO_API_URL = "https://search.patentsview.org/api/v1/"
    SBIR_URL = "https://api.www.sbir.gov/public/api/"

    # Preloaded final dataset (optional): when set, the app will load this CSV
    # instead of scraping/generating businesses. Set via env FINAL_DATASET_CSV.
    FINAL_DATASET_CSV = os.getenv('FINAL_DATASET_CSV')
    
    # Patent Search Configuration
    SKIP_PATENT_SEARCH = os.getenv('SKIP_PATENT_SEARCH', 'false').lower() == 'true'
    
    # Optimization Parameters
    POPULATION_SIZE = 100
    GENERATIONS = 50
    CROSSOVER_PROB = 0.7
    MUTATION_PROB = 0.2
    PARETO_THRESHOLD = 0.30  # 30% threshold - businesses within 70% of top score pass (more inclusive)
    
    # Geographic Boundaries (Kansas City MSA)
    KC_MSA_COUNTIES = [
        # Kansas
        "Johnson County, KS",
        "Leavenworth County, KS",
        "Linn County, KS",
        "Miami County, KS",
        "Wyandotte County, KS",
        # Missouri
        "Bates County, MO",
        "Caldwell County, MO",
        "Cass County, MO",
        "Clay County, MO",
        "Clinton County, MO",
        "Jackson County, MO",
        "Lafayette County, MO",
        "Platte County, MO",
        "Ray County, MO"
    ]
    
    # Universities within 100 miles
    REGIONAL_UNIVERSITIES = [
        "University of Missouri-Kansas City",
        "University of Kansas",
        "Kansas State University",
        "Missouri S&T",
        "University of Missouri",
        "Wichita State University",
        "Pittsburg State University",
        "Johnson County Community College",
        "Metropolitan Community College"
    ]
    
    # Industry NAICS Codes by Cluster
    CLUSTER_NAICS_CODES = {
        "logistics": ["484", "488", "492", "493"],
        "biosciences": ["325", "3391", "621", "5417"],  # Pharma, medical devices, healthcare, biotech R&D
        "technology": ["511", "518", "519", "5415"],  # Software publishing, data processing, info services, computer services
        "manufacturing": ["311", "312", "332", "333", "336"],
        "animal_health": ["3253", "3259", "5419"]
    }
    
    # Longevity Score Factors
    LONGEVITY_FACTORS = {
        "infrastructure_quality": 0.25,
        "workforce_stability": 0.20,
        "political_stability": 0.15,
        "market_growth": 0.20,
        "innovation_ecosystem": 0.20
    }
    
    # Geographic Scope (Kansas City MSA Counties)
    KANSAS_COUNTIES = [
        "Johnson County, KS",
        "Leavenworth County, KS",
        "Linn County, KS",
        "Miami County, KS",
        "Wyandotte County, KS"
    ]
    
    MISSOURI_COUNTIES = [
        "Bates County, MO",
        "Caldwell County, MO",
        "Cass County, MO",
        "Clay County, MO",
        "Clinton County, MO",
        "Jackson County, MO",
        "Lafayette County, MO",
        "Platte County, MO",
        "Ray County, MO"
    ]
    
    GEOGRAPHIC_FOCUS = "both"  # Options: "urban", "suburban", "rural", "both"
    
    # Business Filtering Criteria
    MIN_EMPLOYEES = 2  # Lowered to include micro-businesses
    MAX_EMPLOYEES = 50000  # Raised to include all sizes
    MIN_REVENUE = 25000  # $25K minimum revenue (more inclusive)
    MIN_BUSINESS_AGE = 0  # Include all ages (startups important for clusters)
    EXCLUDED_NAICS = []  # NAICS codes to exclude
    
    # Scoring Weights (Configurable)
    BUSINESS_SCORING_WEIGHTS = {
        "innovation": 0.30,
        "market_potential": 0.40,
        "competition": 0.30
    }
    
    CLUSTER_SCORING_WEIGHTS = {
        "natural_assets": 0.20,
        "infrastructure": 0.20,
        "workforce": 0.15,
        "innovation": 0.15,
        "market_access": 0.15,
        "geopolitical": 0.10,
        "resilience": 0.05
    }
    
    # Conservative Economic Impact Multipliers (from Akinyede & Caruso 2025 paper)
    # These avoid the 200-300% overestimation common in traditional studies
    ECONOMIC_MULTIPLIERS = {
        "gdp": 1.85,           # vs traditional 3.2-4.1× (43% reduction)
        "employment": 2.2,     # vs traditional 3.8-4.5× (44% reduction)
        "indirect_jobs": 1.28, # vs traditional 2.5-3.0× (49% reduction)
        "induced_gdp": 0.65    # Additional induced effects
    }
    
    # Regional Strategic Multipliers for Kansas City MSA
    REGIONAL_STRATEGIC_MULTIPLIERS = {
        "animal_health": 1.30,    # KC Animal Health Corridor
        "biosciences": 1.25,       # Strong biotech presence
        "technology": 1.20,        # Growing tech sector
        "logistics": 1.15,         # Central US logistics hub
        "fintech": 1.15,          # Financial services strength
        "manufacturing": 1.10     # Traditional manufacturing base
    }
    
    # Dynamic Macro Adjustment Triggers
    MACRO_TRIGGERS = {
        "oil_price_high": 80,      # $/barrel threshold
        "interest_rate_high": 5.5, # % threshold
        "unemployment_low": 4.0,   # % threshold (tight labor)
        "volatility_high": 0.10    # 10% monthly commodity volatility
    }
    
    # Natural discovery parameters (new)
    INCLUDE_INFRASTRUCTURE_FEATURES = True  # Add infrastructure to clustering
    USE_SUCCESS_PROBABILITY = True          # Apply realistic probabilities
    GEOGRAPHIC_DIVERSITY_WEIGHT = 0.15      # 15% weight in optimization
    
    # Soft limits for monitoring (not enforcement)
    WARN_IF_TYPE_EXCEEDS = 0.4  # Warn if any type >40% of clusters
    WARN_IF_COUNTY_EXCEEDS = 0.5  # Warn if any county >50% of clusters
    
    # Data Sources Configuration
    ENABLED_DATA_SOURCES = [
        "state_registrations",
        "bls_employment",
        "uspto_patents",
        "sbir_awards",
        "university_partners",
        "infrastructure_assets"
    ]
    
    # Infrastructure Requirements
    REQUIRED_INFRASTRUCTURE = [
        "rail_access",
        "highway_access",
        "air_cargo",
        "broadband",
        "utilities"
    ]
    
    # Algorithm Parameters
    NUM_CLUSTERS = 5  # Number of clusters to identify (increased for better coverage)
    CLUSTER_SIZE_RANGE = (15, 150)  # Min and max businesses per cluster (expanded range)
    MIN_CLUSTER_SIZE = 10  # Minimum businesses needed to form a cluster
    MAX_CLUSTER_SIZE = 150  # Maximum businesses per cluster (increased)
# Real Data Enforcement
USE_ONLY_REAL_DATA = True  # Disable all mock/placeholder/fallback data
NO_FALLBACK_DATA = True    # Return empty results instead of placeholders
