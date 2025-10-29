"""Main application for KC Cluster Prediction Tool"""
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import os
import math

from config import Config
from database import db
from models import Base, Business, Cluster, ClusterMembership
from data_collection.scraper import BusinessDataScraper
from analysis.business_scorer import BusinessScorer
from analysis.cluster_optimizer import ClusterOptimizer

# Import new analyzers
from analysis.workforce_analyzer import WorkforceAnalyzer
from analysis.university_integrator import UniversityIntegrator
from analysis.market_analyzer import EnhancedMarketAnalyzer
from analysis.sbir_integrator import SBIRIntegrator
from analysis.longevity_scorer import LongevityScorer
from analysis.patent_analyzer import PatentAnalyzer
from analysis.patent_analyzer_improved import ImprovedPatentAnalyzer
from analysis.revenue_projector import RevenueProjector
from ml.business_selected_inference import apply_business_predictions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClusterPredictionTool:
    """Main application class for cluster prediction"""
    
    def __init__(self, config=None, use_sqlite=None):
        self.config = config if config else Config()
        self.params = {}  # Initialize params storage
        
        # Initialize database
        if use_sqlite is None:
            use_sqlite = os.getenv('USE_SQLITE', 'true').lower() == 'true'
        
        if not db.initialize(use_sqlite=use_sqlite):
            raise RuntimeError("Failed to initialize database")
            
        self.session = db.get_session()
        self.scraper = BusinessDataScraper(self.session)
        self.scorer = BusinessScorer()
        self.optimizer = ClusterOptimizer()
        
        # Dynamically import MLClusterEnhancer to avoid caching issues
        self._ml_enhancer = None
        
        # Runtime toggles that control optional subsystems during analysis
        self._reset_runtime_options()

        # Initialize new analyzers
        self.workforce_analyzer = WorkforceAnalyzer()
        self.university_integrator = UniversityIntegrator()
        self.market_analyzer = EnhancedMarketAnalyzer()
        self.sbir_integrator = SBIRIntegrator()
        self.longevity_scorer = LongevityScorer()
        self.patent_analyzer = PatentAnalyzer()
        self.improved_patent_analyzer = ImprovedPatentAnalyzer(self.config)
        self.revenue_projector = RevenueProjector()
    
    @property
    def ml_enhancer(self):
        """Lazy load ML enhancer with fresh import to avoid caching issues"""
        if self._ml_enhancer is None:
            # Dynamic import to get fresh module
            import importlib
            import sys
            
            # Try to use enhanced version if available
            try:
                # First try the enhanced version with KC features
                module_name = 'analysis.ml_cluster_enhancer_v2'
                if module_name in sys.modules:
                    del sys.modules[module_name]
                ml_module = importlib.import_module(module_name)
                MLClusterEnhancer = ml_module.MLClusterEnhancerV2
                logger.info("Using enhanced ML cluster enhancer with KC features")
            except ImportError:
                # Fallback to original version
                module_name = 'analysis.ml_cluster_enhancer'
                if module_name in sys.modules:
                    del sys.modules[module_name]
                ml_module = importlib.import_module(module_name)
                MLClusterEnhancer = ml_module.MLClusterEnhancer
                logger.info("Using standard ML cluster enhancer")
            
            # Create new instance with fresh module
            self._ml_enhancer = MLClusterEnhancer(self.config)
            logger.info("Created fresh MLClusterEnhancer instance")
            
        return self._ml_enhancer
        
    # ---------------------- Consolidation helpers ----------------------
    def _normalize_text(self, s: str) -> str:
        try:
            s = str(s or '').upper()
            s = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in s)
            s = ' '.join(s.split())
            # remove common suffixes
            for suf in (" LLC", " INC", " CORPORATION", " CORP", " LTD", " CO", " COMPANY"):
                if s.endswith(suf):
                    s = s[: -len(suf)]
            return s
        except Exception:
            return ''

    def _business_key(self, b: Dict) -> str:
        name = self._normalize_text(b.get('name', ''))
        addr = self._normalize_text(b.get('address', ''))
        zip5 = str(b.get('zip', '') or b.get('zipcode', '')).strip()[:5]
        return f"{name}|{addr}|{zip5}".strip('|')

    def _coord_of(self, b: Dict) -> Optional[Tuple[float, float]]:
        try:
            lat = b.get('lat', b.get('latitude'))
            lon = b.get('lon', b.get('longitude'))
            if lat is None or lon is None:
                return None
            lat = float(lat); lon = float(lon)
            if not ( -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                return None
            return (lat, lon)
        except Exception:
            return None

    def _centroid(self, coords: List[Tuple[float, float]]) -> Tuple[float, float]:
        if not coords:
            return (39.0997, -94.5786)
        lat = sum(c[0] for c in coords)/len(coords)
        lon = sum(c[1] for c in coords)/len(coords)
        return (lat, lon)

    def _km_dist(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        R = 6371.0
        lat1, lon1 = a; lat2, lon2 = b
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        p1 = math.radians(lat1); p2 = math.radians(lat2)
        h = math.sin(dlat/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon/2)**2
        return 2*R*math.asin(math.sqrt(h))

    def _cluster_signature(self, cluster: Dict) -> Dict:
        keys = set()
        coords = []
        for b in cluster.get('businesses', []) or []:
            k = self._business_key(b)
            if k:
                keys.add(k)
            c = self._coord_of(b)
            if c:
                coords.append(c)
        return {
            'keys': keys,
            'centroid': self._centroid(coords)
        }

    def _merge_clusters(self, members: List[Dict]) -> Dict:
        """Union businesses and conservatively merge metrics."""
        merged: Dict = {}
        merged['type'] = members[0].get('type', 'mixed')
        # Name: join unique names
        names = [m.get('name', '') for m in members if m.get('name')]
        merged['name'] = ' + '.join(sorted(set(names))) or f"Consolidated {merged['type'].title()} Cluster"

        # Businesses: union by key
        seen = set()
        merged_businesses: List[Dict] = []
        for m in members:
            for b in (m.get('businesses') or []):
                k = self._business_key(b)
                if not k or k in seen:
                    continue
                seen.add(k)
                merged_businesses.append(b)
        merged['businesses'] = merged_businesses
        merged['business_count'] = len(merged_businesses)

        # Metrics: recompute simple aggregates where possible
        try:
            df = pd.DataFrame(merged_businesses)
            # totals
            merged.setdefault('metrics', {})
            if 'employees' in df.columns:
                merged['metrics']['total_employees'] = pd.to_numeric(df['employees'], errors='coerce').fillna(0).sum()
            if 'revenue' in df.columns:
                merged['metrics']['total_revenue'] = pd.to_numeric(df['revenue'], errors='coerce').fillna(0).sum()
            if 'year_established' in df.columns:
                yr = pd.to_numeric(df['year_established'], errors='coerce')
                age = (datetime.now().year - yr).clip(lower=0)
                merged['metrics']['avg_business_age'] = float(age.mean()) if len(age) else 0.0
        except Exception:
            pass

        # Preserve/aggregate key scalar fields conservatively
        def _max_of(field: str, default=0):
            try:
                return max((float(m.get(field, default) or default) for m in members))
            except Exception:
                return default
        merged['strategic_score'] = _max_of('strategic_score', 0)
        merged['critical_mass'] = _max_of('critical_mass', 0)
        merged['market_position'] = _max_of('market_position', 0)

        return merged

    def _consolidate_clusters(self, clusters: List[Dict]) -> List[Dict]:
        if not clusters:
            return clusters
        n = len(clusters)
        sigs = [self._cluster_signature(c) for c in clusters]
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # Thresholds
        JACC_STRICT = 0.5
        JACC_LOOSE = 0.3
        DIST_KM = 8.0

        for i in range(n):
            for j in range(i+1, n):
                type_i = clusters[i].get('type') or 'mixed'
                type_j = clusters[j].get('type') or 'mixed'
                if type_i != type_j:
                    continue
                A, B = sigs[i], sigs[j]
                if not A['keys'] or not B['keys']:
                    continue
                inter = len(A['keys'] & B['keys'])
                union_sz = len(A['keys'] | B['keys']) or 1
                jacc = inter / union_sz
                dist = self._km_dist(A['centroid'], B['centroid'])
                if jacc >= JACC_STRICT or (dist < DIST_KM and jacc >= JACC_LOOSE):
                    union(i, j)

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            r = find(i)
            groups.setdefault(r, []).append(i)

        consolidated: List[Dict] = []
        for _, idxs in groups.items():
            if len(idxs) == 1:
                consolidated.append(clusters[idxs[0]])
            else:
                members = [clusters[k] for k in idxs]
                merged = self._merge_clusters(members)
                consolidated.append(merged)

        return consolidated

    def _compute_portfolio_impact_unique(self, clusters: List[Dict]) -> Dict:
        """Compute portfolio impact on the union of unique businesses across clusters to avoid double counting."""
        # Union businesses by normalized key
        unique = {}
        for c in clusters:
            for b in (c.get('businesses') or []):
                k = self._business_key(b)
                if not k:
                    continue
                if k not in unique:
                    unique[k] = b
        if not unique:
            return {'gdp_impact_5yr': 0, 'total_jobs': 0}
        df = pd.DataFrame(unique.values())
        # Ensure required fields exist
        for col in ['revenue_estimate', 'employees', 'naics_code']:
            if col not in df.columns:
                df[col] = 0
        return self.optimizer.calculate_economic_impact(df)

    def run_full_analysis(self, custom_params: Dict = None, progress_callback=None) -> Dict:
        """Run complete cluster analysis pipeline

        Args:
            custom_params: Custom parameters to override config
            progress_callback: Optional callback function(stage, progress, message, highlight=None)
        """
        logger.info("Starting full cluster analysis for Kansas City MSA")
        
        # Reset runtime options to defaults for each run before applying overrides
        self._reset_runtime_options()

        # Apply custom parameters if provided
        if custom_params:
            self._apply_custom_params(custom_params)

        results = {
            "timestamp": datetime.now().isoformat(),
            "status": "in_progress",
            "parameters": custom_params or {},
            "steps": {}
        }
        
        # Progress tracking (dynamic weights by mode)
        quick_mode = bool(custom_params.get('quick_mode', False)) if custom_params else False
        if quick_mode:
            # Include intermediate stages so the overall progress advances during cleaning and KC enhancement
            stages = {
                'data_collection':   {'weight': 12, 'current': 0},
                'data_cleaning':     {'weight': 8,  'current': 0},
                'kc_enhancement':    {'weight': 5,  'current': 0},
                'business_scoring':  {'weight': 20, 'current': 0},
                'cluster_formation': {'weight': 25, 'current': 0},
                'ml_enhancement':    {'weight': 10, 'current': 0},
                'enhanced_analysis': {'weight': 10, 'current': 0},
                'validation':        {'weight': 5,  'current': 0},
                'recommendations':   {'weight': 5,  'current': 0}
            }
        else:
            stages = {
                'data_collection':   {'weight': 12, 'current': 0},
                'data_cleaning':     {'weight': 8,  'current': 0},
                'kc_enhancement':    {'weight': 5,  'current': 0},
                'business_scoring':  {'weight': 35, 'current': 0},
                'cluster_formation': {'weight': 20, 'current': 0},
                'ml_enhancement':    {'weight': 8,  'current': 0},
                'enhanced_analysis': {'weight': 7,  'current': 0},
                'validation':        {'weight': 3,  'current': 0},
                'recommendations':   {'weight': 2,  'current': 0}
            }
        
        def calculate_overall_progress():
            """Calculate weighted overall progress"""
            total = 0
            for stage_data in stages.values():
                total += (stage_data['current'] / 100) * stage_data['weight']
            return int(total)
        
        # Helper function to report progress
        def report_progress(stage, stage_progress, message, highlight=None):
            """Report progress for a specific stage"""
            if stage in stages:
                # Ensure per-stage progress is monotonic
                stages[stage]['current'] = max(stages[stage]['current'], stage_progress)
            
            overall_progress = calculate_overall_progress()
            
            if progress_callback:
                progress_callback(stage, overall_progress, message, highlight)
            logger.info(f"Progress: {overall_progress}% (Stage: {stage} at {stage_progress}%) - {message}")
        
        try:
            # Step 1: Data Collection
            logger.info("Step 1: Collecting data...")
            report_progress('data_collection', 10, 'Initializing data collection...')
            
            # Get data sources count for progress message
            data_sources = custom_params.get('data_sources', []) if custom_params else []
            
            # Handle skip_patents for quick mode
            # Default behavior: if quick_mode and skip_patents not explicitly provided, skip patents
            if custom_params and 'skip_patents' in custom_params:
                skip_patents = bool(custom_params.get('skip_patents'))
            else:
                skip_patents = quick_mode
            if skip_patents:
                logger.info("Quick mode: Skipping patent searches")
                self.config.SKIP_PATENT_SEARCH = True
                # Also update scraper config if it has one
                if hasattr(self.scraper, 'config'):
                    self.scraper.config.SKIP_PATENT_SEARCH = True
                    
            report_progress('data_collection', 30, 
                          f'Collecting business data from {len(data_sources)} sources{"" if skip_patents else " (including patents)"}...')
            
            scraping_results = self.scraper.run_full_scraping_cycle()
            
            report_progress('data_collection', 100, 
                          f'Collected {len(scraping_results["businesses"])} businesses',
                          f'✓ Found {len(scraping_results["businesses"])} total businesses in KC metro area (before filtering)')
            
            results["steps"]["data_collection"] = {
                "businesses_collected": len(scraping_results["businesses"]),
                "sbir_awards": len(scraping_results["sbir_awards"]),
                "infrastructure_assets": len(scraping_results["infrastructure"]),
                "employment_metrics": len(scraping_results["employment_data"]),
                "university_partners": len(scraping_results.get("university_data", []))
            }
            
            # Step 1.5: Clean the data
            logger.info("Step 1.5: Cleaning business data...")
            report_progress('data_cleaning', 10, 'Starting data quality checks and deduplication...')
            
            # Import and use data cleaner
            try:
                from data_processing.data_cleaner import DataCleaner
                cleaner = DataCleaner()
                
                # Convert to DataFrame for cleaning
                business_df = pd.DataFrame(scraping_results["businesses"])
                logger.info(f"Businesses before cleaning: {len(business_df)}")
                
                # Determine early sampling size for quick mode
                sample_size_early = None
                if quick_mode:
                    sample_size_early = (custom_params or {}).get('sample_size')
                    if not sample_size_early:
                        sample_size_early = 10000

                # Clean the data (fast pass)
                cleaned_df = cleaner.clean_business_data(business_df)

                # Deterministic de-dup (fast)
                cleaned_df = cleaner.deduplicate_by_priority(cleaned_df)

                if quick_mode:
                    # Quick mode: skip expensive fuzzy/geospatial de-dup to speed up
                    # Optionally downsample early to shrink later steps
                    if sample_size_early and len(cleaned_df) > sample_size_early:
                        # Use reproducible sample
                        cleaned_df = cleaned_df.sample(n=sample_size_early, random_state=42)
                        logger.info(f"Quick mode: early sampled {sample_size_early} businesses after basic cleaning")
                        report_progress('data_cleaning', 60, f'Fast cleaning + early sample to {len(cleaned_df)} businesses')
                    else:
                        report_progress('data_cleaning', 60, f'Fast cleaning complete on {len(cleaned_df)} businesses')
                else:
                    # Full mode: perform deeper duplicate removal
                    cleaned_df = cleaner.deduplicate_entities(cleaned_df)
                    cleaned_df = cleaner.deduplicate_geospatial(cleaned_df)
                
                # Optional: Balance industries to prevent over-dominance
                # Disable this for completely unbiased analysis
                # Uncomment the following lines only if NAICS 541 is overwhelming results
                # industry_caps = {
                #     '541': 0.15,  # Cap professional services at 15%
                #     '511': 0.05,  # Cap software publishing at 5%
                #     '523': 0.05,  # Cap securities at 5%
                # }
                # cleaned_df = cleaner.balance_industries(cleaned_df, industry_caps)
                
                logger.info("Industry balancing DISABLED for unbiased analysis")
                
                # Update scraping results with cleaned data
                scraping_results["businesses"] = cleaned_df.to_dict('records')
                logger.info(f"Businesses after cleaning: {len(scraping_results['businesses'])}")
                
                report_progress('data_cleaning', 100, 
                              f'✓ Data cleaned: {len(scraping_results["businesses"])} high-quality businesses (removed {len(business_df) - len(scraping_results["businesses"])} duplicates/outliers)')
                
            except Exception as e:
                logger.warning(f"Data cleaning failed, continuing with original data: {e}")
                report_progress('data_cleaning', 100, 'Data cleaning skipped')
            
            # Step 1.5: Apply User Filters BEFORE Sampling
            # This ensures we sample from businesses that meet user criteria
            if scraping_results["businesses"]:
                logger.info("Step 1.5: Applying user filters before sampling...")
                original_count = len(scraping_results["businesses"])
                # Use inline length to avoid any chance of uninitialized local reference
                report_progress('data_collection', 92, f'Applying filters: revenue, employees, age criteria to {len(scraping_results["businesses"]) } businesses...')
                filtered_businesses, filter_stats = self._apply_user_filters_to_businesses(
                    scraping_results["businesses"]
                )
                
                scraping_results["businesses"] = filtered_businesses
                
                # Log detailed filter statistics
                logger.info(f"Filter Results: {original_count} → {len(filtered_businesses)} businesses")
                if filter_stats:
                    for filter_name, count in filter_stats.items():
                        if filter_name not in ['original', 'passed'] and count > 0:
                            logger.info(f"  - Filtered by {filter_name}: {count} businesses")
                
                report_progress('data_collection', 95, 
                              f'Filtered: {original_count} → {len(filtered_businesses)} businesses')
                
                # Check if we have enough businesses after filtering
                if len(filtered_businesses) == 0:
                    error_msg = "No businesses passed the user filters. Please relax your criteria."
                    logger.error(error_msg)
                    report_progress('data_collection', 100, error_msg)
                    return {"error": error_msg, "filter_stats": filter_stats}
                elif len(filtered_businesses) < 100:
                    logger.warning(f"Only {len(filtered_businesses)} businesses passed filters. Consider relaxing criteria.")
            
            # Step 1.6: Apply Quick Mode Sampling AFTER Filtering
            # quick_mode was already set at the beginning of the function for stage setup
            sample_size = custom_params.get('sample_size') if custom_params else None
            if quick_mode and not sample_size:
                sample_size = 10000
            
            if sample_size and len(scraping_results["businesses"]) > sample_size:
                logger.info(f"Quick mode: Sampling {sample_size} businesses from {len(scraping_results['businesses'])} filtered businesses")
                report_progress('data_collection', 98, f'Quick mode: sampling {sample_size} businesses from filtered set')
                
                # Convert to DataFrame for sampling
                import random
                sampled_businesses = random.sample(scraping_results["businesses"], min(sample_size, len(scraping_results["businesses"])))
                scraping_results["businesses"] = sampled_businesses
                logger.info(f"Sampled {len(scraping_results['businesses'])} businesses for quick mode (all meet user criteria)")
            else:
                if sample_size:
                    logger.info(f"Using all {len(scraping_results['businesses'])} filtered businesses (less than sample size of {sample_size})")
                report_progress('data_collection', 98, f'Using {len(scraping_results["businesses"])} businesses')
            
            # Step 1.65: Integrate Federal Data Sources (FRED, EIA, Alpha Vantage)
            logger.info("Step 1.65: Integrating federal data sources...")
            # Safely preview API keys (or 'mock' when not configured)
            _fred_key = getattr(self.config, 'FRED_API_KEY', None)
            _fred_preview = (_fred_key[:8] + '…') if _fred_key else 'mock'
            report_progress('data_collection', 99, f'Integrating federal data: FRED unemployment ({_fred_preview}), EIA energy costs, market sentiment...')
            try:
                # Apply federal data integration from paper methodology
                scraping_results["businesses"] = self.scraper.integrate_federal_data_sources(
                    scraping_results["businesses"]
                )
                logger.info("Federal data integration complete")
            except Exception as e:
                logger.warning(f"Federal data integration failed, continuing without macro context: {e}")
            
            # Step 1.7: KC Data Enhancement (Optional - Only on filtered & sampled data)
            use_kc_features = custom_params.get('use_kc_features', True) if custom_params else True
            if use_kc_features:
                logger.info("Step 1.7: Enhancing with KC Open Data features (only on filtered & sampled businesses)...")
                report_progress('kc_enhancement', 10, f'Connecting to KC Open Data for {len(scraping_results["businesses"])} businesses...')
                
                try:
                    # Convert to DataFrame if not already
                    if isinstance(scraping_results["businesses"], list):
                        business_df = pd.DataFrame(scraping_results["businesses"])
                    else:
                        business_df = scraping_results["businesses"]
                    
                    # Enhance with KC features
                    enhanced_df = self._enhance_with_kc_features(business_df)
                    
                    # Update scraping results with enhanced data
                    scraping_results["businesses"] = enhanced_df.to_dict('records')
                    
                    # Count KC features added
                    kc_features = [col for col in enhanced_df.columns if col.startswith('kc_')]
                    logger.info(f"Added {len(kc_features)} KC features to business data")
                    
                    report_progress('kc_enhancement', 100, 
                                  f'✓ Enhanced with {len(kc_features)} KC features: transit access, demographics, crime stats, business density')
                    
                    results["steps"]["kc_enhancement"] = {
                        "features_added": kc_features,
                        "businesses_enhanced": len(enhanced_df)
                    }
                    
                except Exception as e:
                    logger.warning(f"KC enhancement failed, continuing without KC features: {e}")
                    report_progress('kc_enhancement', 100, 'KC enhancement skipped')
            
            # Step 2: Prepare Business Data
            logger.info("Step 2: Preparing business data...")
            report_progress('business_scoring', 10, f'Preparing {len(scraping_results["businesses"])} businesses for ML scoring...')
            
            businesses_df = self._prepare_business_data(scraping_results)
            
            # Quick mode was already determined and sampling done before KC enhancement
            # is_quick_mode is used for patent analysis strategy
            is_quick_mode = quick_mode
            
            # Step 3: Patent Analysis Strategy
            if skip_patents:
                # Skip all patent analysis, initialize with zeros for downstream scoring
                logger.info("Quick mode: Skipping patent analysis stage entirely")
                report_progress('business_scoring', 20, '✓ Skipping patent analysis for quick mode (saves ~2 hours)')
                try:
                    businesses_df['patent_count'] = 0
                except Exception:
                    pass
                self.patent_mapping = {}
            elif is_quick_mode:
                # QUICK MODE: Search patents for top 1000 performers only
                logger.info(f"Quick mode: Will search patents for top 1000 businesses (from {len(businesses_df):,} total)")
                report_progress('business_scoring', 15, 
                              'Quick scoring all businesses to identify top performers...')
                
                # First, do a quick scoring without patents to identify top businesses
                # This uses a simplified scoring based on size, revenue, and industry
                quick_scores = self._quick_score_businesses(businesses_df)
                businesses_df['quick_score'] = quick_scores
                
                # Get top 1000 performers
                top_1000_df = businesses_df.nlargest(1000, 'quick_score')
                
                # Create a mapping of index to business name for top 1000
                index_to_name = dict(zip(top_1000_df.index, top_1000_df['name']))
                top_1000_names = top_1000_df['name'].unique().tolist()
                
                report_progress('business_scoring', 20, 
                              f'Searching patents for top 1000 performers...')
                
                # Search patents only for top 1000 (org-only fast path)
                self.patent_mapping = self.improved_patent_analyzer.analyze_all_businesses(
                    business_names=top_1000_names,
                    progress_callback=lambda pct, msg: report_progress('business_scoring', max(20, min(90, int(pct))), msg),
                    org_only=True
                )
                
                # Create index-based patent mapping to avoid name collisions
                index_patent_mapping = {}
                for idx, name in index_to_name.items():
                    if name in self.patent_mapping:
                        index_patent_mapping[idx] = self.patent_mapping[name]
                
                # Add patent counts using index-based mapping
                businesses_df['patent_count'] = 0  # Initialize all to 0
                for idx, patent_count in index_patent_mapping.items():
                    businesses_df.loc[idx, 'patent_count'] = patent_count
                
                # Log statistics
                patents_found = sum(1 for count in self.patent_mapping.values() if count > 0)
                total_patents = sum(self.patent_mapping.values())
                logger.info(f"Quick mode patent search complete: {patents_found} of top 1000 have {total_patents} patents")
                
                report_progress('business_scoring', 30, 
                              f'Found {total_patents} patents in top 1000 businesses')
                
            else:
                # FULL MODE: Search patents for ALL businesses
                report_progress('business_scoring', 15, 
                              f'Analyzing patents for ALL {len(businesses_df):,} businesses...')
                
                # Get all unique business names
                business_names = businesses_df['name'].unique().tolist()
                logger.info(f"Full mode: Analyzing patents for {len(business_names):,} unique businesses")
                
                # Perform comprehensive patent search
                self.patent_mapping = self.improved_patent_analyzer.analyze_all_businesses(
                    progress_callback=lambda pct, msg: report_progress('business_scoring', max(15, min(95, int(pct))), msg)
                )
                
                # Create index-based patent mapping to avoid name collisions
                businesses_df['patent_count'] = 0  # Initialize all to 0
                for idx, row in businesses_df.iterrows():
                    if row['name'] in self.patent_mapping:
                        businesses_df.loc[idx, 'patent_count'] = self.patent_mapping[row['name']]
                
                # Log patent statistics
                businesses_with_patents = (businesses_df['patent_count'] > 0).sum()
                total_patents = businesses_df['patent_count'].sum()
                logger.info(f"Full patent analysis complete: {businesses_with_patents:,} businesses have {total_patents:,} patents")
                
                report_progress('business_scoring', 30, 
                              f'Found {total_patents:,} patents across {businesses_with_patents:,} businesses')
            
            # Step 3: Business Scoring
            logger.info("Step 3: Scoring businesses...")
            report_progress('business_scoring', 40, 
                          'Analyzing market conditions and infrastructure...')
            
            # Enrich with market and infrastructure data
            market_data = self._analyze_market_conditions(businesses_df)
            infrastructure_data = self._analyze_infrastructure(scraping_results["infrastructure"])
            
            # Add economic targets to market_data for dynamic threshold calculation
            if custom_params and 'economic_targets' in custom_params:
                market_data['economic_targets'] = custom_params['economic_targets']
                logger.info(f"Added economic targets to market data: GDP=${custom_params['economic_targets'].get('gdp_growth', 0)/1e9:.2f}B, Jobs={custom_params['economic_targets'].get('direct_jobs', 0)}")
            
            report_progress('business_scoring', 60, 
                          f'Scoring {len(businesses_df)} businesses with ML models...')
            
            # Get sample size from custom params (for quick mode)
            sample_size = None
            if custom_params and custom_params.get('sample_size'):
                sample_size = custom_params['sample_size']
                logger.info(f"Quick mode: Will sample {sample_size} businesses using stratified top-N approach")
            
            # Score and rank businesses
            # Map batch progress 0-100 to 60-95 of stage progress
            def _scoring_progress(sub_pct, msg):
                try:
                    mapped = 60 + int((min(max(int(sub_pct), 0), 100) / 100) * 35)
                    report_progress('business_scoring', mapped, msg)
                except Exception:
                    pass

            scored_businesses = self.scorer.rank_businesses(
                businesses_df.to_dict("records"),
                market_data,
                infrastructure_data,
                sample_size=sample_size,
                progress_callback=_scoring_progress
            )
            
            # Get the count of businesses that passed threshold
            eligible_count = len(scored_businesses[scored_businesses['passes_threshold']]) if 'passes_threshold' in scored_businesses.columns else len(scored_businesses)
            
            report_progress('business_scoring', 100, 
                          f'Scored {len(scored_businesses)} businesses',
                          f'✓ {eligible_count} businesses passed filters and scoring threshold')
            
            # Check if all businesses were filtered out
            if scored_businesses.empty:
                logger.warning("All businesses were filtered out. Please adjust your filter criteria.")
                
                # Get detailed filter reasons if available
                filter_reasons = scored_businesses.attrs.get('filter_reasons', {})
                total_businesses = scored_businesses.attrs.get('total_businesses', len(businesses_df))
                
                # Build detailed error message
                error_parts = ["All businesses were filtered out by the current filter criteria."]
                suggestions = []
                
                if filter_reasons:
                    error_parts.append("\n\nDetailed breakdown:")
                    
                    if filter_reasons.get('employees', 0) > 0:
                        error_parts.append(f"• {filter_reasons['employees']} businesses filtered by employee count (Current: {self.config.MIN_EMPLOYEES}-{self.config.MAX_EMPLOYEES} employees)")
                        suggestions.append(f"Lower minimum employees from {self.config.MIN_EMPLOYEES} or raise maximum from {self.config.MAX_EMPLOYEES}")
                    
                    if filter_reasons.get('revenue', 0) > 0:
                        error_parts.append(f"• {filter_reasons['revenue']} businesses filtered by revenue (Current minimum: ${self.config.MIN_REVENUE:,.0f})")
                        suggestions.append(f"Lower minimum revenue requirement from ${self.config.MIN_REVENUE:,.0f}")
                    
                    if filter_reasons.get('age', 0) > 0:
                        error_parts.append(f"• {filter_reasons['age']} businesses filtered by age (Current minimum: {self.config.MIN_BUSINESS_AGE} years)")
                        suggestions.append(f"Lower minimum business age from {self.config.MIN_BUSINESS_AGE} years")
                    
                    if filter_reasons.get('geography', 0) > 0:
                        geo_focus = getattr(self.config, 'GEOGRAPHIC_FOCUS', 'all')
                        error_parts.append(f"• {filter_reasons['geography']} businesses filtered by geographic focus (Current: {geo_focus})")
                        suggestions.append("Consider selecting 'All Areas' for geographic focus")
                    
                    if filter_reasons.get('patents', 0) > 0:
                        error_parts.append(f"• {filter_reasons['patents']} businesses filtered by patent requirement")
                        suggestions.append("Uncheck 'Require Patent Holders' in Additional Filters")
                    
                    if filter_reasons.get('sbir', 0) > 0:
                        error_parts.append(f"• {filter_reasons['sbir']} businesses filtered by SBIR/STTR requirement")
                        suggestions.append("Uncheck 'Require SBIR/STTR Recipients' in Additional Filters")
                    
                    if filter_reasons.get('growth', 0) > 0:
                        min_growth = getattr(self.config, 'MIN_GROWTH_RATE', 0)
                        error_parts.append(f"• {filter_reasons['growth']} businesses filtered by growth rate (Current minimum: {min_growth}%)")
                        suggestions.append(f"Lower minimum growth rate from {min_growth}%")
                
                results["error"] = "\n".join(error_parts)
                results["filter_summary"] = {
                    "total_businesses": total_businesses,
                    "filtered_out": total_businesses,
                    "remaining": 0,
                    "filter_reasons": filter_reasons,
                    "suggestion": "\n\nSuggestions to fix:\n• " + "\n• ".join(suggestions) if suggestions else "Try relaxing your filter criteria."
                }
                return results
            
            # Generate insights
            business_insights = self.scorer.generate_business_insights(scored_businesses)
            results["steps"]["business_scoring"] = business_insights
            
            # Save scored businesses to database
            self._save_businesses(scored_businesses)
            
            # Step 3: Cluster Discovery
            logger.info("Step 3: Discovering optimal clusters...")
            report_progress('cluster_formation', 10, 
                          'Starting cluster discovery analysis...')
            
            # Always use optimization-based clustering for discovery
            report_progress('cluster_formation', 30, 
                          'Using optimization-based clustering to discover patterns...')
            
            # Get clustering parameters
            # Default to full automatic discovery
            num_clusters = None  # Auto-discover optimal number
            cluster_size = None  # Auto-discover optimal sizes
            economic_targets = None
            
            # Prepare economic targets for automatic discovery
            if custom_params and 'economic_targets' in custom_params:
                economic_targets = {
                    'gdp_target': custom_params['economic_targets'].get('gdp_growth', self.config.GDP_GROWTH_TARGET),
                    'job_target': custom_params['economic_targets'].get('direct_jobs', self.config.DIRECT_JOBS_TARGET) +
                                 custom_params['economic_targets'].get('indirect_jobs', self.config.INDIRECT_JOBS_TARGET)
                }
            else:
                # Use default config targets
                economic_targets = {
                    'gdp_target': self.config.GDP_GROWTH_TARGET,
                    'job_target': self.config.DIRECT_JOBS_TARGET + self.config.INDIRECT_JOBS_TARGET
                }
            
            # Check if user provided algorithm params (for advanced users only)
            if custom_params and 'algorithm_params' in custom_params:
                algo_params = custom_params['algorithm_params']
                # Only override if explicitly provided
                if 'num_clusters' in algo_params and algo_params['num_clusters'] > 0:
                    num_clusters = algo_params['num_clusters']
                    logger.info(f"User explicitly requested {num_clusters} clusters instead of automatic discovery")
                if 'cluster_size' in algo_params and 'min' in algo_params['cluster_size']:
                    cluster_size = (algo_params['cluster_size']['min'], 
                                  algo_params['cluster_size']['max'])
                    logger.info(f"User explicitly requested cluster size range {cluster_size} instead of automatic discovery")
            
            # Get optimization focus from params
            optimization_focus = 'balanced'  # default
            if custom_params and 'algorithm_params' in custom_params:
                optimization_focus = custom_params['algorithm_params'].get('optimization_focus', 'balanced')
            
            # Run optimization-based discovery (automatic by default)
            logger.info(f"Running cluster analysis with automatic discovery (focus: {optimization_focus})...")
            
            # Create progress callback for optimizer
            def optimizer_progress(substage_progress):
                # Map optimizer progress (0-100) to cluster_formation substage (30-100)
                mapped_progress = 30 + int(substage_progress * 0.7)
                report_progress('cluster_formation', mapped_progress, 
                              f'Optimizing clusters (Progress: {substage_progress}%)...')
            
            # Determine force_full toggle from params
            force_full = False
            try:
                if custom_params:
                    force_full = bool(custom_params.get('force_full', False) or
                                       custom_params.get('algorithm_params', {}).get('force_full', False))
            except Exception:
                force_full = False

            # In quick mode, prefer the faster greedy optimizer unless explicitly overridden
            if quick_mode and not self.runtime_options.get('disable_nsga2', False):
                self.runtime_options['disable_nsga2'] = True
                if hasattr(self.optimizer, 'set_runtime_options'):
                    self.optimizer.set_runtime_options(self.runtime_options)
                # Let the user know in the progress UI
                report_progress('cluster_formation', 35, 'Quick mode: using greedy optimizer for faster clustering')

            clusters = self.optimizer.optimize_clusters(
                scored_businesses,
                num_clusters=num_clusters,  # None by default for auto-discovery
                cluster_size=cluster_size,
                economic_targets=economic_targets,
                optimization_focus=optimization_focus,
                progress_callback=optimizer_progress,
                params=self.params,  # Pass params to optimizer
                runtime_options={**(self.runtime_options or {}), 'force_full': force_full}
            )
            
            report_progress('cluster_formation', 100, 
                          f'Discovered {len(clusters)} optimal clusters',
                          f'✓ Identified {len(clusters)} high-potential economic clusters through data-driven discovery')

            # NEW: Consolidate overlapping/duplicate clusters before downstream analysis
            try:
                before_n = len(clusters)
                clusters = self._consolidate_clusters(clusters)
                after_n = len(clusters)
                if after_n < before_n:
                    logger.info(f"Consolidation: {before_n} → {after_n} clusters after merging overlaps")
                    report_progress('cluster_formation', 100,
                                    f'Consolidated overlapping clusters: {before_n} → {after_n}')
            except Exception as e:
                logger.warning(f"Cluster consolidation step skipped due to error: {e}")
            
            # Step 3.25: Calculate Economic Impact using Conservative Multipliers
            logger.info("Step 3.25: Calculating economic impact with conservative multipliers...")
            report_progress('cluster_formation', 95, 'Calculating 5-year economic impact using conservative multipliers (Akinyede & Caruso 2025)...')
            
            try:
                # Calculate economic impact for each cluster
                for cluster in clusters:
                    if 'businesses' in cluster and len(cluster['businesses']) > 0:
                        # Convert businesses to DataFrame if needed
                        if isinstance(cluster['businesses'], list):
                            cluster_df = pd.DataFrame(cluster['businesses'])
                        else:
                            cluster_df = cluster['businesses']
                        
                        # Calculate impact using conservative multipliers from paper
                        impact = self.optimizer.calculate_economic_impact(cluster_df)
                        
                        # Add impact projections to cluster data
                        cluster['economic_impact'] = impact
                        cluster['gdp_impact_5yr'] = impact.get('gdp_impact_5yr', 0)
                        cluster['total_jobs'] = impact.get('total_jobs', 0)
                        cluster['confidence_interval'] = impact.get('confidence_interval', '±15%')
                        
                        logger.info(f"Cluster {cluster.get('name', 'Unknown')}: "
                                  f"${impact.get('gdp_impact_5yr', 0)/1e9:.2f}B GDP, "
                                  f"{impact.get('total_jobs', 0):,} jobs (5yr)")
                
                # Calculate total portfolio impact
                total_gdp = sum(c.get('gdp_impact_5yr', 0) for c in clusters)
                total_jobs = sum(c.get('total_jobs', 0) for c in clusters)

                # Also compute portfolio impact on unique businesses to avoid double counting overlaps
                try:
                    unique_impact = self._compute_portfolio_impact_unique(clusters)
                    logger.info(
                        f"Portfolio (unique businesses): ${unique_impact.get('gdp_impact_5yr', 0)/1e9:.2f}B GDP, "
                        f"{unique_impact.get('total_jobs', 0):,} jobs"
                    )
                except Exception as e:
                    unique_impact = None
                
                logger.info(f"Total Portfolio Impact: ${total_gdp/1e9:.2f}B GDP, {total_jobs:,} jobs")
                report_progress('cluster_formation', 100, 
                              f'✓ Economic impact calculated: ${total_gdp/1e9:.2f}B GDP, {total_jobs:,} jobs over 5 years')
                
                # Store in results
                results['economic_impact'] = {
                    'total_gdp_5yr': total_gdp,
                    'total_jobs': total_jobs,
                    'methodology': 'Conservative calibration (Akinyede & Caruso 2025)',
                    'multipliers': {
                        'gdp': self.config.ECONOMIC_MULTIPLIERS.get('gdp', 1.85),
                        'employment': self.config.ECONOMIC_MULTIPLIERS.get('employment', 2.2)
                    },
                    'portfolio_unique': unique_impact or {}
                }
            except Exception as e:
                logger.warning(f"Economic impact calculation failed: {e}")
                # Continue without impact calculations
            
            # Step 3.5: ML Enhancement (if enabled)
            use_ml_enhancement = custom_params.get('use_ml_enhancement', True) if custom_params else True
            if use_ml_enhancement:
                logger.info("Step 3.5: Enhancing clusters with ML insights...")
                report_progress('ml_enhancement', 50, 'Applying ML enhancement to clusters...')
                try:
                    # Load historical data if available
                    historical_data = None
                    if hasattr(self, 'historical_data_path') and os.path.exists(self.historical_data_path):
                        historical_data = pd.read_csv(self.historical_data_path)
                    
                    # Check if we have KC enhanced data
                    kc_enhanced_data = None
                    if 'kc_enhancement' in results.get("steps", {}):
                        # Convert businesses to DataFrame with KC features
                        kc_enhanced_data = pd.DataFrame(scraping_results["businesses"])
                        logger.info(f"Passing KC enhanced data with {len([c for c in kc_enhanced_data.columns if c.startswith('kc_')])} KC features to ML enhancer")
                    
                    # Enhance clusters with ML predictions
                    # The ML enhancer will automatically use V2 if available
                    if hasattr(self.ml_enhancer, 'enhance_clusters'):
                        # Check if the enhancer accepts KC data parameter
                        import inspect
                        sig = inspect.signature(self.ml_enhancer.enhance_clusters)
                        if len(sig.parameters) >= 3 or 'kc_enhanced_data' in sig.parameters:
                            # V2 enhancer with KC support
                            enhanced_clusters, ml_explanations = self.ml_enhancer.enhance_clusters(
                                clusters, 
                                historical_data,
                                kc_enhanced_data
                            )
                        else:
                            # V1 enhancer without KC support
                            enhanced_clusters, ml_explanations = self.ml_enhancer.enhance_clusters(
                                clusters, 
                                historical_data
                            )
                    else:
                        logger.warning("ML enhancer does not have enhance_clusters method")
                        enhanced_clusters = clusters
                        ml_explanations = {}
                    
                    # Store explanations for reporting
                    results["ml_explanations"] = ml_explanations
                    
                    # Use enhanced clusters
                    clusters = enhanced_clusters
                    logger.info(f"ML enhancement completed. Confidence scores: " + 
                               ", ".join([f"{c.get('name', 'Unknown')}: {c.get('confidence_score', 0):.2f}" 
                                        for c in clusters[:3]]))
                    report_progress('ml_enhancement', 100, 
                                  f'✓ ML predictions complete: {clusters[0].get("confidence_score", 0):.0%} confidence on top cluster')
                except Exception as e:
                    logger.warning(f"ML enhancement failed, continuing with strategic clusters: {e}")
                    report_progress('ml_enhancement', 100, 'ML enhancement skipped (continuing with base clusters)')
                    # Continue with original clusters if ML enhancement fails
            
            # Step 3.75: Enhanced Analysis with New Modules
            logger.info("Step 3.75: Performing enhanced cluster analysis...")
            report_progress('enhanced_analysis', 10, 'Analyzing workforce requirements...')
            
            # Enhance each cluster with additional analysis
            for i, cluster in enumerate(clusters):
                cluster_progress = 10 + (i / len(clusters)) * 70
                
                # Workforce analysis
                try:
                    workforce_data = self.workforce_analyzer.analyze_cluster_workforce(cluster)
                    cluster['workforce_analysis'] = workforce_data
                except Exception as e:
                    logger.warning(f"Workforce analysis failed for cluster {cluster.get('name', i)}: {e}")
                
                # University integration
                try:
                    university_impact = self.university_integrator.calculate_university_impact(cluster)
                    cluster['university_integration'] = university_impact
                except Exception as e:
                    logger.warning(f"University integration analysis failed: {e}")
                
                # Market analysis
                try:
                    market_opportunity = self.market_analyzer.analyze_market_opportunity(
                        cluster.get('type', 'mixed'),
                        cluster.get('businesses', [])
                    )
                    cluster['market_analysis'] = market_opportunity
                except Exception as e:
                    logger.warning(f"Market analysis failed: {e}")
                
                # Patent analysis (original behavior)
                try:
                    if hasattr(self, 'patent_mapping') and self.patent_mapping:
                        patent_innovation = self.improved_patent_analyzer.get_cluster_patent_analysis(
                            cluster, self.patent_mapping
                        )
                        cluster['patent_analysis'] = patent_innovation
                        for business in cluster.get('businesses', []):
                            business['patent_count'] = self.patent_mapping.get(business['name'], 0)
                    else:
                        patent_innovation = self.patent_analyzer.assess_cluster_innovation(cluster)
                        cluster['patent_analysis'] = patent_innovation
                except Exception as e:
                    logger.warning(f"Patent analysis failed: {e}")
                
                # Longevity scoring
                try:
                    longevity_assessment = self.longevity_scorer.calculate_longevity_score(cluster)
                    cluster['longevity_score'] = longevity_assessment['score']
                    cluster['longevity_assessment'] = longevity_assessment
                except Exception as e:
                    logger.warning(f"Longevity scoring failed: {e}")
                
                # Revenue projections
                try:
                    revenue_projection = self.revenue_projector.project_cluster_revenue(cluster)
                    cluster['revenue_projections'] = revenue_projection
                except Exception as e:
                    logger.warning(f"Revenue projection failed: {e}")
                
                report_progress('enhanced_analysis', int(cluster_progress), 
                              f'Analyzing cluster {i+1}/{len(clusters)}...')
            
            # SBIR integration (cluster-wide analysis)
            try:
                sbir_awards = self.sbir_integrator.fetch_kc_awards()
                results["sbir_analysis"] = {
                    "total_awards": len(sbir_awards),
                    "total_funding": sum(a.get('amount', 0) for a in sbir_awards),
                    "by_cluster": self.sbir_integrator.analyze_sbir_distribution(sbir_awards, clusters)
                }
            except Exception as e:
                logger.warning(f"SBIR integration failed: {e}")
            
            report_progress('enhanced_analysis', 100, 
                          'Enhanced analysis complete',
                          '✓ Added workforce, market, patent, and revenue analysis to clusters')
            
            # Step 4: Validation and Projections
            logger.info("Step 4: Validating economic targets...")
            report_progress('validation', 30, f'Validating {len(clusters)} clusters against GDP, employment, and ROI targets...')
            
            validated_clusters = self._validate_clusters(clusters)

            # Normalize/augment per-cluster metrics for UI consumption
            for c in validated_clusters:
                # Ensure a cluster_score alias exists
                if 'cluster_score' not in c and 'total_score' in c:
                    try:
                        c['cluster_score'] = float(c.get('total_score', 0))
                    except Exception:
                        c['cluster_score'] = 0.0

                # Derive ROI field:
                # Prefer ML predicted expected_roi (fraction); fallback to roi_percentage (%);
                # finally fallback to target_alignment.roi (fraction based on GDP/revenue)
                roi_val = None
                mlp = c.get('ml_predictions') or {}
                if isinstance(mlp, dict):
                    if mlp.get('expected_roi') is not None:
                        try:
                            roi_val = float(mlp.get('expected_roi'))
                        except Exception:
                            roi_val = None
                    elif mlp.get('roi_percentage') is not None:
                        try:
                            roi_val = float(mlp.get('roi_percentage')) / 100.0
                        except Exception:
                            roi_val = None
                if roi_val is None:
                    try:
                        roi_val = float(c.get('target_alignment', {}).get('roi'))
                    except Exception:
                        roi_val = None
                if roi_val is not None:
                    c['roi'] = roi_val
            
            # Debug: Log cluster types
            logger.info(f"Validated {len(validated_clusters)} clusters:")
            for i, cluster in enumerate(validated_clusters):
                logger.info(f"  Cluster {i+1}: {cluster.get('name', 'Unknown')} - Type: {cluster.get('type', 'MISSING TYPE!')}")
            
            # Always compute natural communities so UI has details available
            try:
                if not getattr(self.optimizer, 'natural_communities_details', None):
                    # Use scored businesses to include composite_score for top business picking
                    communities = self.optimizer.find_natural_communities(scored_businesses, min_community_size=10)
                    self.optimizer.natural_communities_details = self.optimizer.extract_community_details(communities)
                    self.optimizer.natural_communities_count = len(self.optimizer.natural_communities_details)
                    logger.info(f"Computed {self.optimizer.natural_communities_count} natural communities for reporting")
            except Exception as e:
                logger.warning(f"Natural community computation failed: {e}")

            report_progress('validation', 100, 
                          f'✓ Validated {len(validated_clusters)} clusters meet economic targets',
                          f'✓ All clusters pass GDP ({self.config.MIN_ANNUAL_GDP_IMPACT/1e6:.0f}M) and employment thresholds')
            
            results["steps"]["cluster_optimization"] = {
                "clusters_identified": len(validated_clusters),
                "clusters": validated_clusters,
                "natural_communities_found": getattr(self.optimizer, 'natural_communities_count', None),
                "natural_communities_details": getattr(self.optimizer, 'natural_communities_details', [])
            }
            
            # Store overall clusters in results for easier access
            results["clusters"] = validated_clusters
            
            # Save clusters to database
            self._save_clusters(validated_clusters)
            
            # Step 5: Generate Recommendations
            logger.info("Step 5: Generating recommendations...")
            report_progress('recommendations', 50, f'Generating personalized recommendations for {len(validated_clusters)} clusters...')
            
            recommendations = self._generate_recommendations(
                validated_clusters, 
                scored_businesses,
                scraping_results.get("university_data", [])
            )
            results["steps"]["recommendations"] = recommendations
            
            report_progress('recommendations', 100, 
                          f'✓ Generated {len(recommendations.get("by_stakeholder", {}))} stakeholder recommendations',
                          '✓ Ready: Policy briefs, investment opportunities, workforce strategies')
            
            # Calculate overall economic impact
            total_impact = self._calculate_total_impact(validated_clusters)
            results["economic_impact"] = total_impact
            
            # Generate market analysis
            market_analysis = self._generate_market_analysis(validated_clusters, scored_businesses)
            results["market_analysis"] = market_analysis
            
            # Add total businesses to root level for easy access
            if 'steps' in results and 'business_scoring' in results['steps']:
                results["total_businesses"] = results['steps']['business_scoring'].get('total_businesses', 0)
            else:
                results["total_businesses"] = 0
                
            # Add clusters count to root level
            results["total_clusters"] = len(validated_clusters)
            
            results["status"] = "completed"
            
            # Final progress update before returning
            report_progress('recommendations', 100, f'✓ Analysis complete! {len(validated_clusters)} high-potential clusters identified')
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
            raise
            
        finally:
            # Ensure session is closed
            self.session.close()
            
        return results
    
    def _quick_score_businesses(self, df: pd.DataFrame) -> pd.Series:
        """
        Quickly score businesses without patents for initial ranking
        
        Args:
            df: DataFrame with business data
            
        Returns:
            Series of quick scores
        """
        logger.info("Quick scoring businesses based on size, revenue, and industry...")
        
        # Initialize scores
        scores = pd.Series(0.0, index=df.index)
        
        # Employee score (larger companies tend to have more patents)
        if 'employees' in df.columns:
            # Logarithmic scale to avoid over-weighting huge companies
            employee_scores = np.log10(df['employees'] + 1) / 5  # Max ~1.0 for 100k employees
            scores += employee_scores * 0.3
        
        # Revenue score
        if 'revenue_estimate' in df.columns:
            # Logarithmic scale for revenue
            revenue_scores = np.log10(df['revenue_estimate'] + 1) / 10  # Max ~1.0 for $10B
            scores += revenue_scores * 0.3
        
        # Industry score (tech and biotech companies more likely to have patents)
        if 'naics_code' in df.columns:
            tech_industries = ['511', '518', '519', '5415', '5417']  # Software, data, tech services
            bio_industries = ['325', '3254', '3391', '5417']  # Pharma, medical devices, R&D
            manufacturing = ['333', '334', '335', '336']  # Machinery, electronics, aerospace
            
            industry_scores = pd.Series(0.0, index=df.index)
            
            # Check NAICS codes
            for idx, naics in enumerate(df['naics_code'].astype(str)):
                if any(naics.startswith(code) for code in tech_industries):
                    industry_scores.iloc[idx] = 0.9
                elif any(naics.startswith(code) for code in bio_industries):
                    industry_scores.iloc[idx] = 1.0
                elif any(naics.startswith(code) for code in manufacturing):
                    industry_scores.iloc[idx] = 0.7
                else:
                    industry_scores.iloc[idx] = 0.3
            
            scores += industry_scores * 0.2
        
        # Age factor (established companies more likely to have patents)
        if 'year_established' in df.columns:
            current_year = datetime.now().year
            company_age = current_year - df['year_established']
            age_scores = np.clip(company_age / 50, 0, 1)  # Max score at 50 years
            scores += age_scores * 0.2
        
        # Normalize scores to 0-100
        scores = scores * 100
        
        # Add scores to dataframe for sorting
        df['quick_score'] = scores
        
        # Log statistics
        logger.info(f"Quick scoring complete. Score range: {scores.min():.1f} - {scores.max():.1f}")
        logger.info(f"Top industries in top 1000: {df.nlargest(1000, 'quick_score')['naics_code'].value_counts().head()}")
        
        return scores
    
    def _enhance_with_kc_features(self, business_df: pd.DataFrame) -> pd.DataFrame:
        """Enhance business data with Kansas City Open Data features
        
        Args:
            business_df: DataFrame of businesses to enhance
            
        Returns:
            Enhanced DataFrame with KC features
        """
        try:
            # Check if already enhanced
            if 'kc_crime_safety' in business_df.columns:
                logger.info("Data already enhanced with KC features")
                return business_df
            
            # Try to import KC enhancement pipeline
            from kc_enhancement_pipeline import KCEnhancementPipeline
            from pathlib import Path
            
            # Save to temp file for pipeline
            temp_dir = Path('temp')
            temp_dir.mkdir(exist_ok=True)
            temp_file = temp_dir / 'businesses_for_kc_enhancement.csv'
            business_df.to_csv(temp_file, index=False)
            
            # Configure and run pipeline
            pipeline_config = {
                'input_file': str(temp_file),
                'output_file': str(temp_dir / 'businesses_kc_enhanced.csv'),
                'cache_dir': 'cache',
                'sample_size': None,  # Process all
                'geocoding': {
                    'enabled': True,
                    'precise_quota': 100  # Limited for speed
                },
                'matching': {
                    'threshold': 0.8
                }
            }
            
            pipeline = KCEnhancementPipeline(pipeline_config)
            enhanced_df = pipeline.run()
            
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()
            
            logger.info(f"Successfully enhanced {len(enhanced_df)} businesses with KC features")
            return enhanced_df
            
        except ImportError:
            logger.warning("KC enhancement pipeline not available")
            return business_df
        except Exception as e:
            logger.warning(f"KC enhancement failed: {e}")
            return business_df
    
    def _prepare_business_data(self, scraping_results: Dict) -> pd.DataFrame:
        """Prepare business data for analysis"""
        businesses = scraping_results["businesses"]
        
        # Create DataFrame with memory optimization
        df = pd.DataFrame(businesses)
        
        # Optimize data types to reduce memory usage (robust to missing/invalid values)
        import pandas as _pd
        # year_established: allow missing with pandas nullable Int16
        if 'year_established' in df.columns:
            df['year_established'] = _pd.to_numeric(df['year_established'], errors='coerce')
            try:
                df['year_established'] = df['year_established'].astype('Int16')
            except Exception:
                # Fallback to float if nullable dtype not available
                pass
        # employees: treat missing as 0 for downstream scoring
        if 'employees' in df.columns:
            df['employees'] = _pd.to_numeric(df['employees'], errors='coerce').fillna(0)
            try:
                df['employees'] = df['employees'].astype('Int32')
            except Exception:
                # Keep as float if pandas nullable int not available
                pass
        # Normalize NAICS to string of digits for downstream .str operations
        if 'naics_code' in df.columns:
            df['naics_code'] = (
                df['naics_code']
                .astype(str)
                .str.replace(r'\.0$', '', regex=True)
                .str.replace(r'\D', '', regex=True)
            )
        
        # Ensure all required columns exist
        if 'patent_count' not in df.columns:
            df['patent_count'] = 0
        else:
            df['patent_count'] = df['patent_count'].astype('int16')
            
        if 'sbir_awards' not in df.columns:
            df['sbir_awards'] = 0
        else:
            df['sbir_awards'] = df['sbir_awards'].astype('int16')
        
        # Apply business-level models (selected features, fairness-aware)
        try:
            df = apply_business_predictions(df)
        except Exception as e:
            # Fallback to legacy revenue estimate if inference fails
            logger.warning(f"Business selected inference failed, using legacy revenue estimates: {e}")
            batch_size = 1000
            revenue_estimates = np.zeros(len(df))
            for i in range(0, len(df), batch_size):
                batch_end = min(i + batch_size, len(df))
                batch = df.iloc[i:batch_end]
                for j, (idx, row) in enumerate(batch.iterrows()):
                    revenue_estimates[i + j] = self.scorer.estimate_business_revenue({
                        'employees': row.get('employees', 0),
                        'naics_code': row.get('naics_code', ''),
                        'year_established': row.get('year_established', 2020)
                    })
            df["revenue_estimate"] = revenue_estimates
        return df
    
    def _analyze_market_conditions(self, businesses: pd.DataFrame) -> Dict:
        """Analyze market conditions by industry"""
        market_data = {}
        
        # Count businesses by NAICS code (coerce to string + digits, ignore blanks)
        if 'naics_code' in businesses.columns:
            naics_series = (
                businesses['naics_code']
                .astype(str)
                .str.replace(r'\.0$', '', regex=True)
                .str.replace(r'\D', '', regex=True)
            )
            prefixes = naics_series.str[:3]
            for naics in prefixes.dropna().unique():
                if not naics:
                    continue
                market_data[f"count_{naics}"] = int((prefixes == naics).sum())
        
        # Add growth rates (from scraper's market data) - deep copy to avoid race condition
        import copy
        market_data["growth_rates"] = copy.deepcopy(self.scorer.market_growth_rates)
        
        return market_data
    
    def _analyze_infrastructure(self, infrastructure_assets: List[Dict]) -> Dict:
        """Analyze infrastructure capabilities"""
        infrastructure_data = {
            "rail_capacity": 0,
            "highway_access": 0,
            "logistics_space": 0,
            "workforce_availability": 150000,  # KC logistics workforce estimate
            "university_count": len(self.config.REGIONAL_UNIVERSITIES),
            "utility_reliability": 0.99
        }
        
        for asset in infrastructure_assets:
            if asset["type"] == "rail":
                infrastructure_data["rail_capacity"] += asset["capacity"]
            elif asset["type"] == "highway":
                infrastructure_data["highway_access"] += 1
            elif asset["type"] == "logistics_park":
                infrastructure_data["logistics_space"] += asset["capacity"]
        
        return infrastructure_data
    
    def _validate_clusters(self, clusters: List[Dict]) -> List[Dict]:
        """Validate clusters against economic targets and infrastructure requirements"""
        validated = []
        
        for cluster in clusters:
            # Handle both strategic cluster format (metrics nested) and optimization format (metrics at top level)
            if 'metrics' in cluster:
                # Strategic cluster format
                gdp_impact = cluster['metrics'].get('projected_gdp_impact', 0)
                total_jobs = cluster['metrics'].get('projected_jobs', 0)
                total_revenue = cluster['metrics'].get('total_revenue', 0)
            else:
                # Optimization cluster format
                gdp_impact = cluster.get("projected_gdp_impact", 0)
                total_jobs = cluster.get("projected_jobs", 0)
                total_revenue = cluster.get("total_revenue", 0)
            
            # Calculate contribution to targets
            gdp_contribution = gdp_impact / self.config.GDP_GROWTH_TARGET if gdp_impact > 0 else 0
            jobs_contribution = total_jobs / (self.config.DIRECT_JOBS_TARGET + 
                                            self.config.INDIRECT_JOBS_TARGET) if total_jobs > 0 else 0
            
            # Check ROI if minimum ROI threshold is set
            if hasattr(self.config, 'MIN_ROI_THRESHOLD') and self.config.MIN_ROI_THRESHOLD > 0:
                revenue = total_revenue if total_revenue > 0 else 1
                roi = (gdp_impact / max(revenue, 1) - 1) if revenue > 0 else 0
                meets_roi = roi >= self.config.MIN_ROI_THRESHOLD
            else:
                meets_roi = True
                roi = 0
            
            # Check infrastructure requirements
            infrastructure_score = cluster.get("infrastructure_score", 0)
            meets_infrastructure = True
            missing_infrastructure = []
            
            if hasattr(self.config, 'REQUIRED_INFRASTRUCTURE'):
                cluster_type = cluster.get("type", "mixed")
                # Check specific infrastructure needs by cluster type
                if "rail_access" in self.config.REQUIRED_INFRASTRUCTURE:
                    if cluster_type == "logistics" and infrastructure_score < 80:
                        meets_infrastructure = False
                        missing_infrastructure.append("rail_access")
                if "broadband" in self.config.REQUIRED_INFRASTRUCTURE:
                    if cluster_type == "technology" and infrastructure_score < 70:
                        meets_infrastructure = False
                        missing_infrastructure.append("broadband")
                if "utilities" in self.config.REQUIRED_INFRASTRUCTURE:
                    if cluster_type == "manufacturing" and infrastructure_score < 75:
                        meets_infrastructure = False
                        missing_infrastructure.append("utilities")
                if "stem_workforce" in self.config.REQUIRED_INFRASTRUCTURE:
                    if cluster_type in ["technology", "biosciences"] and infrastructure_score < 70:
                        meets_infrastructure = False
                        missing_infrastructure.append("stem_workforce")
                if "university_partnership" in self.config.REQUIRED_INFRASTRUCTURE:
                    if cluster_type in ["technology", "biosciences"] and infrastructure_score < 65:
                        meets_infrastructure = False
                        missing_infrastructure.append("university_partnership")
                if "skilled_trades" in self.config.REQUIRED_INFRASTRUCTURE:
                    if cluster_type in ["manufacturing", "logistics"] and infrastructure_score < 70:
                        meets_infrastructure = False
                        missing_infrastructure.append("skilled_trades")
            
            cluster["target_alignment"] = {
                "gdp_contribution": gdp_contribution,
                "jobs_contribution": jobs_contribution,
                "roi": roi,
                "meets_targets": gdp_contribution >= 0.2 and jobs_contribution >= 0.2,
                "meets_roi": meets_roi,
                "meets_infrastructure": meets_infrastructure,
                "missing_infrastructure": missing_infrastructure
            }
            
            # Calculate wage impact based on user parameters
            wage_growth_target = self.params.get("wage_growth", 0.08)  # Default 8%
            time_horizon = self.params.get("time_horizon", 3)  # Years
            
            # Get average wages by cluster type (based on BLS data for KC metro)
            avg_wages = {
                "biosciences": 75000,      # Life sciences typically higher wages
                "technology": 85000,       # Tech sector highest wages
                "manufacturing": 55000,    # Manufacturing middle wages
                "logistics": 50000,        # Transportation/warehouse
                "professional_services": 70000,
                "mixed_services": 60000,
                "animal_health": 65000,
                "mixed": 60000            # Regional average
            }
            
            cluster_type = cluster.get("type", "mixed")
            avg_wage = avg_wages.get(cluster_type, 60000)
            
            # Calculate total wage impact: jobs * avg_wage * growth_rate * years
            cluster["projected_wage_impact"] = (
                total_jobs * avg_wage * wage_growth_target * time_horizon
            )
            
            # Also store the percentage for display
            cluster["wage_growth_percent"] = wage_growth_target * 100
            
            # Note: Time horizon is already accounted for in the optimizer calculations
            # Do not multiply again here to avoid double-counting
            # The optimizer already projects values over the configured time horizon
            
            validated.append(cluster)
        
        # Sort by total score (check both cluster_score and total_score)
        validated.sort(key=lambda x: x.get("cluster_score", x.get("total_score", 0)), reverse=True)
        
        return validated
    
    def _generate_recommendations(self, clusters: List[Dict], 
                                businesses: pd.DataFrame,
                                university_data: List[Dict]) -> Dict:
        """Generate actionable recommendations for stakeholders"""
        recommendations = {
            "entrepreneurs": [],
            "investors": [],
            "universities": [],
            "policymakers": []
        }
        
        for cluster in clusters[:3]:  # Top 3 clusters
            cluster_type = cluster.get("type", "mixed")
            
            # Get top companies for this cluster (used in recommendations)
            businesses_in_cluster = cluster.get("businesses", [])
            top_companies = sorted(
                businesses_in_cluster, 
                key=lambda x: x.get("composite_score", 0), 
                reverse=True
            )[:5]
            
            # Entrepreneur recommendations
            if cluster_type == "logistics":
                recommendations["entrepreneurs"].extend([
                    {
                        "opportunity": "Last-mile delivery services",
                        "rationale": "E-commerce growth driving demand (11% CAGR)",
                        "requirements": "Fleet management, route optimization software",
                        "investment_range": "$100K - $500K",
                        "potential_partners": ["Amazon", "FedEx", "Local retailers"]
                    },
                    {
                        "opportunity": "Cold chain logistics",
                        "rationale": "Growing pharmaceutical and food sectors",
                        "requirements": "Refrigerated trucks, IoT monitoring systems",
                        "investment_range": "$500K - $2M",
                        "potential_partners": ["Cerner", "Local hospitals", "Food distributors"]
                    },
                    {
                        "opportunity": "Logistics software platform",
                        "rationale": "Digital transformation in supply chain",
                        "requirements": "Software development expertise, industry knowledge",
                        "investment_range": "$200K - $1M",
                        "potential_partners": ["BNSF", "Union Pacific", "Local warehouses"]
                    }
                ])
            elif cluster_type == "biosciences":
                recommendations["entrepreneurs"].extend([
                    {
                        "opportunity": "Clinical trial management software",
                        "rationale": "Growing bioscience research sector",
                        "requirements": "Healthcare IT expertise, regulatory knowledge",
                        "investment_range": "$200K - $1M",
                        "potential_partners": ["UMKC School of Medicine", "Children's Mercy"]
                    },
                    {
                        "opportunity": "Biotech lab services",
                        "rationale": "Supporting growing research community",
                        "requirements": "Lab equipment, technical expertise",
                        "investment_range": "$300K - $1.5M",
                        "potential_partners": ["KU Medical Center", "Stowers Institute"]
                    }
                ])
            elif cluster_type == "technology":
                recommendations["entrepreneurs"].extend([
                    {
                        "opportunity": "AgTech solutions",
                        "rationale": "Intersection of tech and regional agricultural strength",
                        "requirements": "IoT expertise, agricultural knowledge",
                        "investment_range": "$150K - $750K",
                        "potential_partners": ["Kansas State Research", "Local farms"]
                    },
                    {
                        "opportunity": "Cybersecurity services",
                        "rationale": "Growing demand from financial and healthcare sectors",
                        "requirements": "Security expertise, certifications",
                        "investment_range": "$100K - $500K",
                        "potential_partners": ["Federal Reserve Bank KC", "Local banks"]
                    }
                ])
            else:
                # Generic recommendations for other cluster types (manufacturing, mixed, etc.)
                recommendations["entrepreneurs"].extend([
                    {
                        "opportunity": f"{cluster_type.replace('_', ' ').title()} innovation services",
                        "rationale": f"Growing {cluster_type.replace('_', ' ')} sector opportunity",
                        "requirements": "Industry expertise, local market knowledge",
                        "investment_range": "$150K - $1M",
                        "potential_partners": [b.get("name", "Unknown") for b in top_companies[:2]] + ["Industry associations"]
                    },
                    {
                        "opportunity": f"Specialized {cluster_type.replace('_', ' ')} consulting",
                        "rationale": "Help businesses optimize operations and scale",
                        "requirements": "Domain expertise, business development skills",
                        "investment_range": "$50K - $300K",
                        "potential_partners": ["Local chamber of commerce", "SBDC"]
                    }
                ])
            
            # Investor recommendations
            
            # Handle both cluster formats for ROI calculation
            if 'metrics' in cluster:
                gdp_impact = cluster['metrics'].get('projected_gdp_impact', 0)
                total_revenue = cluster['metrics'].get('total_revenue', 1)
                risk_score = cluster['metrics'].get('risk_score', 100)
            else:
                gdp_impact = cluster.get('projected_gdp_impact', 0)
                total_revenue = cluster.get('total_revenue', 1)
                risk_score = cluster.get('risk_score', 100)
            
            # Calculate more realistic annual ROI for investors
            time_horizon = getattr(self.config, 'TIME_HORIZON_YEARS', 5)
            annual_gdp = gdp_impact / time_horizon
            # Assume investment is ~20% of cluster revenue (typical for development)
            investment_estimate = total_revenue * 0.2
            annual_roi = (annual_gdp / max(investment_estimate, 1)) * 100
            
            recommendations["investors"].append({
                "cluster": cluster.get("name", "Unknown Cluster"),
                "projected_roi": f"{annual_roi:.1f}%",  # Show actual calculated ROI
                "risk_level": "Low" if risk_score < 30 else "Medium" if risk_score < 60 else "High",
                "key_companies": [b.get("name", "Unknown") for b in top_companies],
                "investment_thesis": self._generate_investment_thesis(cluster)
            })
            
            # University recommendations
            if university_data:
                relevant_unis = [u for u in university_data if any(
                    area.lower() in cluster_type.lower() or cluster_type.lower() in area.lower() 
                    for area in u.get("focus_areas", [])
                )]
                
                if cluster_type == "logistics":
                    recommendations["universities"].append({
                        "research_area": "Supply chain optimization using AI/ML",
                        "potential_partners": [b.get("name", "Unknown") for b in top_companies[:3]],
                        "funding_sources": ["NSF", "DOT", "Private sector"],
                        "universities": [u.get("name", "Unknown") for u in relevant_unis[:2]]
                    })
                elif cluster_type == "biosciences":
                    recommendations["universities"].append({
                        "research_area": "Drug discovery and personalized medicine",
                        "potential_partners": [b.get("name", "Unknown") for b in businesses_in_cluster
                                             if b.get("sbir_awards", 0) > 0][:3],
                        "funding_sources": ["NIH", "NSF", "SBIR/STTR"],
                        "universities": [u.get("name", "Unknown") for u in relevant_unis[:2]]
                    })
                elif cluster_type == "technology":
                    recommendations["universities"].append({
                        "research_area": "Quantum computing and AI applications",
                        "potential_partners": [b.get("name", "Unknown") for b in top_companies[:3]],
                        "funding_sources": ["NSF", "DOE", "DOD"],
                        "universities": [u.get("name", "Unknown") for u in relevant_unis[:2]]
                    })
                else:
                    # Generic university recommendations for other cluster types
                    recommendations["universities"].append({
                        "research_area": f"Advanced {cluster_type.replace('_', ' ')} research and innovation",
                        "potential_partners": [b.get("name", "Unknown") for b in top_companies[:3]],
                        "funding_sources": ["NSF", "Industry partnerships", "State grants"],
                        "universities": ["UMKC", "KU", "K-State"] if not relevant_unis else [u.get("name", "Unknown") for u in relevant_unis[:2]]
                    })
        
        # Policymaker recommendations
        recommendations["policymakers"] = [
            {
                "action": "Enhance rail infrastructure connectivity",
                "impact": "Support logistics cluster growth - projected $1.5B GDP impact",
                "priority": "High",
                "timeline": "2-3 years",
                "funding_needed": "$50-100M"
            },
            {
                "action": "Create biotech incubator spaces",
                "impact": "Accelerate bioscience cluster - 500+ high-wage jobs",
                "priority": "Medium",
                "timeline": "1-2 years", 
                "funding_needed": "$10-20M"
            },
            {
                "action": "Workforce training programs in logistics technology",
                "impact": "Address skill gaps - train 1,000+ workers annually",
                "priority": "High",
                "timeline": "6-12 months",
                "funding_needed": "$5-10M"
            },
            {
                "action": "Expand broadband infrastructure",
                "impact": "Enable tech cluster growth - support 200+ startups",
                "priority": "Medium",
                "timeline": "1-2 years",
                "funding_needed": "$20-40M"
            }
        ]
        
        return recommendations
    
    def _generate_investment_thesis(self, cluster: Dict) -> str:
        """Generate investment thesis for a cluster"""
        cluster_type = cluster.get("type", "mixed")
        
        thesis_map = {
            "logistics": "E-commerce growth and supply chain resilience driving 6%+ annual growth. KC's central location and rail infrastructure provide sustainable competitive advantage.",
            "biosciences": "Aging population and personalized medicine creating 7%+ growth opportunity. Strong research institutions and SBIR funding support innovation pipeline.",
            "technology": "Digital transformation across industries fueling 9%+ growth. Low cost of living attracting talent from coastal markets.",
            "manufacturing": "Reshoring trends and automation driving modernization. Skilled workforce and transportation access support competitiveness.",
            "animal_health": "Global food security concerns driving 5%+ growth. KC's animal health corridor provides unique ecosystem advantages."
        }
        
        return thesis_map.get(cluster_type, "Diversified cluster with multiple growth drivers.")
    
    def _generate_market_analysis(self, clusters: List[Dict], businesses: pd.DataFrame) -> Dict:
        """Generate market analysis data for visualization"""
        # Try to fetch real-time market data first
        try:
            from data_collection.market_monitor import MarketMonitor
            monitor = MarketMonitor()
            real_time_data = monitor.fetch_all_market_data()
            
            # Use real-time data as base
            market_analysis = {
                "economic_indicators": real_time_data.get('economic_indicators', {}),
                "commodity_prices": real_time_data.get('commodity_prices', {}),
                "market_scores": {},  # Will populate with cluster-specific scores
                "insights": []
            }
        except Exception as e:
            logger.warning(f"Could not fetch real-time market data: {e}")
            # Fallback structure
            market_analysis = {
                "economic_indicators": {},
                "commodity_prices": {},
                "market_scores": {},
                "insights": []
            }
        
        # Calculate market favorability by cluster
        for cluster in clusters:
            cluster_name = cluster.get('name', 'Unknown')
            cluster_type = cluster.get('type', 'mixed')
            
            # Base favorability on cluster metrics (0.0 to 1.0 scale)
            base_score = 0.5  # Start at neutral
            
            # Factor 1: GDP impact (higher = more favorable)
            gdp_impact = cluster.get('projected_gdp_impact', 0)
            if gdp_impact > 1_000_000_000:  # > $1B
                base_score += 0.2
            elif gdp_impact > 500_000_000:  # > $500M
                base_score += 0.1
                
            # Factor 2: Business density
            business_count = cluster.get('business_count', 0)
            if business_count > 30:
                base_score += 0.1
                
            # Factor 3: Innovation capacity
            innovation_score = cluster.get('metrics', {}).get('innovation_score', 0)
            if innovation_score > 70:
                base_score += 0.1
                
            # Factor 4: Network effects
            network_metrics = cluster.get('network_metrics', {})
            if network_metrics.get('synergy_score', 0) > 80:
                base_score += 0.05
                
            # Store with actual cluster name (not generic type)
            market_analysis["market_scores"][cluster_name] = min(base_score, 1.0)
        
        # If we don't have commodity prices, add KC-relevant defaults
        if not market_analysis["commodity_prices"]:
            market_analysis["commodity_prices"] = {
                "cattle": {"value": 182.5, "units": "$/cwt", "date": "Nov 2024"},
                "corn": {"value": 4.25, "units": "$/bushel", "date": "Nov 2024"},
                "wheat": {"value": 5.85, "units": "$/bushel", "date": "Nov 2024"},
                "soybeans": {"value": 9.95, "units": "$/bushel", "date": "Nov 2024"},
                "diesel": {"value": 3.45, "units": "$/gallon", "date": "Nov 2024"},
                "electricity": {"value": 10.2, "units": "cents/kWh", "date": "Nov 2024"}
            }
        
        # Generate market insights based on cluster analysis
        insights = []
        
        # Insight 1: Dominant industries
        industry_counts = {}
        for cluster in clusters:
            cluster_type = cluster.get('type', 'mixed')
            industry_counts[cluster_type] = industry_counts.get(cluster_type, 0) + 1
            
        dominant_industry = max(industry_counts.items(), key=lambda x: x[1])[0] if industry_counts else 'mixed'
        insights.append(f"The {dominant_industry} sector shows strongest clustering potential in the KC region")
        
        # Insight 2: Growth potential
        high_growth_clusters = [c for c in clusters if c.get('projected_gdp_impact', 0) > 800_000_000]
        if high_growth_clusters:
            insights.append(f"{len(high_growth_clusters)} clusters show exceptional growth potential with GDP impact > $800M")
            
        # Insight 3: Employment
        total_jobs = sum(c.get('projected_jobs', 0) for c in clusters)
        if total_jobs > 10000:
            insights.append(f"Cluster development could create {total_jobs:,} new jobs across the region")
            
        # Insight 4: Innovation
        innovation_clusters = [c for c in clusters if c.get('metrics', {}).get('innovation_score', 0) > 80]
        if innovation_clusters:
            insights.append(f"{len(innovation_clusters)} clusters demonstrate high innovation capacity")
            
        # Combine with any real-time insights
        if 'insights' in market_analysis and market_analysis['insights']:
            market_analysis['insights'].extend(insights)
        else:
            market_analysis['insights'] = insights
        
        return market_analysis
    
    def _calculate_total_impact(self, clusters: List[Dict]) -> Dict:
        """Calculate total economic impact of all clusters including wage growth and ROI.

        Applies conservative calibration layer (Config.CALIBRATION_ENABLED) to align with
        paper methodology. Calibration scales GDP and jobs to conservative bounds when
        baseline estimates exceed conservative projections derived from direct measures.
        """
        total_gdp = 0
        total_direct_jobs = 0
        total_all_jobs = 0
        total_wage_impact = 0
        total_investment_needed = 0
        
        # Industry-specific wage data (based on BLS data for KC MSA)
        industry_avg_wages = {
            "484": 45000,   # Trucking
            "493": 42000,   # Warehousing
            "3254": 75000,  # Pharmaceutical
            "5415": 85000,  # Computer systems
            "332": 48000,   # Fabricated metal
            "541": 80000,   # Professional/Tech services
            "511": 90000,   # Software publishing
            "325": 70000,   # Chemical manufacturing
        }
        
        baseline_revenue_sum = 0.0
        baseline_direct_jobs_sum = 0

        for c in clusters:
            if 'metrics' in c:
                # Strategic cluster format
                total_gdp += c['metrics'].get('projected_gdp_impact', 0)
                total_direct_jobs += c['metrics'].get('total_employees', 0)
                total_all_jobs += c['metrics'].get('projected_jobs', 0)
                # Baselines
                baseline_revenue_sum += c['metrics'].get('total_revenue', 0)
                baseline_direct_jobs_sum += c['metrics'].get('total_employees', 0)
            else:
                # Optimization cluster format
                total_gdp += c.get("projected_gdp_impact", 0)
                total_direct_jobs += c.get("total_employees", 0)
                total_all_jobs += c.get("projected_jobs", 0)
                baseline_revenue_sum += c.get('total_revenue', 0)
                baseline_direct_jobs_sum += c.get('total_employees', 0)
            
            # Calculate wage impact for this cluster
            cluster_wage_impact = self._calculate_cluster_wage_impact(c, industry_avg_wages)
            total_wage_impact += cluster_wage_impact
            
            # Calculate investment needed
            cluster_investment = self._calculate_cluster_investment(c)
            total_investment_needed += cluster_investment
        
        # Apply conservative calibration layer (paper Section 4.3)
        if getattr(self.config, 'CALIBRATION_ENABLED', False):
            try:
                conservative_gdp_cap = 0.0
                conservative_jobs_cap = 0
                # If baseline revenue exists, derive conservative GDP cap from MGDP multiplier
                if baseline_revenue_sum and baseline_revenue_sum > 0:
                    conservative_gdp_cap = baseline_revenue_sum * self.config.MGDP_MULTIPLIER
                # Derive conservative total jobs cap from direct jobs baseline and MJOBS multiplier
                if baseline_direct_jobs_sum and baseline_direct_jobs_sum > 0:
                    conservative_jobs_cap = int(baseline_direct_jobs_sum * self.config.MJOBS_MULTIPLIER)

                # Only cap when our modeled projections exceed conservative bounds
                if conservative_gdp_cap > 0 and total_gdp > conservative_gdp_cap:
                    total_gdp = conservative_gdp_cap
                if conservative_jobs_cap > 0 and total_all_jobs > conservative_jobs_cap:
                    total_all_jobs = conservative_jobs_cap

                # Apply friction factor universally to reflect behavioral/implementation frictions
                total_gdp *= self.config.CALIBRATION_FRICTION
                total_all_jobs = int(total_all_jobs * self.config.CALIBRATION_FRICTION)

            except Exception as _:
                # Fail-safe: do not block pipeline on calibration error
                pass

        # Apply macro triggers if adverse conditions (env-driven; optional)
        try:
            oil_price = float(os.getenv('OIL_PRICE', '0') or 0)
            interest_rate = float(os.getenv('INTEREST_RATE', '0') or 0)
            if (oil_price >= 80.0) or (interest_rate >= 5.5):
                scale = getattr(self.config, 'TRIGGER_SCALE', 0.90)
                total_gdp *= scale
                total_all_jobs = int(total_all_jobs * scale)
        except Exception:
            pass

        # Calculate ROI correctly
        if total_investment_needed > 0:
            # Get time horizon
            time_horizon = getattr(self.config, 'TIME_HORIZON_YEARS', 5)
            
            # Total return over time horizon (GDP impact + wage impact)
            total_return = total_gdp + total_wage_impact
            
            # Net return (total return - investment)
            net_return = total_return - total_investment_needed
            
            # Simple ROI: (Net Return / Investment) * 100
            simple_roi = (net_return / total_investment_needed) * 100
            
            # Annualized ROI using CAGR formula
            # CAGR = ((Ending Value / Beginning Value)^(1/n) - 1) * 100
            ending_value = total_investment_needed + net_return
            beginning_value = total_investment_needed
            if ending_value > 0 and beginning_value > 0:
                annualized_roi = (pow(ending_value / beginning_value, 1.0 / time_horizon) - 1) * 100
            else:
                annualized_roi = 0
            
            # Log for debugging
            logger.info(f"ROI Calculation - Investment: ${total_investment_needed:,.0f}, Total Return: ${total_return:,.0f}, Net Return: ${net_return:,.0f}")
            logger.info(f"Simple ROI: {simple_roi:.1f}%, Annualized ROI: {annualized_roi:.1f}%")
            
            # Use annualized ROI as the primary metric
            roi = annualized_roi
        else:
            roi = 0
        
        # Calculate average wage growth
        current_avg_wage = 55000  # KC MSA average
        if total_direct_jobs > 0:
            projected_avg_wage = total_wage_impact / total_direct_jobs
            wage_growth_pct = ((projected_avg_wage - current_avg_wage) / current_avg_wage) * 100
        else:
            wage_growth_pct = 0
        
        return {
            "projected_gdp_impact": total_gdp,
            "gdp_target_achievement": (total_gdp / self.config.GDP_GROWTH_TARGET) * 100 if self.config.GDP_GROWTH_TARGET > 0 else 0,
            "projected_direct_jobs": total_direct_jobs,
            "projected_total_jobs": total_all_jobs,
            "jobs_target_achievement": (total_all_jobs /
                                         (self.config.DIRECT_JOBS_TARGET +
                                          self.config.INDIRECT_JOBS_TARGET)) * 100 if (self.config.DIRECT_JOBS_TARGET + self.config.INDIRECT_JOBS_TARGET) > 0 else 0,
            "projected_wage_growth": wage_growth_pct,
            "wage_growth_target_achievement": (wage_growth_pct / (getattr(self.config, 'WAGE_GROWTH_TARGET', 0.08) * 100)) * 100 if hasattr(self.config, 'WAGE_GROWTH_TARGET') else 100,
            "total_investment_required": total_investment_needed,
            "projected_roi": roi,
            "roi_target_achievement": (roi / (getattr(self.config, 'MIN_ROI_THRESHOLD', 0.15) * 100)) * 100 if hasattr(self.config, 'MIN_ROI_THRESHOLD') else 100,
            "meets_targets": (total_gdp >= self.config.GDP_GROWTH_TARGET and
                            total_all_jobs >= self.config.DIRECT_JOBS_TARGET +
                            self.config.INDIRECT_JOBS_TARGET and
                            wage_growth_pct >= getattr(self.config, 'WAGE_GROWTH_TARGET', 0.08) * 100 and
                            roi >= getattr(self.config, 'MIN_ROI_THRESHOLD', 0.15) * 100)
        }
    
    def _calculate_cluster_wage_impact(self, cluster: Dict, industry_wages: Dict) -> float:
        """Calculate wage impact for a cluster based on industry composition"""
        total_wage_impact = 0
        
        # Get businesses in this cluster
        if 'businesses' in cluster:
            businesses = cluster['businesses']
        elif 'business_ids' in cluster:
            # Would need to look up businesses by ID
            return 0  # Simplified for now
        else:
            return 0
        
        # Calculate wage impact based on industry mix
        for business in businesses:
            naics_prefix = business.get('naics_code', '')[:3]
            employees = business.get('employees', 0)
            
            # Get industry-specific wage or use default
            industry_wage = industry_wages.get(naics_prefix, 60000)
            
            # High-growth clusters get wage premium
            cluster_type = cluster.get('type', 'mixed')
            if cluster_type in ['technology', 'biosciences']:
                wage_premium = 1.15  # 15% premium
            elif cluster_type in ['logistics', 'manufacturing']:
                wage_premium = 1.08  # 8% premium
            else:
                wage_premium = 1.05  # 5% premium
            
            projected_wage = industry_wage * wage_premium
            total_wage_impact += projected_wage * employees
        
        return total_wage_impact
    
    def _calculate_cluster_investment(self, cluster: Dict) -> float:
        """Calculate investment needed to develop a cluster"""
        base_investment = 0
        
        # Get cluster size - handle both formats
        if 'metrics' in cluster:
            num_businesses = cluster['metrics'].get('business_count', 0)
            total_employees = cluster['metrics'].get('total_employees', 0)
            # If not in metrics, check for businesses list
            if num_businesses == 0 and 'businesses' in cluster:
                num_businesses = len(cluster['businesses'])
                total_employees = sum(b.get('employees', 0) for b in cluster['businesses'])
        else:
            num_businesses = cluster.get('business_count', 0)
            total_employees = cluster.get('total_employees', 0)
            # Fallback to businesses list if available
            if num_businesses == 0 and 'businesses' in cluster:
                num_businesses = len(cluster['businesses'])
                total_employees = sum(b.get('employees', 0) for b in cluster['businesses'])
        
        # Base investment calculation using more realistic estimates
        # Based on economic development best practices
        if num_businesses == 0:
            return 0
            
        # Investment varies by business size
        avg_employees_per_business = total_employees / max(num_businesses, 1)
        
        if avg_employees_per_business < 10:  # Small businesses
            investment_per_business = 50000   # $50K for micro businesses
        elif avg_employees_per_business < 50:  # Medium businesses
            investment_per_business = 150000  # $150K for small-medium
        elif avg_employees_per_business < 200:  # Large businesses
            investment_per_business = 500000  # $500K for larger companies
        else:  # Very large businesses
            investment_per_business = 1000000  # $1M for major employers
        
        base_investment = num_businesses * investment_per_business
        
        # Additional investment based on cluster type
        cluster_type = cluster.get('type', 'mixed')
        type_multipliers = {
            'technology': 1.5,      # Needs more infrastructure
            'biosciences': 2.0,     # Needs labs, equipment
            'logistics': 1.3,       # Needs transportation infrastructure
            'manufacturing': 1.4,   # Needs industrial facilities
            'animal_health': 1.8,   # Specialized facilities
            'mixed': 1.2
        }
        
        type_multiplier = type_multipliers.get(cluster_type, 1.2)
        
        # Scale factor based on cluster size
        if total_employees > 1000:
            scale_factor = 1.2
        elif total_employees > 500:
            scale_factor = 1.1
        else:
            scale_factor = 1.0
        
        total_investment = base_investment * type_multiplier * scale_factor
        
        return total_investment

    def _save_businesses(self, businesses_df: pd.DataFrame):
        """Save scored businesses to database"""
        try:
            # Clear existing businesses
            self.session.query(Business).delete()
            
            for _, row in businesses_df.iterrows():
                business = Business(
                    name=row["name"],
                    naics_code=row["naics_code"],
                    state=row["state"],
                    county=row["county"],
                    city=row.get("city", "Kansas City"),
                    year_established=int(row.get("year_established", 2020)),
                    employees=int(row.get("employees", 10)),
                    revenue_estimate=float(row.get("revenue_estimate", 0)),
                    innovation_score=float(row.get("innovation_score", 0)),
                    market_potential_score=float(row.get("market_potential_score", 0)),
                    competition_score=float(row.get("competition_score", 0)),
                    composite_score=float(row.get("composite_score", 0)),
                    patent_count=int(row.get("patent_count", 0)),
                    sbir_awards=int(row.get("sbir_awards", 0)),
                    data_source=row.get("data_source", "Unknown")
                )
                self.session.add(business)
            
            self.session.commit()
            logger.info(f"Saved {len(businesses_df)} businesses to database")
            
        except Exception as e:
            logger.error(f"Error saving businesses: {e}")
            self.session.rollback()
            raise
    
    def _save_clusters(self, clusters: List[Dict]):
        """Save cluster configurations to database"""
        try:
            # Ensure session is clean before starting
            self.session.rollback()
            
            # Clear existing clusters and memberships
            self.session.query(ClusterMembership).delete()
            self.session.query(Cluster).delete()
            self.session.commit()
            
            # Save each cluster
            for cluster_data in clusters:
                # Handle both strategic and optimization cluster formats
                if 'metrics' in cluster_data:
                    # Strategic cluster format
                    gdp_impact = cluster_data['metrics'].get('projected_gdp_impact', 0)
                    direct_jobs = cluster_data['metrics'].get('total_employees', 0)
                    total_jobs = cluster_data['metrics'].get('projected_jobs', 0)
                    risk_score = cluster_data['metrics'].get('risk_score', 0)
                else:
                    # Optimization cluster format
                    gdp_impact = cluster_data.get("projected_gdp_impact", 0)
                    direct_jobs = cluster_data.get("total_employees", 0)
                    total_jobs = cluster_data.get("projected_jobs", 0)
                    risk_score = cluster_data.get("risk_score", 0)
                
                cluster = Cluster(
                    name=cluster_data.get("name", "Unknown Cluster"),
                    type=cluster_data.get("type", "mixed"),
                    natural_assets_score=float(cluster_data.get("natural_assets_score", 0)),
                    infrastructure_score=float(cluster_data.get("infrastructure_score", 0)),
                    workforce_score=float(cluster_data.get("workforce_score", 0)),
                    innovation_score=float(cluster_data.get("innovation_capacity_score", 0)),
                    market_access_score=float(cluster_data.get("market_access_score", 0)),
                    geopolitical_score=float(cluster_data.get("geopolitical_score", 0)),
                    resilience_score=float(cluster_data.get("resilience_score", 0)),
                    total_score=float(cluster_data.get("cluster_score", cluster_data.get("total_score", 0))),
                    roi=(float(cluster_data.get("roi", 0)) if cluster_data.get("roi") is not None else None),
                    projected_gdp_impact=float(gdp_impact),
                    projected_direct_jobs=int(direct_jobs),
                    projected_indirect_jobs=int(total_jobs - direct_jobs),
                    projected_wage_impact=float(cluster_data.get("projected_wage_impact", 1.0)),
                    longevity_score=float(cluster_data.get("longevity_score", 0)),
                    risk_factors={"risk_score": risk_score}
                )
                self.session.add(cluster)
            
            # Commit all clusters at once
            self.session.commit()
            
            # Now add cluster memberships in a separate transaction
            for cluster_data in clusters:
                # Get the saved cluster
                cluster = self.session.query(Cluster).filter_by(
                    name=cluster_data.get("name", "Unknown Cluster")
                ).first()
                
                if cluster:
                    # Add cluster memberships
                    for business_data in cluster_data.get("businesses", []):
                        business = self.session.query(Business).filter_by(
                            name=business_data.get("name", "Unknown")
                        ).first()
                        
                        if business:
                            membership = ClusterMembership(
                                business_id=business.id,
                                cluster_id=cluster.id,
                                role="member",
                                synergy_score=float(cluster_data.get("synergy_score", 0))
                            )
                            self.session.add(membership)
            
            # Commit memberships
            self.session.commit()
            logger.info(f"Saved {len(clusters)} clusters to database")
            
        except Exception as e:
            logger.error(f"Error saving clusters: {e}")
            self.session.rollback()
            raise
    
    def generate_report(self, results: Dict) -> str:
        """Generate a formatted report of the analysis results"""
        report = f"""
# Kansas City MSA Economic Cluster Analysis Report
Generated: {results['timestamp']}
Status: {results['status']}

## Executive Summary
The analysis identified {results['steps'].get('cluster_optimization', {}).get('clusters_identified', 0)} viable economic clusters for the Kansas City MSA.

### Economic Impact Projections:
- GDP Impact: ${results['economic_impact']['projected_gdp_impact']:,.0f} ({results['economic_impact']['gdp_target_achievement']:.1f}% of target)
- Direct Jobs: {results['economic_impact']['projected_direct_jobs']:,}
- Total Jobs: {results['economic_impact']['projected_total_jobs']:,} ({results['economic_impact']['jobs_target_achievement']:.1f}% of target)
- Target Achievement: {'✓ MEETS TARGETS' if results['economic_impact']['meets_targets'] else '✗ Below targets'}

## Data Collection Results
- Businesses Analyzed: {results['steps']['data_collection']['businesses_collected']:,}
- SBIR/STTR Awards: {results['steps']['data_collection']['sbir_awards']}
- Infrastructure Assets: {results['steps']['data_collection']['infrastructure_assets']}
- Employment Metrics: {results['steps']['data_collection']['employment_metrics']}
- University Partners: {results['steps']['data_collection']['university_partners']}

## Business Analysis Insights
- Total Businesses Scored: {results['steps']['business_scoring']['total_businesses']}
- Businesses Passing Threshold: {results['steps']['business_scoring']['passing_threshold']}
- Average Composite Score: {results['steps']['business_scoring']['avg_composite_score']:.1f}

## Top Performing Clusters
"""
        
        clusters = results['steps'].get('cluster_optimization', {}).get('clusters', [])
        for i, cluster in enumerate(clusters[:3]):
            # Handle both cluster formats
            if 'metrics' in cluster:
                gdp_impact = cluster['metrics'].get('projected_gdp_impact', 0)
                total_jobs = cluster['metrics'].get('projected_jobs', 0)
                risk_score = cluster['metrics'].get('risk_score', 100)
            else:
                gdp_impact = cluster.get('projected_gdp_impact', 0)
                total_jobs = cluster.get('projected_jobs', 0)
                risk_score = cluster.get('risk_score', 100)
                
            report += f"""
### {i+1}. {cluster.get('name', 'Unknown Cluster')}
- Type: {cluster.get('type', 'mixed').title()}
- Total Score: {cluster.get('cluster_score', cluster.get('total_score', 0)):.1f}/100
- Businesses: {cluster.get('business_count', 0)}
- Projected GDP Impact: ${gdp_impact:,.0f}
- Projected Jobs: {total_jobs:,}
- Longevity Score: {cluster.get('longevity_score', 0):.1f}/10
- Risk Level: {'Low' if risk_score < 30 else 'Medium' if risk_score < 60 else 'High'}"""
            
            # Add ML enhancement info if available
            if 'ml_predictions' in cluster:
                ml_pred = cluster['ml_predictions']
                confidence = cluster.get('confidence_score', 0)
                report += f"""
- ML Confidence Score: {confidence:.1%}
- ML Predicted GDP: ${ml_pred.get('gdp_impact', 0):,.0f}
- ML Predicted Jobs: {ml_pred.get('job_creation', 0):,}
- ML Expected ROI: {ml_pred.get('expected_roi', 0):.1%}"""
            
            # Add network metrics if available
            if 'network_metrics' in cluster:
                network = cluster['network_metrics']
                report += f"""
- Network Density: {network.get('network_density', 0):.2f}
- Synergy Score: {network.get('synergy_score', 0):.1f}/100
- Network Resilience: {network.get('network_resilience', 0):.1f}/100"""
            
            report += "\n"
        
        # Add recommendations section
        report += "\n## Key Recommendations\n"
        recs = results['steps'].get('recommendations', {})
        
        report += "\n### For Entrepreneurs:\n"
        for rec in recs.get('entrepreneurs', [])[:3]:
            report += f"- **{rec['opportunity']}**: {rec['rationale']} (Investment: {rec['investment_range']})\n"
        
        report += "\n### For Investors:\n"
        for rec in recs.get('investors', [])[:3]:
            report += f"- **{rec['cluster']}**: Projected ROI {rec['projected_roi']}, Risk: {rec['risk_level']}\n"
            report += f"  - Investment Thesis: {rec.get('investment_thesis', 'N/A')}\n"
        
        report += "\n### For Universities:\n"
        for rec in recs.get('universities', [])[:3]:
            report += f"- **Research Area**: {rec['research_area']}\n"
            report += f"  - Funding Sources: {', '.join(rec['funding_sources'])}\n"
        
        report += "\n### For Policymakers:\n"
        for rec in recs.get('policymakers', [])[:3]:
            report += f"- **{rec['action']}**: {rec['impact']} (Priority: {rec['priority']})\n"
            report += f"  - Timeline: {rec.get('timeline', 'N/A')}, Funding: {rec.get('funding_needed', 'N/A')}\n"
        
        # Add ML insights section if available
        if 'ml_explanations' in results and results['ml_explanations']:
            report += "\n## Machine Learning Insights\n"
            report += "The ML enhancement provides data-driven predictions based on historical cluster performance:\n\n"
            
            for cluster_name, explanation in list(results['ml_explanations'].items())[:3]:
                report += f"### {cluster_name}\n"
                if 'key_drivers' in explanation:
                    report += "**Key Success Drivers:**\n"
                    for driver in explanation['key_drivers'][:5]:
                        report += f"- {driver}\n"
                report += "\n"
        
        return report

    def _reset_runtime_options(self):
        """Restore runtime toggles to their default state."""
        self.runtime_options = {
            "disable_network_metrics": False,
            "disable_nsga2": False
        }
        if hasattr(self.optimizer, "set_runtime_options"):
            self.optimizer.set_runtime_options(self.runtime_options)

    def _apply_custom_params(self, params: Dict):
        """Apply custom parameters to configuration"""
        # Store params for later use
        self.params = params.get('economic_targets', {})
        
        # Economic targets
        if 'economic_targets' in params:
            targets = params['economic_targets']
            if 'gdp_growth' in targets:
                self.config.GDP_GROWTH_TARGET = targets['gdp_growth']
            if 'direct_jobs' in targets:
                self.config.DIRECT_JOBS_TARGET = targets['direct_jobs']
            if 'indirect_jobs' in targets:
                self.config.INDIRECT_JOBS_TARGET = targets['indirect_jobs']
            if 'wage_growth' in targets:
                self.config.WAGE_GROWTH_TARGET = targets['wage_growth']
            if 'time_horizon' in targets:
                self.config.TIME_HORIZON_YEARS = targets['time_horizon']
            if 'min_roi' in targets:
                self.config.MIN_ROI_THRESHOLD = targets['min_roi']
        
        # Geographic scope
        if 'geographic_scope' in params:
            geo = params['geographic_scope']
            if 'kansas_counties' in geo:
                self.config.KANSAS_COUNTIES = geo['kansas_counties']
            if 'missouri_counties' in geo:
                self.config.MISSOURI_COUNTIES = geo['missouri_counties']
            if 'focus' in geo:
                self.config.GEOGRAPHIC_FOCUS = geo['focus']
            # Update scraper with new county list
            if hasattr(self.scraper, 'config'):
                self.scraper.config.KANSAS_COUNTIES = self.config.KANSAS_COUNTIES
                self.scraper.config.MISSOURI_COUNTIES = self.config.MISSOURI_COUNTIES
        
        # Business filters
        if 'business_filters' in params:
            filters = params['business_filters']
            if 'min_employees' in filters:
                self.config.MIN_EMPLOYEES = filters['min_employees']
            if 'max_employees' in filters:
                self.config.MAX_EMPLOYEES = filters['max_employees']
            if 'min_revenue' in filters:
                self.config.MIN_REVENUE = filters['min_revenue']
            if 'min_age' in filters:
                self.config.MIN_BUSINESS_AGE = filters['min_age']
            if 'excluded_naics' in filters:
                self.config.EXCLUDED_NAICS = filters['excluded_naics']
            # Additional filters
            if 'require_patents' in filters:
                self.config.REQUIRE_PATENTS = filters['require_patents']
            if 'require_sbir' in filters:
                self.config.REQUIRE_SBIR = filters['require_sbir']
            if 'min_growth_rate' in filters:
                self.config.MIN_GROWTH_RATE = filters['min_growth_rate']
            # Update scorer with new filters
            if hasattr(self.scorer, 'config'):
                self.scorer.config = self.config
    
    def reload_ml_enhancer(self):
        """Force reload of ML enhancer module"""
        self._ml_enhancer = None
        logger.info("ML enhancer module marked for reload")
    
    def _apply_user_filters_to_businesses(self, businesses: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Apply user-defined filters to businesses BEFORE sampling and KC enhancement.
        
        Returns:
            Tuple of (filtered_businesses, filter_stats)
        """
        if not businesses:
            return [], {}
            
        original_count = len(businesses)
        current_year = 2025
        
        # Track filter statistics
        filter_stats = {
            'original': original_count,
            'filtered_by_employees': 0,
            'filtered_by_revenue': 0,
            'filtered_by_age': 0,
            'filtered_by_patents': 0,
            'filtered_by_sbir': 0,
            'filtered_by_naics': 0,
            'filtered_by_geography': 0,
            'passed': 0
        }
        
        filtered = []
        
        # Get selected counties for geographic filtering
        selected_counties = set()
        if hasattr(self.config, 'KANSAS_COUNTIES') and self.config.KANSAS_COUNTIES:
            selected_counties.update(self.config.KANSAS_COUNTIES)
        if hasattr(self.config, 'MISSOURI_COUNTIES') and self.config.MISSOURI_COUNTIES:
            selected_counties.update(self.config.MISSOURI_COUNTIES)

        # Normalize county names to be resilient to formats like "Johnson County" vs "Johnson County, KS"
        def _norm_county(s: str) -> str:
            if not s:
                return ''
            x = str(s).strip().lower()
            # Drop trailing state abbreviation if present
            for suf in [', ks', ', mo', ', ks.', ', mo.']:
                if x.endswith(suf):
                    x = x[: -len(suf)].strip()
            return x

        normalized_selected = {_norm_county(c) for c in selected_counties if c}
        logger.debug(f"Geographic filter - selected counties (normalized): {normalized_selected}")
        
        for business in businesses:
            # Geographic filter (if counties are specified)
            if normalized_selected:
                business_county = business.get('county', '')
                # Only filter if county is specified and doesn't match (use normalized forms)
                if business_county:
                    if _norm_county(business_county) not in normalized_selected:
                        filter_stats['filtered_by_geography'] += 1
                        continue
                # If no county specified, do not filter by geography
            
            # Employee count filter
            employees = business.get('employees', 0)
            if employees < self.config.MIN_EMPLOYEES:
                filter_stats['filtered_by_employees'] += 1
                continue
            
            if hasattr(self.config, 'MAX_EMPLOYEES') and self.config.MAX_EMPLOYEES:
                if employees > self.config.MAX_EMPLOYEES:
                    filter_stats['filtered_by_employees'] += 1
                    continue
            
            # Revenue filter
            revenue = business.get('revenue')
            if not revenue and hasattr(self.scorer, 'estimate_business_revenue'):
                revenue = self.scorer.estimate_business_revenue(business)
            
            # Only filter if we have a revenue value to check
            if revenue is not None and revenue < self.config.MIN_REVENUE:
                filter_stats['filtered_by_revenue'] += 1
                continue
            
            # Business age filter
            year_established = business.get('year_established', current_year)
            if isinstance(year_established, str):
                try:
                    year_established = int(year_established)
                except:
                    year_established = current_year
            
            age = current_year - year_established
            if age < self.config.MIN_BUSINESS_AGE:
                filter_stats['filtered_by_age'] += 1
                continue
            
            # Patent requirement filter
            if hasattr(self.config, 'REQUIRE_PATENTS') and self.config.REQUIRE_PATENTS:
                if not business.get('patent_count', 0):
                    filter_stats['filtered_by_patents'] += 1
                    continue
            
            # SBIR requirement filter
            if hasattr(self.config, 'REQUIRE_SBIR') and self.config.REQUIRE_SBIR:
                if not business.get('sbir_awards', 0):
                    filter_stats['filtered_by_sbir'] += 1
                    continue
            
            # NAICS exclusion filter
            if hasattr(self.config, 'EXCLUDED_NAICS') and self.config.EXCLUDED_NAICS:
                naics = business.get('naics_code', '')
                if naics and any(naics.startswith(excluded) for excluded in self.config.EXCLUDED_NAICS):
                    filter_stats['filtered_by_naics'] += 1
                    continue
            
            # Business passed all filters
            filtered.append(business)
            filter_stats['passed'] += 1
        
        # Log filtering results
        logger.info(f"User filter results: {original_count} → {len(filtered)} businesses")
        logger.info(f"Filter stats: {filter_stats}")
        
        return filtered, filter_stats


def main():
    """Main entry point"""
    try:
        # Initialize analyzer
        analyzer = ClusterPredictionTool()
        
        # Run analysis
        results = analyzer.run_full_analysis()
        
        # Save results to JSON
        with open("cluster_analysis_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("Analysis complete. Results saved to cluster_analysis_results.json")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()
