"""Cluster optimization module using NSGA-II algorithm"""
import numpy as np
from datetime import datetime as _dt
import pandas as pd
from deap import base, creator, tools, algorithms
from typing import List, Dict, Tuple, Optional
import random
import logging
from collections import defaultdict
import sys
import os
import math
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Optional network imports for Louvain communities
try:
    import networkx as nx
    from networkx.algorithms import community as nx_community
    _HAS_NETWORKX = True
except Exception:
    _HAS_NETWORKX = False

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models import Cluster, ClusterMembership

logger = logging.getLogger(__name__)

# Note: DEAP fitness and individual will be created dynamically based on optimization focus

class ClusterOptimizer:
    """Optimizes cluster formation using multi-objective genetic algorithm"""
    
    def __init__(self):
        self.config = Config()
        self.optimization_focus = 'balanced'  # default
        # Use configurable cluster scoring weights
        self.cluster_weights = self.config.CLUSTER_SCORING_WEIGHTS
        self.runtime_options = {
            "disable_network_metrics": False,
            "disable_nsga2": False
        }

    def _default_runtime_options(self) -> Dict[str, bool]:
        """Return the default runtime toggle configuration."""
        return {
            "disable_network_metrics": False,
            "disable_nsga2": False
        }

    def set_runtime_options(self, options: Optional[Dict] = None) -> Dict[str, bool]:
        """Merge and store runtime toggles controlling optional subsystems."""
        merged = self._default_runtime_options()
        if options:
            for key in merged:
                if key in options:
                    merged[key] = bool(options[key])
        self.runtime_options = merged
        return self.runtime_options

    def _runtime_flag(self, key: str) -> bool:
        """Helper to read boolean runtime flags safely."""
        return bool(self.runtime_options.get(key, False))
        
    def setup_deap(self, optimization_focus='balanced'):
        """Setup DEAP framework for multi-objective optimization with dynamic weights"""
        # Clear any existing definitions
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual
        
        # Define weights based on optimization focus
        # Objectives: (GDP, Innovation, Jobs, Synergies, Risk, Critical Mass)
        # Critical Mass rewards larger clusters for ecosystem effects
        if optimization_focus == 'gdp':
            weights = (2.0, 0.5, 0.5, 0.5, -0.5, 0.8)  # Heavy GDP, good mass
        elif optimization_focus == 'jobs':
            weights = (0.5, 0.5, 2.0, 0.5, -0.5, 0.8)  # Heavy jobs, good mass
        elif optimization_focus == 'innovation':
            weights = (0.5, 2.0, 0.5, 0.5, -0.5, 0.8)  # Heavy innovation, good mass
        else:  # balanced
            weights = (1.0, 1.0, 1.0, 1.0, -1.0, 1.2)  # Equal weights, prioritize mass
        
        # Create fitness and individual classes with appropriate weights
        creator.create("FitnessMulti", base.Fitness, weights=weights)
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        
        self.toolbox = base.Toolbox()
    
    def determine_optimal_clusters(self, businesses: pd.DataFrame, 
                                   economic_targets: Optional[Dict] = None,
                                   min_k: int = 2, 
                                   max_k: int = 10) -> Dict:
        """
        Automatically determine optimal number of clusters using multiple validation metrics
        
        Args:
            businesses: DataFrame of eligible businesses
            economic_targets: Optional dict with 'gdp_target', 'job_target' keys
            min_k: Minimum number of clusters to test
            max_k: Maximum number of clusters to test
            
        Returns:
            Dict with optimal k and validation metrics
        """
        logger.info("Starting automatic cluster number determination...")
        
        # Prepare features for clustering analysis
        features = self._prepare_clustering_features(businesses)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Ensure max_k is feasible
        max_feasible_k = min(max_k, len(businesses) // 15)  # At least 15 businesses per cluster
        if max_feasible_k < min_k:
            logger.warning(f"Not enough businesses for {min_k} clusters. Setting k=2")
            return {'optimal_k': 2, 'reason': 'Insufficient businesses for more clusters'}
        
        # Test different values of k
        validation_results = []
        
        for k in range(min_k, max_feasible_k + 1):
            logger.info(f"Testing k={k} clusters...")
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(scaled_features)
            
            # Calculate validation metrics
            metrics = {}
            
            # Silhouette Score (higher is better, -1 to 1)
            if k > 1:
                metrics['silhouette'] = silhouette_score(scaled_features, labels)
            else:
                metrics['silhouette'] = 0
            
            # Davies-Bouldin Score (lower is better)
            if k > 1:
                metrics['davies_bouldin'] = davies_bouldin_score(scaled_features, labels)
            else:
                metrics['davies_bouldin'] = float('inf')
            
            # Calinski-Harabasz Score (higher is better)
            if k > 1:
                metrics['calinski_harabasz'] = calinski_harabasz_score(scaled_features, labels)
            else:
                metrics['calinski_harabasz'] = 0
            
            # Economic viability assessment
            metrics['economic_viability'] = self._assess_economic_viability(
                businesses, labels, k, economic_targets
            )
            
            # Combined score (normalized)
            metrics['combined_score'] = self._calculate_combined_score(metrics)
            
            validation_results.append({
                'k': k,
                'metrics': metrics,
                'labels': labels
            })
            
            logger.info(f"k={k}: Silhouette={metrics['silhouette']:.3f}, "
                       f"Combined={metrics['combined_score']:.3f}")
        
        # Select optimal k
        optimal_result = self._select_optimal_k(validation_results, economic_targets)
        
        # Generate reasoning
        reasoning = self._generate_cluster_reasoning(validation_results, optimal_result)
        
        return {
            'optimal_k': optimal_result['k'],
            'metrics': optimal_result['metrics'],
            'reasoning': reasoning,
            'all_results': validation_results
        }
    
    def detect_communities_louvain(self, businesses: pd.DataFrame, min_modularity: float = 0.30) -> Dict:
        """
        Apply Louvain algorithm for network-based community detection
        Based on Akinyede & Caruso (2025) methodology
        
        Args:
            businesses: DataFrame of businesses to cluster
            min_modularity: Minimum modularity Q score (paper uses 0.30)
            
        Returns:
            Dictionary with communities and modularity score
        """
        try:
            import networkx as nx
            # Try to import python-louvain
            try:
                import community as community_louvain
                LOUVAIN_AVAILABLE = True
            except ImportError:
                logger.warning("python-louvain not installed, falling back to natural communities")
                LOUVAIN_AVAILABLE = False
                
            if not LOUVAIN_AVAILABLE:
                # Fall back to existing method
                return self.find_natural_communities(businesses)
                
            logger.info("Building business network for Louvain community detection...")
            
            # Create weighted network
            G = nx.Graph()
            
            # Add nodes (businesses) with attributes
            for idx, business in businesses.iterrows():
                G.add_node(idx, **business.to_dict())
            
            # Calculate edge weights based on multiple factors
            from itertools import combinations
            import numpy as np
            
            edges_added = 0
            for i, j in combinations(businesses.index, 2):
                bus_i = businesses.loc[i]
                bus_j = businesses.loc[j]
                
                # Geographic proximity (30% weight)
                geo_weight = 0
                if bus_i.get('county') == bus_j.get('county'):
                    geo_weight = 0.3
                elif bus_i.get('state') == bus_j.get('state'):
                    geo_weight = 0.15
                    
                # Industry similarity (30% weight) - using NAICS codes
                naics_i = str(bus_i.get('naics_code', ''))[:3]
                naics_j = str(bus_j.get('naics_code', ''))[:3]
                ind_weight = 0
                if naics_i == naics_j:
                    ind_weight = 0.3
                elif naics_i[:2] == naics_j[:2]:  # Same 2-digit sector
                    ind_weight = 0.15
                    
                # Size compatibility (20% weight)
                emp_i = bus_i.get('employees', 1)
                emp_j = bus_j.get('employees', 1)
                size_ratio = min(emp_i, emp_j) / max(emp_i, emp_j)
                size_weight = 0.2 * size_ratio
                
                # Supply chain linkage (20% weight) - simplified
                supply_chain_weight = 0
                # Check if industries typically interact
                supply_pairs = [
                    ('484', '493'),  # Trucking & Warehousing
                    ('311', '445'),  # Food mfg & Food retail
                    ('325', '621'),  # Chemical & Healthcare
                    ('332', '336'),  # Metal & Auto
                ]
                for pair in supply_pairs:
                    if (naics_i in pair and naics_j in pair):
                        supply_chain_weight = 0.2
                        break
                
                # Calculate total edge weight
                total_weight = geo_weight + ind_weight + size_weight + supply_chain_weight
                
                # Only add edge if weight is significant
                if total_weight > 0.25:
                    G.add_edge(i, j, weight=total_weight)
                    edges_added += 1
            
            logger.info(f"Network created: {len(G.nodes())} nodes, {edges_added} edges")
            
            # Apply Louvain algorithm
            partition = community_louvain.best_partition(G, weight='weight')
            modularity = community_louvain.modularity(partition, G, weight='weight')
            
            logger.info(f"Louvain modularity Q = {modularity:.3f}")
            
            # Check if modularity meets threshold
            if modularity < min_modularity:
                logger.warning(f"Modularity {modularity:.3f} below threshold {min_modularity}")
            
            # Convert partition to communities
            communities_dict = {}
            for node, comm_id in partition.items():
                if comm_id not in communities_dict:
                    communities_dict[comm_id] = []
                communities_dict[comm_id].append(node)
            
            # Create community DataFrames
            communities = []
            for comm_id, node_indices in communities_dict.items():
                if len(node_indices) >= 20:  # Minimum community size
                    community_df = businesses.loc[node_indices].copy()
                    community_df['louvain_community'] = comm_id
                    community_df['modularity_score'] = modularity
                    communities.append(community_df)
            
            logger.info(f"Found {len(communities)} communities via Louvain (Q={modularity:.3f})")
            
            return {
                'communities': communities,
                'modularity': modularity,
                'partition': partition,
                'network': G,
                'method': 'louvain'
            }
            
        except Exception as e:
            logger.error(f"Louvain detection failed: {e}")
            # Fall back to existing method
            return {'communities': self.find_natural_communities(businesses), 'method': 'fallback'}
    
    def find_natural_communities(self, businesses: pd.DataFrame, min_community_size: int = 10) -> List[pd.DataFrame]:
        """
        Identify natural business communities based on organic relationships
        
        Natural communities form around:
        1. Geographic proximity (same county)
        2. Industry relationships (supply chain connections)
        3. Size similarity (peer businesses)
        
        Returns:
            List of DataFrames, each representing a natural community
        """
        logger.info("Finding natural business communities...")
        
        communities = []
        
        # Group by county first (geographic communities)
        for county, county_businesses in businesses.groupby('county'):
            if len(county_businesses) < min_community_size:
                continue
                
            # Within each county, find industry clusters
            # Define industry relationships (who trades with whom)
            industry_relationships = {
                '511': ['518', '519', '541'],  # Software â†’ Data/Info/Tech services
                '518': ['511', '519', '541'],  # Data â†’ Software/Info/Tech services  
                '493': ['484', '488', '492'],  # Warehousing â†’ Transportation
                '484': ['493', '488', '492'],  # Trucking â†’ Warehousing/Transport
                '336': ['332', '333', '811'],  # Auto mfg â†’ Metal/Machinery/Repair
                '311': ['493', '484', '445'],  # Food mfg â†’ Distribution/Retail
                '325': ['3254', '3391', '621'], # Chemical â†’ Pharma/Medical/Healthcare
            }
            
            # Create sub-communities by industry relationships
            processed = set()
            
            for primary_naics in county_businesses['naics_code'].str[:3].unique():
                if primary_naics in processed:
                    continue
                    
                # Find all related businesses
                related_naics = industry_relationships.get(primary_naics, [])
                related_naics.append(primary_naics)  # Include self
                
                # Select businesses in this industry cluster
                industry_mask = county_businesses['naics_code'].str[:3].isin(related_naics)
                industry_community = county_businesses[industry_mask]
                
                if len(industry_community) >= min_community_size:
                    # Further segment by size (small/medium/large)
                    size_segments = []
                    
                    # Define size categories by employees
                    small = industry_community[industry_community['employees'] < 50]
                    medium = industry_community[(industry_community['employees'] >= 50) & 
                                               (industry_community['employees'] < 500)]
                    large = industry_community[industry_community['employees'] >= 500]
                    
                    for segment, label in [(small, 'small'), (medium, 'medium'), (large, 'large')]:
                        if len(segment) >= min_community_size:
                            # Add community metadata
                            segment = segment.copy()
                            segment['community_type'] = f"{county}_{primary_naics}_{label}"
                            segment['community_county'] = county
                            segment['community_industry'] = primary_naics
                            segment['community_size_class'] = label
                            communities.append(segment)
                            
                            logger.debug(f"Found community: {county} - {primary_naics} - {label} "
                                       f"({len(segment)} businesses)")
                
                # Mark as processed
                processed.update(related_naics)
        
        # Add any remaining businesses as "mixed" communities
        all_assigned = pd.concat(communities) if communities else pd.DataFrame()
        unassigned = businesses[~businesses.index.isin(all_assigned.index)]
        
        if len(unassigned) >= min_community_size:
            for county, county_unassigned in unassigned.groupby('county'):
                if len(county_unassigned) >= min_community_size:
                    community = county_unassigned.copy()
                    community['community_type'] = f"{county}_mixed_varied"
                    community['community_county'] = county
                    community['community_industry'] = 'mixed'
                    community['community_size_class'] = 'varied'
                    communities.append(community)
        
        logger.info(f"Found {len(communities)} natural communities with "
                   f"{sum(len(c) for c in communities)} total businesses")
        
        return communities
    
    def extract_community_details(self, communities: List[pd.DataFrame]) -> List[Dict]:
        """Extract detailed information about each natural community for reporting"""
        community_details = []
        
        # Industry name mapping for better readability
        industry_names = {
            '484': 'Trucking Transportation',
            '493': 'Warehousing & Storage',
            '488': 'Support Activities for Transportation',
            '511': 'Software Publishing',
            '518': 'Data Processing & Hosting',
            '541': 'Professional & Technical Services',
            '332': 'Fabricated Metal Products',
            '333': 'Machinery Manufacturing',
            '336': 'Transportation Equipment Mfg',
            '311': 'Food Manufacturing',
            '325': 'Chemical Manufacturing',
            '3254': 'Pharmaceutical Manufacturing',
            'mixed': 'Mixed Industries'
        }
        
        for i, community in enumerate(communities):
            if len(community) == 0:
                continue
                
            # Get community metadata
            comm_type = community.iloc[0].get('community_type', 'Unknown')
            county = community.iloc[0].get('community_county', 'Unknown')
            primary_industry = community.iloc[0].get('community_industry', 'mixed')
            size_class = community.iloc[0].get('community_size_class', 'varied')
            
            # Calculate community statistics
            industries = community['naics_code'].str[:3].value_counts()
            top_industries = industries.head(3).to_dict()
            
            # Format industry breakdown
            industry_breakdown = []
            for naics, count in top_industries.items():
                industry_name = industry_names.get(naics, f'NAICS {naics}')
                percentage = (count / len(community)) * 100
                industry_breakdown.append(f"{industry_name} ({percentage:.0f}%)")
            
            # Determine grouping reason
            grouping_reasons = []
            if len(industries) == 1:
                grouping_reasons.append("Same industry")
            elif primary_industry != 'mixed':
                grouping_reasons.append("Supply chain relationships")
            grouping_reasons.append(f"Geographic proximity ({county})")
            if size_class != 'varied':
                grouping_reasons.append(f"Similar business size ({size_class})")
            
            from datetime import datetime as _dt
            detail = {
                'id': i + 1,
                'name': f"Community {i+1}: {county} - {industry_names.get(primary_industry, primary_industry).title()} ({size_class.title()})",
                'county': county,
                'business_count': len(community),
                'primary_industries': industry_breakdown,
                'avg_employees': int(community['employees'].mean()),
                'total_revenue': community['revenue_estimate'].sum(),
                'avg_business_age': int(_dt.now().year - community['year_established'].mean()),
                'size_distribution': {
                    'small': len(community[community['employees'] < 50]),
                    'medium': len(community[(community['employees'] >= 50) & (community['employees'] < 500)]),
                    'large': len(community[community['employees'] >= 500])
                },
                'grouping_reasons': grouping_reasons,
                'top_businesses': community.nlargest(3, 'composite_score')[['name', 'employees']].to_dict('records')
            }
            
            community_details.append(detail)
        
        return community_details
    
    def calculate_economic_impact(self, cluster_businesses: pd.DataFrame, years: int = 5) -> Dict:
        """
        Calculate conservative economic impact projections using paper's multipliers
        Based on Akinyede & Caruso (2025) methodology

        Fixed: Employment multiplier is applied only to incremental job growth to
        avoid double-counting growth and multiplier effects.
        """
        # Get conservative multipliers from config
        gdp_multiplier = self.config.ECONOMIC_MULTIPLIERS.get('gdp', 1.85)
        employment_multiplier = self.config.ECONOMIC_MULTIPLIERS.get('employment', 2.2)

        # Calculate base metrics
        total_revenue = cluster_businesses['revenue_estimate'].sum()
        baseline_employees = cluster_businesses['employees'].sum()
        num_businesses = len(cluster_businesses)

        # Estimate growth rate based on industry mix
        avg_growth_rate = 0.06  # Default 6% annual growth

        # Calculate 5-year GDP projections with conservative multipliers
        direct_gdp = total_revenue * (1 + avg_growth_rate) ** years
        total_gdp_impact = direct_gdp * gdp_multiplier

        # Employment projections (jobs grow slightly slower than GDP)
        job_growth_rate = avg_growth_rate * 0.8
        future_direct_jobs = baseline_employees * math.pow(1 + job_growth_rate, years)
        direct_job_growth = max(0.0, future_direct_jobs - baseline_employees)
        indirect_job_growth = direct_job_growth * max(0.0, employment_multiplier - 1)
        total_jobs_impact = baseline_employees + direct_job_growth + indirect_job_growth
        # For reporting
        direct_jobs = baseline_employees + direct_job_growth
        indirect_jobs = indirect_job_growth
        
        # Apply regional strategic multipliers if applicable
        cluster_industries = cluster_businesses['naics_code'].str[:4].value_counts()
        strategic_multiplier = 1.0
        
        # Check for strategic sectors
        for naics_prefix, count in cluster_industries.head(5).items():
            if naics_prefix.startswith('3254'):  # Pharmaceutical
                strategic_multiplier = max(strategic_multiplier, self.config.REGIONAL_STRATEGIC_MULTIPLIERS.get('biosciences', 1.25))
            elif naics_prefix.startswith('54'):  # Tech services
                strategic_multiplier = max(strategic_multiplier, self.config.REGIONAL_STRATEGIC_MULTIPLIERS.get('technology', 1.20))
            elif naics_prefix.startswith('484') or naics_prefix.startswith('493'):  # Logistics
                strategic_multiplier = max(strategic_multiplier, self.config.REGIONAL_STRATEGIC_MULTIPLIERS.get('logistics', 1.15))
            elif naics_prefix.startswith('52'):  # Finance
                strategic_multiplier = max(strategic_multiplier, self.config.REGIONAL_STRATEGIC_MULTIPLIERS.get('fintech', 1.15))
        
        # Apply strategic multiplier to impacts and recompute to preserve identities
        total_gdp_impact *= strategic_multiplier
        total_jobs_scaled = total_jobs_impact * strategic_multiplier
        direct_jobs_scaled = direct_jobs * strategic_multiplier
        # Round consistently and enforce identity: total = direct + indirect
        total_jobs_int = int(round(total_jobs_scaled))
        direct_jobs_int = int(round(direct_jobs_scaled))
        indirect_jobs_int = max(0, total_jobs_int - direct_jobs_int)
        
        # Calculate confidence intervals (±15% for conservative estimates)
        confidence_level = 0.15
        gdp_lower = total_gdp_impact * (1 - confidence_level)
        gdp_upper = total_gdp_impact * (1 + confidence_level)
        jobs_lower = int(round(total_jobs_int * (1 - confidence_level)))
        jobs_upper = int(round(total_jobs_int * (1 + confidence_level)))
        
        return {
            'gdp_impact_5yr': total_gdp_impact,
            'gdp_impact_lower': gdp_lower,
            'gdp_impact_upper': gdp_upper,
            'total_jobs': total_jobs_int,
            'direct_jobs': direct_jobs_int,
            'indirect_jobs': indirect_jobs_int,
            'jobs_lower': jobs_lower,
            'jobs_upper': jobs_upper,
            'num_businesses': num_businesses,
            'base_revenue': total_revenue,
            'base_employees': baseline_employees,
            'multipliers_used': {
                'gdp': gdp_multiplier,
                'employment': employment_multiplier,
                'strategic': strategic_multiplier
            },
            'confidence_interval': f"±{confidence_level*100:.0f}%",
            'methodology': 'Conservative calibration (Akinyede & Caruso 2025)'
        }

    def _safe_float(self, value, default=0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    def _geo_weight(self, lat1, lon1, lat2, lon2) -> float:
        """Compute geographic proximity weight ~[0,1], higher if closer. Uses a simple inverse distance."""
        try:
            lat1 = float(lat1); lon1 = float(lon1); lat2 = float(lat2); lon2 = float(lon2)
        except Exception:
            return 0.0
        d = ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5
        return float(max(0.0, min(1.0, 1.0 / (1.0 + (d / 0.2)))))

    def find_louvain_communities(self, businesses: pd.DataFrame, min_community_size: int = 15) -> List[pd.DataFrame]:
        """Build a business graph with weighted edges and run Louvain to identify communities."""
        if not _HAS_NETWORKX:
            raise RuntimeError("networkx not available")

        df = businesses.copy()
        if df.empty:
            return []

        for col in ['name', 'naics_code', 'employees', 'county']:
            if col not in df.columns:
                df[col] = '' if col in ['name', 'naics_code', 'county'] else 0

        G = nx.Graph()
        for idx, row in df.iterrows():
            G.add_node(int(idx),
                       name=row.get('name', ''),
                       naics=str(row.get('naics_code', '')),
                       employees=int(row.get('employees', 0)),
                       county=row.get('county', ''),
                       lat=self._safe_float(row.get('latitude', None), None),
                       lon=self._safe_float(row.get('longitude', None), None))

        supply_map = {
            '336': {'332', '333'},
            '311': {'111', '112'},
            '325': {'324', '326', '339'},
            '511': {'334', '541'},
            '484': {'493', '488'}
        }

        nodes = list(G.nodes())
        naics3 = df['naics_code'].astype(str).str[:3].fillna('')
        index_to_naics3 = dict(zip(df.index, naics3))

        for i_idx, i in enumerate(nodes):
            i_attrs = G.nodes[i]
            i_naics3 = index_to_naics3.get(i, '')
            for j in nodes[i_idx+1:]:
                j_attrs = G.nodes[j]
                j_naics3 = index_to_naics3.get(j, '')

                w_ind = 1.0 if i_naics3 and i_naics3 == j_naics3 else 0.0
                w_sup = 1.0 if (i_naics3 in supply_map and j_naics3 in supply_map[i_naics3]) or \
                               (j_naics3 in supply_map and i_naics3 in supply_map[j_naics3]) else 0.0
                ei = max(1, i_attrs.get('employees', 0))
                ej = max(1, j_attrs.get('employees', 0))
                size_ratio = min(ei, ej) / max(ei, ej)
                w_size = float(size_ratio)
                w_geo = self._geo_weight(i_attrs.get('lat', None), i_attrs.get('lon', None),
                                         j_attrs.get('lat', None), j_attrs.get('lon', None))
                if w_geo == 0.0 and i_attrs.get('county') and i_attrs.get('county') == j_attrs.get('county'):
                    w_geo = 0.3

                weight = 0.4 * w_geo + 0.3 * w_ind + 0.2 * w_size + 0.1 * w_sup
                if weight > 0.0:
                    G.add_edge(i, j, weight=float(weight))

        if G.number_of_edges() == 0:
            return []

        parts = nx_community.louvain_communities(G, weight='weight', seed=42)
        communities: List[pd.DataFrame] = []
        for com in parts:
            members = list(com)
            if len(members) >= min_community_size:
                communities.append(df.loc[members].copy())

        return communities
    
    def _prepare_clustering_features(self, businesses: pd.DataFrame) -> np.ndarray:
        """Extract and prepare features for clustering analysis including infrastructure"""
        features = []
        
        # Core business metrics
        features.append(businesses['composite_score'].values)
        features.append(np.log1p(businesses['employees'].values))
        features.append(np.log1p(businesses['revenue_estimate'].values))
        
        # Innovation metrics if available
        if 'patent_count' in businesses.columns:
            features.append(np.log1p(businesses['patent_count'].fillna(0).values))
        if 'sbir_awards' in businesses.columns:
            features.append(np.log1p(businesses['sbir_awards'].fillna(0).values))
        
        # Add infrastructure proximity features (objective geographic facts)
        infrastructure_features = self._calculate_infrastructure_features(businesses)
        for infra_feature in infrastructure_features:
            features.append(infra_feature)
        
        # Industry representation (simplified)
        # Get top 10 most common NAICS prefixes
        # Safely handle NAICS codes
        valid_naics = businesses['naics_code'].astype(str).replace({'nan': '', 'None': ''})
        valid_naics = valid_naics[valid_naics.str.len() >= 3]
        naics_prefixes = valid_naics.str[:3]
        
        # Get top NAICS codes, handling empty case
        if not naics_prefixes.empty:
            top_naics = naics_prefixes.value_counts().head(10).index
            for prefix in top_naics:
                features.append((businesses['naics_code'].astype(str).str[:3] == prefix).astype(float).values)
        else:
            # If no valid NAICS codes, add a dummy feature
            features.append(np.zeros(len(businesses)))
        
        return np.column_stack(features)
    
    def _get_safe_naics_prefix(self, naics_series, length=3, default="000"):
        """Safely extract NAICS prefix handling edge cases
        
        Args:
            naics_series: Series of NAICS codes
            length: Number of digits to extract (default 3)
            default: Default value if extraction fails
            
        Returns:
            Most common NAICS prefix or default if none found
        """
        # Convert to string and handle None/NaN values
        valid_naics = naics_series.astype(str).replace({'nan': '', 'None': ''})
        
        # Filter to only valid NAICS codes with sufficient length
        valid_naics = valid_naics[valid_naics.str.len() >= length]
        
        if valid_naics.empty:
            return default
            
        # Extract prefix
        prefixes = valid_naics.str[:length]
        
        # Get mode
        mode_result = prefixes.mode()
        if len(mode_result) > 0:
            return mode_result.iloc[0]
        else:
            return default
    
    def _calculate_infrastructure_features(self, businesses: pd.DataFrame) -> List[np.ndarray]:
        """Calculate infrastructure-related features based on county location"""
        # Infrastructure density by county (objective data based on KC geography)
        rail_density = {
            "Jackson County, MO": 0.9,      # Major rail hub with multiple yards
            "Jackson County": 0.9,          # Handle both formats
            "Wyandotte County, KS": 0.8,    # Significant rail presence
            "Wyandotte County": 0.8,
            "Clay County, MO": 0.6,         # Some rail
            "Clay County": 0.6,
            "Johnson County, KS": 0.3,      # Limited rail
            "Johnson County": 0.3,
            "Platte County, MO": 0.4,       # Airport area
            "Platte County": 0.4,
            "Leavenworth County, KS": 0.5,
            "Leavenworth County": 0.5,
            "Miami County, KS": 0.2,
            "Miami County": 0.2,
            "Cass County, MO": 0.4,
            "Cass County": 0.4,
            "Ray County, MO": 0.3,
            "Ray County": 0.3,
            "Douglas County, KS": 0.2,
            "Douglas County": 0.2
        }
        
        highway_access = {
            "Jackson County, MO": 0.9,      # I-70, I-35 intersection
            "Jackson County": 0.9,
            "Johnson County, KS": 0.8,      # Multiple highways
            "Johnson County": 0.8,
            "Wyandotte County, KS": 0.7,    # I-70, I-435
            "Wyandotte County": 0.7,
            "Clay County, MO": 0.7,         # I-35, I-435
            "Clay County": 0.7,
            "Platte County, MO": 0.8,       # I-29, I-435, airport
            "Platte County": 0.8,
            "Leavenworth County, KS": 0.5,
            "Leavenworth County": 0.5,
            "Miami County, KS": 0.4,
            "Miami County": 0.4,
            "Cass County, MO": 0.5,
            "Cass County": 0.5,
            "Ray County, MO": 0.4,
            "Ray County": 0.4,
            "Douglas County, KS": 0.5,
            "Douglas County": 0.5
        }
        
        # Industrial/warehouse concentration (based on zoning and development)
        industrial_concentration = {
            "Jackson County, MO": 0.7,
            "Jackson County": 0.7,
            "Wyandotte County, KS": 0.8,    # Fairfax industrial district
            "Wyandotte County": 0.8,
            "Clay County, MO": 0.5,
            "Clay County": 0.5,
            "Johnson County, KS": 0.3,      # More office/retail
            "Johnson County": 0.3,
            "Platte County, MO": 0.6,       # Growing industrial
            "Platte County": 0.6,
            "Leavenworth County, KS": 0.4,
            "Leavenworth County": 0.4,
            "Miami County, KS": 0.3,
            "Miami County": 0.3,
            "Cass County, MO": 0.4,
            "Cass County": 0.4,
            "Ray County, MO": 0.3,
            "Ray County": 0.3,
            "Douglas County, KS": 0.2,
            "Douglas County": 0.2
        }
        
        # Extract features for each business
        rail_features = []
        highway_features = []
        industrial_features = []
        
        for _, business in businesses.iterrows():
            county = business.get('county', '')
            # Get infrastructure scores with default fallback
            rail_features.append(rail_density.get(county, 0.5))
            highway_features.append(highway_access.get(county, 0.5))
            industrial_features.append(industrial_concentration.get(county, 0.4))
        
        return [
            np.array(rail_features),
            np.array(highway_features),
            np.array(industrial_features)
        ]
    
    def _assess_economic_viability(self, businesses: pd.DataFrame, 
                                   labels: np.ndarray, 
                                   k: int,
                                   targets: Optional[Dict]) -> float:
        """Assess if cluster configuration can meet economic targets"""
        total_gdp_potential = 0
        total_job_potential = 0
        cluster_viabilities = []
        
        for cluster_id in range(k):
            cluster_mask = labels == cluster_id
            cluster_businesses = businesses[cluster_mask]
            
            if len(cluster_businesses) < 15:  # Too small
                cluster_viabilities.append(0)
                continue
            
            # Economic calculations
            cluster_revenue = cluster_businesses['revenue_estimate'].sum()
            cluster_employees = cluster_businesses['employees'].sum()
            
            # Estimate multiplier based on diversity and quality
            naics_diversity = len(cluster_businesses['naics_code'].str[:3].unique())
            avg_score = cluster_businesses['composite_score'].mean()
            
            # Use same GDP calculation as evaluate_cluster for consistency
            # Get dominant NAICS for multiplier
            naics_counts = cluster_businesses['naics_code'].str[:3].value_counts()
            dominant_naics = naics_counts.index[0] if len(naics_counts) > 0 else "000"
            
            # Industry-specific GDP multipliers (from _calculate_economic_output)
            naics_gdp_multipliers = {
                "511": 2.30, "518": 1.86, "336": 2.14, "3254": 2.48,
                "325": 2.20, "493": 1.82, "484": 1.74, "541": 1.68
            }
            base_multiplier = naics_gdp_multipliers.get(dominant_naics, 1.75)
            
            # Calculate raw GDP impact
            raw_gdp_impact = cluster_revenue * base_multiplier
            
            # Apply success probability (simplified version)
            success_prob = 0.35 + (avg_score / 100) * 0.3 + (min(naics_diversity, 5) / 5) * 0.15
            success_prob = min(0.80, max(0.35, success_prob))
            
            gdp_impact = raw_gdp_impact * success_prob
            job_impact = cluster_employees * base_multiplier * success_prob
            
            total_gdp_potential += gdp_impact
            total_job_potential += job_impact
            
            # Cluster viability score
            viability = (
                min(len(cluster_businesses) / 50, 1.0) * 0.3 +  # Size factor
                min(naics_diversity / 5, 1.0) * 0.3 +          # Diversity factor
                min(avg_score / 80, 1.0) * 0.4                 # Quality factor
            )
            cluster_viabilities.append(viability)
        
        # Overall economic viability
        base_viability = np.mean(cluster_viabilities) if cluster_viabilities else 0
        
        # Bonus for meeting targets - but penalize if far from targets
        if targets:
            gdp_target = targets.get('gdp_target', 2.87e9)
            job_target = targets.get('job_target', 3000)
            
            gdp_target_ratio = total_gdp_potential / gdp_target
            job_target_ratio = total_job_potential / job_target
            
            # Log for debugging
            logger.debug(f"k={k}: GDP potential=${total_gdp_potential/1e9:.2f}B (target=${gdp_target/1e9:.2f}B), ratio={gdp_target_ratio:.2f}")
            logger.debug(f"k={k}: Job potential={total_job_potential:.0f} (target={job_target}), ratio={job_target_ratio:.2f}")
            
            # If we're below 80% of target, heavily penalize
            if gdp_target_ratio < 0.8:
                gdp_penalty = (0.8 - gdp_target_ratio) * 0.5  # Up to 0.4 penalty
                base_viability = max(0, base_viability - gdp_penalty)
            
            # Bonus only if we're close to or exceeding targets
            if gdp_target_ratio >= 0.8 and job_target_ratio >= 0.8:
                target_bonus = min((gdp_target_ratio + job_target_ratio) / 2 * 0.2, 0.2)
                base_viability = min(base_viability + target_bonus, 1.0)
        
        return base_viability
    
    def _calculate_combined_score(self, metrics: Dict) -> float:
        """Calculate weighted combined score from multiple metrics"""
        # Normalize metrics to 0-1 scale
        silhouette_norm = (metrics['silhouette'] + 1) / 2  # Convert from [-1,1] to [0,1]
        
        # Davies-Bouldin: lower is better, so invert
        db_norm = 1 / (1 + metrics['davies_bouldin']) if metrics['davies_bouldin'] != float('inf') else 0
        
        # Calinski-Harabasz: normalize by typical range (0-1000)
        ch_norm = min(metrics['calinski_harabasz'] / 1000, 1.0)
        
        # Economic viability is already 0-1
        econ_norm = metrics['economic_viability']
        
        # Weighted combination
        combined = (
            silhouette_norm * 0.3 +      # Cluster separation
            db_norm * 0.2 +              # Cluster compactness
            ch_norm * 0.2 +              # Cluster density
            econ_norm * 0.3              # Economic potential
        )
        
        return combined
    
    def _select_optimal_k(self, results: List[Dict], targets: Optional[Dict]) -> Dict:
        """Select optimal k based on validation results"""
        # First preference: Solutions that meet economic targets
        if targets:
            meeting_targets = [r for r in results if r['metrics']['economic_viability'] > 0.8]
            if meeting_targets:
                return max(meeting_targets, key=lambda x: x['metrics']['combined_score'])
        
        # Otherwise: Best combined score with preference for diversity
        # Industry best practice: More clusters = more resilience and opportunities
        # Only penalize if clusters become too small (<20 businesses each)
        def score_with_diversity(result):
            base_score = result['metrics']['combined_score']
            k = result['k']
            
            # Bonus for reasonable cluster count (5-10 is ideal)
            if 5 <= k <= 10:
                diversity_bonus = 0.05 * (k - 2) / 8  # Up to 5% bonus
            else:
                diversity_bonus = 0
            
            # Penalty only if clusters too small
            avg_cluster_size = len(results[0].get('businesses', [])) / k if k > 0 else 0
            size_penalty = 0.1 if avg_cluster_size < 20 else 0
            
            return base_score + diversity_bonus - size_penalty
        
        best_result = max(results, key=score_with_diversity)
        return best_result
    
    def _generate_cluster_reasoning(self, results: List[Dict], optimal: Dict) -> str:
        """Generate human-readable reasoning for the cluster choice"""
        reasons = []
        
        reasons.append(f"Optimal number of clusters determined: {optimal['k']}")
        reasons.append("\nValidation metrics:")
        reasons.append(f"- Silhouette Score: {optimal['metrics']['silhouette']:.3f} "
                      f"(cluster separation quality)")
        reasons.append(f"- Economic Viability: {optimal['metrics']['economic_viability']:.3f} "
                      f"(potential to meet targets)")
        
        # Compare with alternatives
        silhouette_scores = [r['metrics']['silhouette'] for r in results]
        if optimal['metrics']['silhouette'] == max(silhouette_scores):
            reasons.append("- This configuration has the best cluster separation")
        
        # Economic reasoning
        if optimal['metrics']['economic_viability'] > 0.8:
            reasons.append("- This configuration can meet your economic targets")
        
        # Trade-off explanation
        if optimal['k'] < len(results):
            next_k = [r for r in results if r['k'] == optimal['k'] + 1]
            if next_k and next_k[0]['metrics']['combined_score'] < optimal['metrics']['combined_score']:
                reasons.append(f"- Adding more clusters would reduce overall quality")
        
        return "\n".join(reasons)
        
    def evaluate_cluster(self, individual: List[int], businesses: pd.DataFrame, all_businesses: pd.DataFrame = None) -> Tuple:
        """
        Evaluate a cluster configuration on multiple objectives:
        1. Economic output (GDP impact)
        2. Innovation potential
        3. Job creation
        4. Resource efficiency (synergies)
        5. Risk (minimize)
        6. Critical Mass (ecosystem completeness)
        """
        # Extract businesses in this cluster configuration
        cluster_businesses = businesses.iloc[individual]
        
        if len(cluster_businesses) == 0:
            return 0, 0, 0, 0, 100, 0  # Worst case
        
        # 1. Economic output estimation using standard input-output methodology
        total_revenue = cluster_businesses["revenue_estimate"].sum()
        output_multiplier = self._get_industry_multiplier(cluster_businesses)
        
        # Apply GDP/output multipliers based on BEA RIMS II data (industry standards)
        # These represent total economic impact per dollar of direct output
        naics_gdp_multipliers = {
            # Transportation & Logistics (strong local supply chains)
            "484": 1.82,   # Trucking - fuel, maintenance, insurance
            "493": 1.95,   # Warehousing - utilities, equipment, services
            "488": 1.89,   # Support activities for transportation
            
            # Manufacturing (high multiplier due to supply chains)
            "332": 2.16,   # Fabricated metal - raw materials, machinery
            "333": 2.23,   # Machinery manufacturing - parts, assembly
            "334": 1.95,   # Computer/electronic manufacturing
            "335": 2.08,   # Electrical equipment manufacturing
            "336": 2.34,   # Transportation equipment - highest multiplier
            
            # Life Sciences (high value-add)
            "3254": 2.45,  # Pharmaceutical - R&D, clinical trials
            "3391": 2.12,  # Medical equipment manufacturing
            "5417": 1.78,  # Scientific R&D services
            
            # Technology (knowledge spillovers)
            "5112": 1.92,  # Software publishers
            "5415": 2.05,  # Computer systems design
            "5182": 1.86,  # Data processing
            
            # Professional Services
            "5416": 1.71,  # Management consulting
            "5413": 1.68,  # Architectural/engineering
            "5411": 1.64,  # Legal services
            
            # Default multiplier for other industries
            "default": 1.75
        }
        
        # Calculate weighted GDP multiplier
        total_gdp_impact = 0
        for naics, group in cluster_businesses.groupby(cluster_businesses["naics_code"].str[:4]):
            naics_3 = naics[:3]
            # Try 4-digit first, then 3-digit, then default
            multiplier = naics_gdp_multipliers.get(naics, 
                        naics_gdp_multipliers.get(naics_3, 
                        naics_gdp_multipliers.get("default", 1.75)))
            
            # Direct output (revenue) Ã— multiplier = total economic impact
            group_revenue = group["revenue_estimate"].sum()
            group_gdp_impact = group_revenue * multiplier
            total_gdp_impact += group_gdp_impact
        
        # GDP impact calculation complete
        gdp_impact = total_gdp_impact
        
        # Growth potential with cluster optimization
        # Dynamic growth rate based on cluster quality and industry
        base_growth_rate = 0.05  # 5% baseline
        
        # Adjust growth rate based on industry potential
        naics_3 = self._get_safe_naics_prefix(cluster_businesses["naics_code"]) if len(cluster_businesses) > 0 else "000"
        growth_adjustments = {
            "511": 1.5,   # Software - high growth
            "518": 1.4,   # Data processing - high growth
            "3254": 1.3,  # Pharmaceutical - good growth
            "5415": 1.4,  # Computer systems - high growth
            "493": 1.2,   # Warehousing - e-commerce driven
        }
        industry_factor = growth_adjustments.get(naics_3, 1.0)
        
        # Adjust for innovation intensity
        avg_innovation = cluster_businesses["innovation_score"].mean() if "innovation_score" in cluster_businesses else 50
        innovation_factor = 1.0 + (avg_innovation - 50) / 100  # +/- 50% based on innovation
        
        annual_growth_rate = base_growth_rate * industry_factor * innovation_factor
        annual_growth_rate = min(0.15, max(0.03, annual_growth_rate))  # Keep between 3-15%
        
        time_horizon = self.params.get('time_horizon', self.config.TIME_HORIZON_YEARS)  # years
        
        # Calculate GDP impact using compound growth for realism
        base_gdp = total_gdp_impact  # Use the calculated GDP impact as base
        
        # Apply a cluster synergy factor (10-30% boost from clustering)
        # This represents the additional value from cluster effects
        synergy_factor = 0.10 + (avg_innovation / 100) * 0.20  # 10-30% based on innovation

        if self._runtime_flag("disable_network_metrics"):
            synergy_factor *= 0.5  # Reduced uplift when network effects are disabled

        # Calculate future value with compound growth AND synergy
        # This avoids double counting by applying synergy to the growth calculation
        future_gdp_with_synergy = base_gdp * math.pow(1 + annual_growth_rate, time_horizon) * (1 + synergy_factor)
        
        # Total GDP impact (theoretical maximum)
        gdp_impact = future_gdp_with_synergy
        
        # Calculate success probability based on objective factors
        success_probability = self._calculate_success_probability(cluster_businesses)
        
        # Apply success probability to get realistic GDP
        realistic_gdp_impact = gdp_impact * success_probability
        
        # Log the adjustment
        revenue_multiplier = gdp_impact / total_revenue if total_revenue > 0 else 0
        if revenue_multiplier > 4.0 or gdp_impact > 1_000_000_000:
            logger.info(f"GDP projection with success probability:")
            logger.info(f"  Theoretical: ${gdp_impact:,.0f} ({revenue_multiplier:.1f}x revenue)")
            logger.info(f"  Success probability: {success_probability:.1%}")
            logger.info(f"  Realistic projection: ${realistic_gdp_impact:,.0f}")
        
        # Use realistic GDP for optimization
        gdp_impact = realistic_gdp_impact
        
        # 2. Innovation potential
        total_patents = cluster_businesses["patent_count"].sum()
        total_sbir = cluster_businesses["sbir_awards"].sum()
        innovation_score = total_patents * 10 + total_sbir * 20
        
        # 3. Job creation using employment multipliers
        direct_jobs = cluster_businesses["employees"].sum()
        
        # Employment multipliers by industry (BLS RIMS II)
        employment_multipliers = {
            "484": 1.8,   # Trucking
            "493": 1.7,   # Warehousing  
            "3254": 2.1,  # Pharmaceutical
            "5415": 2.3,  # Computer systems
            "332": 1.6,   # Fabricated metal
            "3253": 1.9,  # Agricultural chemical
        }
        
        # Calculate weighted employment multiplier
        total_emp_mult = 0
        total_emp_weight = 0
        for naics, group in cluster_businesses.groupby(cluster_businesses["naics_code"].str[:4]):
            naics_3 = naics[:3]
            emp_mult = employment_multipliers.get(naics_3, 1.7)
            weight = len(group)
            total_emp_mult += emp_mult * weight
            total_emp_weight += weight
            
        avg_emp_multiplier = total_emp_mult / max(total_emp_weight, 1)
        
        # Calculate incremental job creation, not total employment
        # Assume job growth proportional to revenue growth
        annual_job_growth_rate = 0.10  # 10% annual job growth with support
        time_horizon = self.params.get('time_horizon', self.config.TIME_HORIZON_YEARS)  # years
        
        # Job creation calculation based on economic growth
        # Jobs should grow proportionally to GDP growth
        
        # Use the same growth rate as GDP for consistency
        job_growth_rate = annual_growth_rate * 0.8  # Jobs grow slightly slower than GDP
        
        # Calculate job growth over time horizon
        future_direct_jobs = direct_jobs * math.pow(1 + job_growth_rate, time_horizon)
        direct_job_growth = future_direct_jobs - direct_jobs
        
        # Apply employment multiplier to NEW jobs only
        indirect_job_growth = direct_job_growth * (avg_emp_multiplier - 1)
        
        # Total employment projection (existing + new jobs)
        total_jobs = int(direct_jobs + direct_job_growth + indirect_job_growth)
        
        # 4. Resource efficiency (synergies)
        synergy_score = self._calculate_synergies(cluster_businesses)
        
        # 5. Risk assessment
        risk_score = self._calculate_risk(cluster_businesses)
        
        # 6. Critical Mass - rewards ecosystem completeness
        # Use all_businesses if provided, otherwise use businesses
        critical_mass_score = self._calculate_critical_mass(cluster_businesses, all_businesses if all_businesses is not None else businesses)
        
        return gdp_impact, innovation_score, total_jobs, synergy_score, risk_score, critical_mass_score
    
    def _get_industry_multiplier(self, businesses: pd.DataFrame) -> float:
        """Get economic multiplier based on industry mix (BLS RIMS II based)"""
        # More conservative multipliers based on actual BLS data
        multipliers = {
            "484": 1.8,   # Trucking (reduced from 2.8)
            "493": 1.7,   # Warehousing (reduced from 2.5)
            "3254": 1.9,  # Pharmaceutical (reduced from 2.2)
            "5415": 1.6,  # Computer systems (reduced from 2.0)
            "332": 1.7,   # Fabricated metal (reduced from 2.3)
            "3253": 1.8,  # Agricultural chemical (reduced from 2.1)
        }
        
        # Weighted average based on business count
        total_multiplier = 0
        total_weight = 0
        
        for naics, group in businesses.groupby(businesses["naics_code"].str[:3]):
            multiplier = multipliers.get(naics, 2.0)
            weight = len(group)
            total_multiplier += multiplier * weight
            total_weight += weight
        
        return total_multiplier / total_weight if total_weight > 0 else 2.0
    
    def _calculate_synergies(self, businesses: pd.DataFrame) -> float:
        """Calculate synergy score based on supply chain linkages"""
        if self._runtime_flag("disable_network_metrics"):
            # Without network insights, only acknowledge basic co-location effects
            baseline = min(10.0, len(businesses) * 0.2)
            return float(baseline)

        synergy_score = 0
        
        # Industry complementarity matrix
        synergies = {
            ("484", "493"): 30,   # Trucking + Warehousing
            ("484", "332"): 20,   # Trucking + Manufacturing
            ("3254", "5417"): 25, # Pharma + R&D services
            ("5415", "5416"): 20, # Computer systems + Data processing
        }
        
        # Check for synergistic pairs
        naics_codes = businesses["naics_code"].str[:3].unique()
        for i, naics1 in enumerate(naics_codes):
            for naics2 in naics_codes[i+1:]:
                pair = tuple(sorted([naics1, naics2]))
                if pair in synergies:
                    synergy_score += synergies[pair]
        
        # Geographic clustering bonus
        counties = businesses["county"].value_counts()
        if len(counties) <= 3:  # Concentrated in few counties
            synergy_score += 20
        
        # Size diversity bonus (mix of large and small businesses)
        emp_std = businesses["employees"].std()
        if emp_std > 50:
            synergy_score += 15
        
        return synergy_score
    
    def _calculate_risk(self, businesses: pd.DataFrame) -> float:
        """Calculate cluster risk score"""
        risk_score = 0
        
        # Industry concentration risk
        industry_counts = businesses["naics_code"].str[:3].value_counts()
        if len(industry_counts) == 1:  # Single industry
            risk_score += 30
        elif industry_counts.iloc[0] / len(businesses) > 0.7:  # Dominated by one industry
            risk_score += 20
        
        # Age distribution risk (too many new businesses)
        from datetime import datetime as _dt
        avg_age = _dt.now().year - businesses["year_established"].mean()
        if avg_age < 3:
            risk_score += 20
        
        # Size risk (too many small businesses)
        small_businesses = len(businesses[businesses["employees"] < 20])
        if small_businesses / len(businesses) > 0.7:
            risk_score += 15
        
        # Market volatility risk
        volatile_industries = ["484", "332"]  # Transportation, manufacturing
        volatile_count = businesses["naics_code"].str[:3].isin(volatile_industries).sum()
        risk_score += (volatile_count / len(businesses)) * 20
        
        return risk_score
    
    def _extract_cluster_data(self, cluster_businesses: pd.DataFrame, all_businesses: pd.DataFrame) -> Dict:
        """Extract cluster data with full metrics calculation"""
        # Determine cluster type dynamically based on business composition
        cluster_type = self._determine_cluster_type(cluster_businesses)
        
        # Generate descriptive cluster name
        cluster_name = self._generate_cluster_name(cluster_businesses, cluster_type, 1)
        
        # Create individual representation (list of indices) for fitness evaluation
        individual = list(cluster_businesses.index)
        
        # Use the existing evaluate_cluster method to get fitness values
        fitness_values = self.evaluate_cluster(individual, all_businesses, all_businesses)
        
        # Unpack fitness values
        gdp_impact, innovation_score, job_creation, synergy_score, risk_score, critical_mass = fitness_values
        
        # Create cluster data structure
        cluster_data = {
            "name": cluster_name,
            "type": cluster_type,
            "businesses": cluster_businesses.to_dict("records"),
            "business_count": len(cluster_businesses),
            "total_employees": cluster_businesses["employees"].sum(),
            "total_revenue": cluster_businesses["revenue_estimate"].sum(),
            "fitness_values": fitness_values,
            "projected_gdp_impact": gdp_impact,
            "innovation_score": innovation_score,
            "projected_jobs": job_creation,
            "synergy_score": synergy_score,
            "risk_score": risk_score,
            "critical_mass": critical_mass,
            # Unified metrics dictionary for downstream ML enhancer
            "metrics": {
                "total_employees": cluster_businesses["employees"].sum(),
                "total_revenue": cluster_businesses["revenue_estimate"].sum(),
                "innovation_score": innovation_score,
                "avg_business_age": ((_dt.now().year - cluster_businesses["year_established"]).mean()),
                "projected_gdp_impact": gdp_impact,
                "projected_jobs": job_creation,
                "risk_score": risk_score,
                "critical_mass": critical_mass
            }
        }
        
        # Calculate comprehensive cluster scores
        scores = self._calculate_cluster_scores(cluster_businesses, cluster_data)
        cluster_data.update(scores)
        # Provide strategic_score alias expected by ML enhancer
        cluster_data["strategic_score"] = scores.get("total_score", 0)
        
        # Add network metrics for visualization unless disabled
        if not self._runtime_flag("disable_network_metrics"):
            network_metrics = self._calculate_network_metrics(cluster_businesses)
            cluster_data["network_metrics"] = network_metrics

        return cluster_data
    
    def _calculate_critical_mass(self, cluster_businesses: pd.DataFrame, all_businesses: pd.DataFrame) -> float:
        """
        Calculate critical mass score - rewards ecosystem completeness
        
        Critical mass considers:
        1. Size relative to natural community
        2. Supply chain completeness
        3. Workforce diversity
        4. Geographic density
        
        Returns score 0-100 where 100 is a complete ecosystem
        """
        score = 0.0
        
        # 1. Size factor - what percentage of available businesses are included?
        # If this is a natural community, use community size; otherwise use all businesses
        if 'community_type' in cluster_businesses.columns:
            # This cluster is from a natural community
            community_type = cluster_businesses['community_type'].iloc[0]
            community_businesses = all_businesses[all_businesses.get('community_type', '') == community_type]
            size_ratio = len(cluster_businesses) / max(len(community_businesses), 1)
        else:
            # Generic cluster - compare to businesses in same counties/industries
            cluster_counties = cluster_businesses['county'].unique()
            cluster_industries = cluster_businesses['naics_code'].str[:3].unique()
            
            potential_businesses = all_businesses[
                (all_businesses['county'].isin(cluster_counties)) &
                (all_businesses['naics_code'].str[:3].isin(cluster_industries))
            ]
            size_ratio = len(cluster_businesses) / max(len(potential_businesses), 1)
        
        # Size contributes 30 points (0-30)
        score += min(size_ratio * 30, 30)
        
        # 2. Supply chain completeness - do we have full value chain?
        supply_chains = {
            'manufacturing': ['332', '333', '336'],  # Metal â†’ Machinery â†’ Assembly
            'logistics': ['484', '488', '493'],      # Trucking â†’ Support â†’ Warehousing
            'tech': ['511', '518', '541'],           # Software â†’ Data â†’ Services
            'life_sciences': ['325', '3254', '5417'], # Chemical â†’ Pharma â†’ R&D
        }
        
        industries = set(cluster_businesses['naics_code'].str[:3])
        completeness_score = 0
        
        for chain_name, chain_codes in supply_chains.items():
            chain_present = sum(1 for code in chain_codes if code in industries)
            if chain_present >= 2:  # At least 2 parts of chain
                completeness_score += (chain_present / len(chain_codes)) * 10
        
        score += min(completeness_score, 25)  # Max 25 points
        
        # 3. Business size diversity - mix of small/medium/large
        employees = cluster_businesses['employees']
        small = len(employees[employees < 50])
        medium = len(employees[(employees >= 50) & (employees < 500)])
        large = len(employees[employees >= 500])
        
        total = len(employees)
        if total > 0:
            # Ideal mix: 60% small, 30% medium, 10% large (matches US economy)
            size_mix_score = 0
            size_mix_score += min(small/total, 0.6) / 0.6 * 10  # Up to 10 points
            size_mix_score += min(medium/total, 0.3) / 0.3 * 10  # Up to 10 points  
            size_mix_score += min(large/total, 0.1) / 0.1 * 5   # Up to 5 points
            score += size_mix_score
        
        # 4. Geographic density - concentration is good for clusters
        counties = cluster_businesses['county'].nunique()
        if counties <= 2:  # Highly concentrated
            score += 20
        elif counties <= 4:  # Moderately concentrated
            score += 15
        elif counties <= 6:  # Somewhat dispersed
            score += 10
        else:  # Too dispersed
            score += 5
        
        return min(score, 100)  # Cap at 100
    
    def create_individual(self, businesses: pd.DataFrame, cluster_size: Tuple[int, int]):
        """Create a random individual (cluster configuration) with diverse businesses"""
        min_size, max_size = cluster_size
        # Ensure we don't try to sample more than available businesses
        available_businesses = len(businesses)
        actual_max_size = min(max_size, available_businesses)
        actual_min_size = min(min_size, actual_max_size)
        
        if actual_min_size <= 0:
            logger.warning(f"Not enough businesses ({available_businesses}) for minimum cluster size ({min_size})")
            return creator.Individual([])
            
        # Reset index to ensure we use position-based indices
        businesses_reset = businesses.reset_index(drop=True)
        
        # Create diverse cluster by sampling from different NAICS categories
        naics_groups = businesses_reset.groupby(businesses_reset['naics_code'].str[:3])
        selected_positions = []
        
        # First, ensure we get at least one business from each major NAICS group
        for naics, group in naics_groups:
            if len(selected_positions) < actual_max_size and len(group) > 0:
                # Sample 1-3 businesses from each group
                sample_size = min(3, len(group), actual_max_size - len(selected_positions))
                # Get positional indices within the reset dataframe
                group_positions = group.index.tolist()
                sampled_positions = random.sample(group_positions, sample_size)
                selected_positions.extend(sampled_positions)
        
        # If we need more businesses, randomly sample from the remaining
        if len(selected_positions) < actual_min_size:
            all_positions = list(range(available_businesses))
            remaining_positions = list(set(all_positions) - set(selected_positions))
            if remaining_positions:
                additional_needed = random.randint(actual_min_size - len(selected_positions), 
                                                 actual_max_size - len(selected_positions))
                additional = random.sample(remaining_positions, 
                                         min(additional_needed, len(remaining_positions)))
                selected_positions.extend(additional)
        
        # Ensure we don't exceed max size
        if len(selected_positions) > actual_max_size:
            selected_positions = random.sample(selected_positions, actual_max_size)
        
        return creator.Individual(selected_positions)
    
    def _custom_crossover(self, ind1: List[int], ind2: List[int], businesses: pd.DataFrame, cluster_size: Tuple[int, int]) -> Tuple[List[int], List[int]]:
        """Custom crossover that maintains uniqueness within each individual"""
        # Create copies
        child1 = ind1[:]
        child2 = ind2[:]
        
        # Perform standard two-point crossover
        size = min(len(child1), len(child2))
        if size >= 2:
            cxpoint1 = random.randint(1, size)
            cxpoint2 = random.randint(1, size - 1)
            if cxpoint2 >= cxpoint1:
                cxpoint2 += 1
            else:
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1
            
            # Swap the segments
            child1[cxpoint1:cxpoint2], child2[cxpoint1:cxpoint2] = \
                child2[cxpoint1:cxpoint2], child1[cxpoint1:cxpoint2]
        
        # Remove duplicates while preserving order
        seen1 = set()
        child1[:] = [x for x in child1 if not (x in seen1 or seen1.add(x))]
        
        seen2 = set()
        child2[:] = [x for x in child2 if not (x in seen2 or seen2.add(x))]
        
        # Validate and repair children to meet size constraints
        available_positions = set(range(len(businesses)))
        
        min_size, max_size = cluster_size
        
        # Repair child1 if too small
        if len(child1) < min_size:
            available_for_child1 = available_positions - set(child1)
            needed = min_size - len(child1)
            if available_for_child1 and needed > 0:
                additional = random.sample(list(available_for_child1), 
                                         min(needed, len(available_for_child1)))
                child1.extend(additional)
        
        # Repair child2 if too small
        if len(child2) < min_size:
            available_for_child2 = available_positions - set(child2)
            needed = min_size - len(child2)
            if available_for_child2 and needed > 0:
                additional = random.sample(list(available_for_child2), 
                                         min(needed, len(available_for_child2)))
                child2.extend(additional)
        
        # Trim if too large
        if len(child1) > max_size:
            child1[:] = random.sample(child1, max_size)
        
        if len(child2) > max_size:
            child2[:] = random.sample(child2, max_size)
        
        return child1, child2
    
    def _calculate_success_probability(self, cluster_businesses: pd.DataFrame) -> float:
        """Calculate realistic success probability based on objective cluster characteristics"""
        success_factors = []
        
        # 1. Business quality distribution
        if 'composite_score' in cluster_businesses.columns:
            score_distribution = cluster_businesses['composite_score'].describe()
            if score_distribution['75%'] > 80:  # 75% of businesses are high quality
                success_factors.append(0.9)
            elif score_distribution['50%'] > 70:  # Median is good
                success_factors.append(0.7)
            elif score_distribution['50%'] > 60:  # Median is decent
                success_factors.append(0.5)
            else:
                success_factors.append(0.3)
        else:
            success_factors.append(0.5)  # Default if no scores
        
        # 2. Size viability (too small or too large clusters are riskier)
        cluster_size = len(cluster_businesses)
        if 30 <= cluster_size <= 150:
            success_factors.append(0.85)
        elif 20 <= cluster_size <= 200:
            success_factors.append(0.7)
        elif 10 <= cluster_size <= 300:
            success_factors.append(0.5)
        else:
            success_factors.append(0.3)
        
        # 3. Industry concentration (some diversity is good, but not too much)
        naics_diversity = len(cluster_businesses['naics_code'].str[:3].unique())
        total_businesses = len(cluster_businesses)
        
        if total_businesses > 0:
            concentration_ratio = naics_diversity / total_businesses
            if 0.05 <= concentration_ratio <= 0.3:  # Focused but not monolithic
                success_factors.append(0.8)
            elif 0.03 <= concentration_ratio <= 0.5:
                success_factors.append(0.6)
            else:
                success_factors.append(0.4)
        else:
            success_factors.append(0.5)
        
        # 4. Business maturity (mix of established and growing)
        if 'year_established' in cluster_businesses.columns:
            from datetime import datetime as _dt
            current_year = _dt.now().year
            ages = current_year - cluster_businesses['year_established']
            avg_age = ages.mean()
            
            if 5 <= avg_age <= 15:  # Sweet spot - established but not stagnant
                success_factors.append(0.8)
            elif 3 <= avg_age <= 20:
                success_factors.append(0.6)
            else:
                success_factors.append(0.4)
        else:
            success_factors.append(0.6)
        
        # 5. Revenue distribution (avoid clusters with only tiny or huge businesses)
        if 'revenue_estimate' in cluster_businesses.columns:
            mean_revenue = cluster_businesses['revenue_estimate'].mean()
            std_revenue = cluster_businesses['revenue_estimate'].std()
            
            # Handle edge cases where mean is 0 or NaN
            if pd.isna(mean_revenue) or pd.isna(std_revenue) or mean_revenue == 0:
                # If we can't calculate CV, assume moderate success
                success_factors.append(0.5)
            else:
                revenue_cv = std_revenue / mean_revenue
                if not pd.isna(revenue_cv) and revenue_cv < 2.0:  # Reasonable variation
                    success_factors.append(0.7)
                else:
                    success_factors.append(0.5)
        else:
            success_factors.append(0.6)
        
        # Combined probability using weighted average (industry best practice)
        # Weights based on factor importance from economic development research
        weights = [0.30, 0.20, 0.20, 0.15, 0.15]  # Quality, Size, Concentration, Age, Revenue
        
        # Validate we have the expected number of factors
        if len(success_factors) != len(weights):
            logger.error(f"Success factor count mismatch: {len(success_factors)} factors vs {len(weights)} weights")
            logger.error(f"Success factors collected: {success_factors}")
            # Still calculate but log the issue
            if len(success_factors) > 0:
                # Use equal weights if mismatch
                equal_weights = [1.0 / len(success_factors)] * len(success_factors)
                success_probability = np.average(success_factors, weights=equal_weights)
                logger.warning(f"Using equal weights for {len(success_factors)} factors")
            else:
                success_probability = 0.5  # Default
                logger.warning("No success factors available, using default 0.5")
        else:
            success_probability = np.average(success_factors, weights=weights)
        
        # Industry standard bounds: 35% floor (struggling), 80% ceiling (excellent)
        # Based on McKinsey cluster success studies
        success_probability = max(0.35, min(0.80, success_probability))
        
        return success_probability
    
    def optimize_clusters_with_communities(self, businesses: pd.DataFrame,
                                          num_clusters: Optional[int] = None,
                                          economic_targets: Optional[Dict] = None,
                                          optimization_focus: str = 'balanced',
                                          progress_callback: Optional[callable] = None) -> List[Dict]:
        """
        Optimize clusters using natural community detection first, then NSGA-II
        
        This hybrid approach:
        1. Finds natural business communities
        2. Uses NSGA-II to decide which communities to activate
        3. Optimizes connections between communities
        
        Returns:
            List of optimized cluster configurations
        """
        logger.info("Starting community-based cluster optimization...")
        
        # Initialize params for evaluation
        self.params = {}
        self.optimization_focus = optimization_focus
        
        # Step 1: Discover communities (with caching)
        if _HAS_NETWORKX:
            try:
                from utils.cache import cache
                sig = {
                    'n': int(len(businesses)),
                    'naics3': businesses['naics_code'].astype(str).str[:3].value_counts().to_dict(),
                    'county': businesses['county'].value_counts().to_dict(),
                }
                import json
                cache_key = 'louvain_communities_' + str(abs(hash(json.dumps(sig, sort_keys=True))))
                communities = cache.get(cache_key)
                if communities is None:
                    communities = self.find_louvain_communities(businesses)
                    cache.set(cache_key, communities, timeout=900)
                logger.info(f"Found {len(communities)} Louvain communities")
            except Exception as e:
                logger.warning(f"Louvain community detection failed ({e}); falling back to heuristic communities")
                communities = self.find_natural_communities(businesses)
                logger.info(f"Found {len(communities)} natural communities (fallback)")
        else:
            communities = self.find_natural_communities(businesses)
            logger.info(f"Found {len(communities)} natural communities (networkx unavailable)")
        
        # Store for reporting
        self.natural_communities_count = len(communities)
        
        # Extract and store detailed community information
        self.natural_communities_details = self.extract_community_details(communities)
        
        # Step 2: For each community, evaluate its potential
        community_scores = []
        for i, community in enumerate(communities):
            # Quick evaluation of community potential
            avg_score = community['composite_score'].mean()
            size = len(community)
            diversity = community['naics_code'].str[:3].nunique()
            
            potential = {
                'index': i,
                'community': community,
                'size': size,
                'avg_score': avg_score,
                'diversity': diversity,
                'potential_gdp': community['revenue_estimate'].sum() * 2.0  # Rough estimate
            }
            community_scores.append(potential)
        
        # Step 3: Use NSGA-II to select optimal community combinations
        # Instead of selecting individual businesses, we select entire communities
        selected_communities = self._optimize_community_selection_nsga(
            community_scores,
            num_clusters,
            economic_targets,
            optimization_focus,
            progress_callback
        )
        
        # Step 4: Convert selected communities back to cluster format
        clusters = []
        for community_indices in selected_communities:
            # Combine businesses from selected communities
            cluster_businesses = pd.concat([communities[i] for i in community_indices])
            
            # Calculate full metrics
            cluster_data = self._extract_cluster_data(cluster_businesses, businesses)
            clusters.append(cluster_data)
        
        return clusters
    
    def _optimize_community_selection(self, community_scores: List[Dict],
                                     num_clusters: Optional[int],
                                     economic_targets: Optional[Dict],
                                     optimization_focus: str,
                                     progress_callback: Optional[callable]) -> List[List[int]]:
        """Deprecated greedy selection (kept for compatibility)."""
        
        if num_clusters is None:
            # Automatically determine based on economic targets
            if economic_targets:
                gdp_target = economic_targets.get('gdp_target', 2.87e9)
                # Estimate how many communities needed
                total_gdp = sum(c['potential_gdp'] for c in community_scores)
                num_clusters = max(3, min(10, int(gdp_target / (total_gdp / len(community_scores)))))
            else:
                num_clusters = 5  # Default
        
        # Sort communities by potential
        sorted_communities = sorted(community_scores, 
                                  key=lambda x: x['potential_gdp'] * x['diversity'], 
                                  reverse=True)
        
        # Select top communities
        selected = []
        for i in range(min(num_clusters, len(sorted_communities))):
            selected.append([sorted_communities[i]['index']])
        
        return selected

    def _optimize_community_selection_nsga(self, community_scores: List[Dict],
                                           num_clusters: Optional[int],
                                           economic_targets: Optional[Dict],
                                           optimization_focus: str,
                                           progress_callback: Optional[callable]) -> List[List[int]]:
        """Select communities via NSGA-II optimizing total potential GDP, diversity, and minimizing overlap."""
        if not community_scores:
            return []

        # Decide number of communities to select
        if num_clusters is None:
            avg_gdp = np.mean([c['potential_gdp'] for c in community_scores]) if community_scores else 0
            if economic_targets and avg_gdp > 0:
                gdp_target = economic_targets.get('gdp_target', 2.87e9)
                k = int(max(3, min(10, round(gdp_target / max(avg_gdp, 1)))))
            else:
                k = min(5, len(community_scores))
        else:
            k = max(1, min(num_clusters, len(community_scores)))

        # Build business index sets for overlap penalty
        com_business_indices = []
        for c in community_scores:
            df = c['community']
            com_business_indices.append(set(df.index.tolist()))

        # Local DEAP types
        try:
            if hasattr(creator, 'FitnessComm'):
                del creator.FitnessComm
            if hasattr(creator, 'IndividualComm'):
                del creator.IndividualComm
        except Exception:
            pass
        creator.create('FitnessComm', base.Fitness, weights=(1.0, 0.5, -1.0))
        creator.create('IndividualComm', list, fitness=creator.FitnessComm)

        toolbox = base.Toolbox()
        indices = list(range(len(community_scores)))

        # Clamp k to available communities to avoid sampling errors
        k = max(1, min(int(k), len(indices)))

        def init_individual():
            return creator.IndividualComm(random.sample(indices, k))

        toolbox.register('individual', init_individual)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)

        def evaluate(ind):
            sel = list(dict.fromkeys(ind))
            total_gdp = sum(community_scores[i]['potential_gdp'] for i in sel)
            avg_div = np.mean([community_scores[i]['diversity'] for i in sel]) if sel else 0.0
            overlap = 0
            if len(sel) > 1:
                seen = set(); dup = set()
                for i in sel:
                    s = com_business_indices[i]
                    inter = s & seen
                    if inter:
                        dup |= inter
                    seen |= s
                overlap = len(dup)
            return (float(total_gdp), float(avg_div), float(overlap))

        def mate(ind1, ind2):
            # If chromosome length < 2, skip crossover
            if k < 2:
                return ind1, ind2
            cx1, cx2 = sorted(random.sample(range(k), 2))
            ind1[cx1:cx2], ind2[cx1:cx2] = ind2[cx1:cx2], ind1[cx1:cx2]
            def repair(ind):
                seen = set()
                for pos in range(len(ind)):
                    if ind[pos] in seen:
                        choices = list(set(indices) - set(ind))
                        if choices:
                            ind[pos] = random.choice(choices)
                    seen.add(ind[pos])
                return ind
            repair(ind1); repair(ind2)
            return ind1, ind2

        def mutate(ind):
            # Ensure mutation is valid for small k
            if k >= 2 and random.random() < 0.5:
                i, j = sorted(random.sample(range(k), 2))
                ind[i], ind[j] = ind[j], ind[i]
            else:
                pos = random.randrange(k)
                choices = list(set(indices) - set(ind))
                if choices:
                    ind[pos] = random.choice(choices)
            return (ind,)

        toolbox.register('evaluate', evaluate)
        toolbox.register('mate', mate)
        toolbox.register('mutate', mutate)
        toolbox.register('select', tools.selNSGA2)

        pop = toolbox.population(n=50)
        invalid = [ind for ind in pop if not ind.fitness.valid]
        fits = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit

        # Ensure crowding distances are set at least once before the first
        # selTournamentDCD call (required by DEAP's tournament operator)
        try:
            from deap.tools.emo import assignCrowdingDist
            fronts = tools.sortNondominated(pop, len(pop))
            for f in fronts:
                assignCrowdingDist(f)
        except Exception:
            pass

        ngen = 40
        for gen in range(1, ngen + 1):
            if progress_callback:
                progress_callback(int(gen / ngen * 100))

            # Ensure crowding distances exist on current population
            try:
                from deap.tools.emo import assignCrowdingDist
                fronts = tools.sortNondominated(pop, len(pop))
                for f in fronts:
                    assignCrowdingDist(f)
            except Exception:
                pass
            # Final guard: default any missing values
            try:
                for ind in pop:
                    if not hasattr(ind.fitness, 'crowding_dist'):
                        ind.fitness.crowding_dist = 0.0
            except Exception:
                pass

            # Debug: log status of crowding distances before selection
            try:
                missing = [ind for ind in pop if not hasattr(ind.fitness, 'crowding_dist')]
                if missing:
                    logger.warning("Crowding distance missing on %d individuals (gen=%d); set default 0.0", len(missing), gen)
                else:
                    logger.debug("Crowding distances OK (gen=%d, pop=%d)", gen, len(pop))
            except Exception:
                pass

            n = len(pop)
            k_sel = n - (n % 4)
            if k_sel >= 4:
                offspring = tools.selTournamentDCD(pop, k_sel)
            else:
                offspring = tools.selNSGA2(pop, n)
            offspring = [toolbox.clone(ind) for ind in offspring]
            for i in range(0, len(offspring), 2):
                if random.random() < 0.7 and i + 1 < len(offspring):
                    offspring[i], offspring[i+1] = toolbox.mate(offspring[i], offspring[i+1])
                    del offspring[i].fitness.values, offspring[i+1].fitness.values
            for i in range(len(offspring)):
                if random.random() < 0.2:
                    offspring[i], = toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fits = list(map(toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit
            pop = toolbox.select(pop + offspring, 50)

        front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        solutions = []
        assigned = set()
        for ind in front:
            sel = list(dict.fromkeys(ind))
            combined = set()
            for i in sel:
                combined |= com_business_indices[i]
            if not (combined & assigned):
                solutions.append(sel)
                assigned |= combined
            if len(solutions) >= k:
                break
        if not solutions:
            best = max(pop, key=lambda ind: ind.fitness.values[0])
            solutions = [list(dict.fromkeys(best))]
        return solutions
    
    def optimize_clusters(self, businesses: pd.DataFrame, 
                         num_clusters: Optional[int] = None,
                         cluster_size: Optional[Tuple[int, int]] = None,
                         economic_targets: Optional[Dict] = None,
                         optimization_focus: str = 'balanced',
                         progress_callback: Optional[callable] = None,
                         params: Optional[Dict] = None,
                         runtime_options: Optional[Dict] = None) -> List[Dict]:
        """
        Run optimization to find optimal cluster configurations
        
        This method now uses a hybrid approach:
        1. First finds natural communities based on geography, industry, and size
        2. Then uses NSGA-II to optimize within these communities with critical mass objective
        
        Args:
            businesses: DataFrame of businesses to cluster
            num_clusters: Number of clusters to create. If None, automatically determine optimal number
            cluster_size: Min/max size constraints for each cluster. If None, automatically determine optimal sizes
            economic_targets: Optional economic targets for automatic discovery
            optimization_focus: One of 'balanced', 'gdp', 'jobs', 'innovation'
            progress_callback: Optional callback for progress updates (current_step, total_steps)
        
        Returns:
            List of optimized cluster configurations
        """
        # Apply runtime flags (defaults restored each call)
        self.set_runtime_options(runtime_options)

        # Use the hybrid community-based approach unless NSGA-II is disabled
        if self._runtime_flag("disable_nsga2"):
            logger.info("Runtime flag disable_nsga2=True detected; running greedy classic optimizer")
            return self.optimize_clusters_classic(
                businesses=businesses,
                num_clusters=num_clusters,
                cluster_size=cluster_size,
                economic_targets=economic_targets,
                optimization_focus=optimization_focus,
                progress_callback=progress_callback,
                params=params,
                force_greedy=True
            )

        return self.optimize_clusters_with_communities(
            businesses=businesses,
            num_clusters=num_clusters,
            economic_targets=economic_targets,
            optimization_focus=optimization_focus,
            progress_callback=progress_callback
        )

    def _build_greedy_clusters(self,
                               businesses: pd.DataFrame,
                               num_clusters: Optional[int],
                               cluster_size: Optional[Tuple[int, int]],
                               economic_targets: Optional[Dict],
                               optimization_focus: str,
                               progress_callback: Optional[callable],
        params: Optional[Dict]) -> List[Dict]:
        """Construct clusters using a deterministic greedy heuristic."""
        # Ensure evaluation has access to runtime parameters (e.g., time_horizon)
        self.params = params or {}
        if businesses.empty:
            logger.warning("Greedy optimizer received no businesses; returning empty result")
            return []

        # Preserve original indices but order by composite strength signals
        sort_columns = []
        ascending = []
        if 'composite_score' in businesses.columns:
            sort_columns.append('composite_score')
            ascending.append(False)
        if 'revenue_estimate' in businesses.columns:
            sort_columns.append('revenue_estimate')
            ascending.append(False)
        if 'employees' in businesses.columns:
            sort_columns.append('employees')
            ascending.append(False)

        if sort_columns:
            ordered = businesses.sort_values(by=sort_columns, ascending=ascending)
        else:
            try:
                ordered = businesses.sort_values(by=businesses.columns.tolist(), axis=0)
            except Exception:
                ordered = businesses.sort_index()

        total_businesses = len(ordered)

        if not num_clusters or num_clusters <= 0:
            try:
                discovery = self.determine_optimal_clusters(ordered, economic_targets)
                num_clusters = max(1, discovery.get('optimal_k', 5))
                self.discovery_result = discovery
            except Exception as exc:
                logger.warning("Greedy optimizer could not determine optimal k automatically: %s", exc)
                num_clusters = min(5, max(1, total_businesses // 50)) or 3

        num_clusters = max(1, min(num_clusters, total_businesses))

        if cluster_size is None:
            # Derive reasonable bounds based on dataset size
            min_size = max(8, total_businesses // max(1, num_clusters * 3))
            max_size = max(min_size, min(200, math.ceil(total_businesses / max(1, num_clusters - 1)) + 5))
        else:
            min_size, max_size = cluster_size

        min_size = max(1, min(min_size, total_businesses))
        max_size = max(min_size, min(max_size, total_businesses))

        ordered_indices = list(ordered.index)
        start = 0
        clusters: List[Dict] = []

        if progress_callback:
            progress_callback(5)

        for idx in range(num_clusters):
            if start >= total_businesses:
                break

            remaining = total_businesses - start
            clusters_remaining = num_clusters - idx

            if clusters_remaining <= 1:
                end = total_businesses
            else:
                target_size = max(min_size, math.ceil(remaining / clusters_remaining))
                target_size = min(max_size, target_size)
                end = min(total_businesses, start + target_size)

            cluster_indices = ordered_indices[start:end]
            start = end

            cluster_businesses = businesses.loc[cluster_indices]
            cluster_data = self._extract_cluster_data(cluster_businesses, businesses)
            clusters.append(cluster_data)

            if progress_callback:
                progress_callback(min(95, int(((idx + 1) / num_clusters) * 100)))

        if progress_callback:
            progress_callback(100)

        logger.info("Greedy optimizer produced %d clusters (min=%d, max=%d, total=%d)",
                    len(clusters), min_size, max_size, total_businesses)

        self.params = params or {}

        return clusters

    def optimize_clusters_classic(self, businesses: pd.DataFrame, 
                         num_clusters: Optional[int] = None,
                         cluster_size: Optional[Tuple[int, int]] = None,
                         economic_targets: Optional[Dict] = None,
                         optimization_focus: str = 'balanced',
                         progress_callback: Optional[callable] = None,
                         params: Optional[Dict] = None,
                         force_greedy: bool = False) -> List[Dict]:
        """
        Classic NSGA-II optimization without community detection
        
        This is the original approach that treats all businesses equally
        """
        # Store optimization focus for use in evaluation
        self.optimization_focus = optimization_focus

        greedy_mode = force_greedy or self._runtime_flag("disable_nsga2")

        if greedy_mode:
            logger.info("Executing deterministic greedy clustering (NSGA-II disabled)")
            return self._build_greedy_clusters(
                businesses=businesses,
                num_clusters=num_clusters,
                cluster_size=cluster_size,
                economic_targets=economic_targets,
                optimization_focus=optimization_focus,
                progress_callback=progress_callback,
                params=params
            )

        # Store params for use in evaluation
        self.params = params or {}

        # Re-setup DEAP with the appropriate weights for this optimization focus
        self.setup_deap(optimization_focus)
        
        # Use ALL businesses for clustering - the dynamic threshold is just for guidance
        # This allows the genetic algorithm to discover natural clusters
        eligible_businesses = businesses.copy()
        logger.info(f"Using ALL {len(eligible_businesses)} businesses for clustering (no threshold filter)")
        
        # Determine cluster size constraints if not provided
        if cluster_size is None:
            logger.info("No cluster size specified. Using automatic size discovery...")
            # Intelligent size bounds based on number of eligible businesses
            total_businesses = len(eligible_businesses)
            
            # Minimum size: ensure economic viability (at least 10, or 2% of total)
            min_size = max(10, int(total_businesses * 0.02))
            
            # Maximum size: keep manageable (at most 200, or 20% of total)
            max_size = min(200, int(total_businesses * 0.20))
            
            # Ensure min <= max
            if min_size > max_size:
                min_size = int(total_businesses * 0.05)
                max_size = int(total_businesses * 0.15)
            
            cluster_size = (min_size, max_size)
            logger.info(f"Auto-determined cluster size range: {cluster_size[0]}-{cluster_size[1]} businesses")
        
        # Use automatic discovery if num_clusters not specified
        if num_clusters is None:
            logger.info("No number of clusters specified. Using automatic discovery...")
            discovery_result = self.determine_optimal_clusters(
                eligible_businesses,
                economic_targets=economic_targets,
                min_k=2,
                max_k=min(10, len(eligible_businesses) // cluster_size[0])  # Ensure feasible k
            )
            
            num_clusters = discovery_result['optimal_k']
            logger.info(f"Automatic discovery selected {num_clusters} clusters")
            logger.info(f"Reasoning: {discovery_result['reasoning']}")
            
            # Store discovery result for later reference
            self.discovery_result = discovery_result
        
        # Log NAICS distribution of eligible businesses
        naics_distribution = eligible_businesses['naics_code'].str[:3].value_counts()
        logger.info(f"NAICS distribution in eligible businesses:")
        for naics, count in naics_distribution.head(10).items():
            logger.info(f"  {naics}: {count} businesses")
        
        if len(eligible_businesses) < cluster_size[0]:
            logger.warning(f"Not enough eligible businesses ({len(eligible_businesses)}) for minimum cluster size ({cluster_size[0]})")
            # Try to create smaller clusters if possible
            if len(eligible_businesses) >= 5:  # Minimum viable cluster size
                logger.info(f"Adjusting cluster size to work with {len(eligible_businesses)} businesses")
                cluster_size = (5, min(cluster_size[1], len(eligible_businesses)))
            else:
                logger.error("Not enough businesses for any meaningful clustering")
                return []
        
        # Reset index for eligible businesses to ensure consistent indexing
        eligible_businesses = eligible_businesses.reset_index(drop=True)
        
        # Register genetic operators
        self.toolbox.register("individual", self.create_individual, 
                            eligible_businesses, cluster_size)
        self.toolbox.register("population", tools.initRepeat, list, 
                            self.toolbox.individual)
        # Pass both eligible businesses and all businesses for critical mass calculation
        self.toolbox.register("evaluate", lambda ind: self.evaluate_cluster(ind, eligible_businesses, eligible_businesses))
        self.toolbox.register("mate", self._custom_crossover, 
                            businesses=eligible_businesses,
                            cluster_size=cluster_size)
        self.toolbox.register("mutate", self._mutate_cluster, 
                            businesses=eligible_businesses)
        self.toolbox.register("select", tools.selNSGA2)
        
        # Create initial population
        population = self.toolbox.population(n=self.config.POPULATION_SIZE)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        
        # Run NSGA-II algorithm with progress tracking
        logger.info(f"Starting NSGA-II with population size: {self.config.POPULATION_SIZE}, generations: {self.config.GENERATIONS}")
        
        # Initialize hall of fame and logbook
        halloffame = tools.ParetoFront()
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "avg", "std", "min", "max"
        
        # Evaluate initial population
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update hall of fame
        halloffame.update(population)
        
        # Record statistics
        record = stats.compile(population)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        
        # Evolution loop with progress tracking
        total_steps = self.config.GENERATIONS
        for gen in range(1, self.config.GENERATIONS + 1):
            # Report progress if callback provided
            if progress_callback:
                progress_percent = int((gen / total_steps) * 100)
                progress_callback(progress_percent)
            
            # Select parents
            offspring = self.toolbox.select(population, len(population))
            offspring = [self.toolbox.clone(ind) for ind in offspring]
            
            # Apply crossover and mutation
            for i in range(0, len(offspring), 2):
                if random.random() < self.config.CROSSOVER_PROB:
                    if i + 1 < len(offspring):
                        self.toolbox.mate(offspring[i], offspring[i+1])
                        del offspring[i].fitness.values, offspring[i+1].fitness.values
            
            for i in range(len(offspring)):
                if random.random() < self.config.MUTATION_PROB:
                    self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values
            
            # Evaluate offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Update population using mu+lambda
            population[:] = self.toolbox.select(population + offspring, self.config.POPULATION_SIZE)
            
            # Update hall of fame
            halloffame.update(population)
            
            # Record statistics
            record = stats.compile(population)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            
            if gen % 10 == 0:
                logger.info(f"Generation {gen}: {record}")
        
        # Log diversity of final population
        logger.info("Final population diversity:")
        for i, ind in enumerate(population[:5]):  # Check first 5 individuals
            cluster_businesses = eligible_businesses.iloc[ind]
            naics_counts = cluster_businesses['naics_code'].str[:3].value_counts()
            logger.info(f"  Individual {i}: {len(naics_counts)} unique NAICS codes, top 3: {naics_counts.head(3).to_dict()}")
        
        # Extract Pareto front solutions
        pareto_front = tools.sortNondominated(population, len(population), True)[0]
        
        # Ensure diversity in selected clusters
        diverse_solutions = self._select_diverse_solutions(pareto_front, num_clusters, eligible_businesses)
        
        # Convert to cluster configurations
        clusters = self._extract_clusters(diverse_solutions, 
                                        eligible_businesses)
        
        return clusters
    
    def _mutate_cluster(self, individual: List[int], businesses: pd.DataFrame) -> Tuple[List[int]]:
        """Custom mutation operator for cluster configurations that maintains diversity"""
        if random.random() < 0.5:  # Add or remove a business
            available = set(range(len(businesses))) - set(individual)
            if available and random.random() < 0.5:  # Add
                # Add a business from an underrepresented NAICS category
                current_naics = businesses.iloc[individual]['naics_code'].str[:3].value_counts()
                available_positions = list(available)
                
                # Try to find underrepresented NAICS
                underrep_found = False
                for pos in random.sample(available_positions, min(20, len(available_positions))):
                    naics = businesses.iloc[pos]['naics_code'][:3]
                    if naics not in current_naics or current_naics[naics] < 3:
                        individual.append(pos)
                        underrep_found = True
                        break
                
                if not underrep_found:
                    # If no underrepresented NAICS, add random
                    individual.append(random.choice(available_positions))
            elif len(individual) > 10:  # Remove (keep minimum size)
                # Remove from overrepresented NAICS categories
                current_naics = businesses.iloc[individual]['naics_code'].str[:3].value_counts()
                if len(current_naics) > 0 and current_naics.iloc[0] > 5:
                    # Find businesses from the most common NAICS
                    most_common_naics = current_naics.index[0]
                    candidates = [idx for idx in individual 
                                if businesses.iloc[idx]['naics_code'].startswith(most_common_naics)]
                    if candidates:
                        individual.remove(random.choice(candidates))
                    else:
                        individual.remove(random.choice(individual))
                else:
                    individual.remove(random.choice(individual))
        else:  # Replace a business to increase diversity
            if len(individual) > 0:
                available = set(range(len(businesses))) - set(individual)
                if available:
                    # Replace with a business from a different NAICS category
                    current_naics = businesses.iloc[individual]['naics_code'].str[:3].unique()
                    available_positions = list(available)
                    
                    # Try to find diverse option
                    diverse_found = False
                    for pos in random.sample(available_positions, min(20, len(available_positions))):
                        naics = businesses.iloc[pos]['naics_code'][:3]
                        if naics not in current_naics:
                            individual[random.randint(0, len(individual)-1)] = pos
                            diverse_found = True
                            break
                    
                    if not diverse_found:
                        individual[random.randint(0, len(individual)-1)] = \
                            random.choice(available_positions)
        
        # Ensure uniqueness after mutation and preserve original order
        seen = set()
        individual[:] = [idx for idx in individual if not (idx in seen or seen.add(idx))]
        
        return individual,
    
    def _select_diverse_solutions(self, pareto_front: List, num_clusters: int, businesses: pd.DataFrame) -> List:
        """Select diverse solutions from Pareto front ensuring NO business overlap between clusters"""
        if len(pareto_front) <= 1:
            return pareto_front[:num_clusters]
        
        # Track which businesses have been assigned to clusters
        assigned_businesses = set()
        selected_solutions = []
        
        # Sort pareto front by combined fitness score
        scored_solutions = []
        for sol in pareto_front:
            # Prioritize high GDP impact with low risk
            combined_score = sol.fitness.values[0] - sol.fitness.values[4] * 100
            scored_solutions.append((combined_score, sol))
        
        scored_solutions.sort(key=lambda x: x[0], reverse=True)
        
        # Select solutions ensuring no business overlap
        for score, solution in scored_solutions:
            if len(selected_solutions) >= num_clusters:
                break
                
            # Check if this solution has overlap with already assigned businesses
            solution_businesses = set(solution)
            overlap = solution_businesses & assigned_businesses
            
            if not overlap:  # No overlap - add this cluster
                selected_solutions.append(solution)
                assigned_businesses.update(solution_businesses)
            else:
                # Try to create a modified version without overlap
                non_overlapping = [b for b in solution if b not in assigned_businesses]
                if len(non_overlapping) >= 20:  # Minimum viable cluster size
                    # Create new individual with non-overlapping businesses
                    new_individual = creator.Individual(non_overlapping)
                    new_individual.fitness.values = solution.fitness.values
                    selected_solutions.append(new_individual)
                    assigned_businesses.update(non_overlapping)
        
        # If we don't have enough clusters, create new ones from unassigned businesses
        if len(selected_solutions) < num_clusters:
            all_indices = set(range(len(businesses)))
            unassigned = list(all_indices - assigned_businesses)
            
            while len(selected_solutions) < num_clusters and len(unassigned) >= 20:
                # Create clusters from unassigned businesses grouped by industry
                remaining_clusters = num_clusters - len(selected_solutions)
                cluster_size = min(50, max(20, len(unassigned) // remaining_clusters))
                
                # Create DataFrame with just unassigned businesses, preserving original indices
                unassigned_mask = pd.Series(False, index=businesses.index)
                unassigned_mask.iloc[unassigned] = True
                unassigned_df = businesses[unassigned_mask]
                
                if len(unassigned_df) == 0:
                    break
                
                # Group by NAICS for better clustering
                naics_groups = unassigned_df.groupby(unassigned_df['naics_code'].str[:3])
                
                # Find the largest NAICS group
                largest_group_indices = []
                for naics, group in naics_groups:
                    group_indices = group.index.tolist()
                    if len(group_indices) > len(largest_group_indices):
                        largest_group_indices = group_indices
                
                if len(largest_group_indices) >= 10:
                    # Take up to cluster_size businesses from this group
                    cluster_indices = largest_group_indices[:cluster_size]
                else:
                    # Random selection from unassigned as fallback
                    cluster_indices = unassigned[:cluster_size]
                
                if len(cluster_indices) >= 10:
                    new_individual = creator.Individual(cluster_indices)
                    # Evaluate the new cluster
                    new_individual.fitness.values = self.evaluate_cluster(new_individual, businesses)
                    selected_solutions.append(new_individual)
                    assigned_businesses.update(cluster_indices)
                    # Remove assigned indices from unassigned list
                    unassigned = [idx for idx in unassigned if idx not in cluster_indices]
                else:
                    # Can't create viable cluster, stop trying
                    break
        
        # Final validation: ensure no business appears in multiple clusters
        final_solutions = []
        final_assigned = set()
        
        for solution in selected_solutions:
            # Remove any businesses that have already been assigned
            clean_indices = [idx for idx in solution if idx not in final_assigned]
            
            if len(clean_indices) >= 15:  # Minimum viable cluster
                clean_solution = creator.Individual(clean_indices)
                clean_solution.fitness.values = self.evaluate_cluster(clean_solution, businesses)
                final_solutions.append(clean_solution)
                final_assigned.update(clean_indices)
        
        logger.info(f"Selected {len(final_solutions)} diverse clusters from {len(pareto_front)} Pareto solutions")
        logger.info(f"Total unique businesses assigned: {len(final_assigned)}")
        
        return final_solutions
    
    def _extract_clusters(self, pareto_solutions: List, 
                         businesses: pd.DataFrame) -> List[Dict]:
        """Extract cluster configurations from Pareto solutions"""
        clusters = []
        
        for i, solution in enumerate(pareto_solutions):
            # Ensure solution contains unique indices
            unique_indices = list(dict.fromkeys(solution))  # Preserves order
            if len(unique_indices) < len(solution):
                logger.warning(f"Cluster {i+1} had {len(solution) - len(unique_indices)} duplicate businesses")
            
            cluster_businesses = businesses.iloc[unique_indices]
            
            # Determine cluster type dynamically based on business composition
            cluster_type = self._determine_cluster_type(cluster_businesses)
            
            # Generate descriptive cluster name
            cluster_name = self._generate_cluster_name(cluster_businesses, cluster_type, i+1)
            
            # Calculate cluster scores
            cluster_data = {
                "name": cluster_name,
                "type": cluster_type,
                "businesses": cluster_businesses.to_dict("records"),
                "business_count": len(cluster_businesses),
                "total_employees": cluster_businesses["employees"].sum(),
                "total_revenue": cluster_businesses["revenue_estimate"].sum(),
                "fitness_values": solution.fitness.values,
                "projected_gdp_impact": solution.fitness.values[0],
                "innovation_score": solution.fitness.values[1],
                "projected_jobs": solution.fitness.values[2],
                "synergy_score": solution.fitness.values[3],
                "risk_score": solution.fitness.values[4],
                # Unified metrics dictionary for downstream ML enhancer
                "metrics": {
                    "total_employees": cluster_businesses["employees"].sum(),
                    "total_revenue": cluster_businesses["revenue_estimate"].sum(),
                    "innovation_score": solution.fitness.values[1],
                    "avg_business_age": ((_dt.now().year - cluster_businesses["year_established"]).mean()),
                    "projected_gdp_impact": solution.fitness.values[0],
                    "projected_jobs": solution.fitness.values[2],
                    "risk_score": solution.fitness.values[4]  # Include risk_score in metrics
                }
            }
            
            # Add discovery metrics if available
            if hasattr(self, 'discovery_result') and self.discovery_result:
                cluster_data['discovery_metrics'] = {
                    'optimal_k': self.discovery_result['optimal_k'],
                    'reasoning': self.discovery_result['reasoning'],
                    'validation_scores': self.discovery_result['metrics']
                }
            
            # Calculate comprehensive cluster scores
            scores = self._calculate_cluster_scores(cluster_businesses, cluster_data)
            cluster_data.update(scores)
            # Provide strategic_score alias expected by ML enhancer
            cluster_data["strategic_score"] = scores.get("total_score", 0)
            
            # Add network metrics for visualization unless disabled
            if not self._runtime_flag("disable_network_metrics"):
                network_metrics = self._calculate_network_metrics(cluster_businesses)
                cluster_data["network_metrics"] = network_metrics

            clusters.append(cluster_data)
        
        return clusters
    
    def _determine_cluster_type(self, businesses: pd.DataFrame) -> str:
        """Dynamically determine cluster type based on discovered business composition"""
        # Count businesses by NAICS prefix (both 3 and 4 digit)
        naics_counts = {}
        
        for _, business in businesses.iterrows():
            naics = str(business.get('naics_code', ''))
            if len(naics) >= 3:
                # Count 3-digit prefix
                prefix_3 = naics[:3]
                naics_counts[prefix_3] = naics_counts.get(prefix_3, 0) + 1
                
                # Count 4-digit prefix if available
                if len(naics) >= 4:
                    prefix_4 = naics[:4]
                    naics_counts[prefix_4] = naics_counts.get(prefix_4, 0) + 1
        
        # Define cluster mappings - comprehensive including 3-digit codes
        logistics_codes = {
            # 3-digit
            "423": "Merchant Wholesalers",  # Added wholesale trade
            "484": "Trucking", 
            "488": "Support Activities for Transportation",
            "492": "Couriers", 
            "493": "Warehousing",
            # 4-digit for specificity
            "4841": "General Freight Trucking",
            "4842": "Specialized Freight Trucking",
            "4931": "Warehousing and Storage"
        }
        
        manufacturing_codes = {
            # 3-digit comprehensive
            "321": "Wood Products",
            "322": "Paper",
            "323": "Printing",
            "324": "Petroleum and Coal",
            "326": "Plastics and Rubber",
            "327": "Nonmetallic Mineral",
            "331": "Primary Metal",
            "332": "Fabricated Metal",
            "333": "Machinery",
            "334": "Computer and Electronic",
            "335": "Electrical Equipment",
            "336": "Transportation Equipment",
            "337": "Furniture",
            # 339 subcategories (excluding medical)
            "3392": "Communication Equipment",
            "3393": "Other Electrical Equipment",
            "3399": "Other Miscellaneous Manufacturing"
            # Note: 311 (Food) moved to animal_health
            # Note: 325 (Chemical) handled by specific 4-digit codes
            # Note: 3391 (Medical Equipment) moved to biosciences
        }
        
        biosciences_codes = {
            # 3-digit
            "621": "Ambulatory Healthcare",
            "622": "Hospitals",
            "623": "Nursing and Residential Care",
            # 4-digit specifics
            "3254": "Pharmaceutical",
            "3256": "Soap and Cleaning Compounds",
            "3391": "Medical Equipment",
            "5417": "Scientific R&D",
            "6215": "Medical Labs",
            # Handle 325 chemical subcategories
            "3251": "Basic Chemical",
            "3252": "Resin and Synthetic",
            "3255": "Paint and Coating",
            "3259": "Other Chemical"
        }
        
        technology_codes = {
            # 3-digit
            "334": "Computer and Electronic Manufacturing",  # Added
            "511": "Publishing (includes software)",
            "517": "Telecommunications",
            "518": "Data Processing and Hosting",
            "519": "Other Information Services",
            # 4-digit specifics - technology focused
            "5112": "Software Publishers",
            "5171": "Wired Telecommunications",  # Added
            "5172": "Wireless Telecommunications",  # Added
            "5182": "Data Processing",
            "5415": "Computer Systems Design",
            "5417": "Scientific Research and Development",  # Tech R&D
            "5418": "Advertising and Related Services"
        }
        
        professional_services_codes = {
            # 3-digit - be more selective
            "221": "Utilities",  # Added utilities
            "522": "Credit Intermediation",
            "523": "Securities and Financial",
            "524": "Insurance Carriers",
            "525": "Insurance Related Activities",  # Added
            "531": "Real Estate",
            "533": "Lessors of Nonfinancial Intangible Assets",  # Added
            # Remove broad "541" to be more specific with 4-digit codes
            "551": "Management of Companies",
            "561": "Administrative Support",
            # 4-digit specifics for professional services only
            "5411": "Legal Services",
            "5412": "Accounting Services",
            "5413": "Architecture/Engineering",
            "5414": "Specialized Design",
            "5416": "Management Consulting",  # Moved from technology
            "5419": "Other Professional Services"
        }
        
        community_services_codes = {
            # 3-digit
            "611": "Educational Services",
            "624": "Social Assistance",
            "711": "Performing Arts",
            "712": "Museums",
            "713": "Amusement and Recreation",
            "721": "Accommodation",
            "722": "Food Services",
            "811": "Repair and Maintenance",
            "812": "Personal Services",
            "813": "Religious/Civic Organizations",
            # 4-digit specifics
            "6111": "Elementary and Secondary Schools",
            "6113": "Colleges and Universities",
            "6116": "Other Schools and Instruction"
        }
        
        animal_health_codes = {
            # 3-digit
            "112": "Animal Production",
            "115": "Agriculture Support",
            "311": "Food Manufacturing",
            # 4-digit specifics
            "3111": "Animal Food",
            "3253": "Agricultural Chemicals",
            "1152": "Animal Production Support",
            "3112": "Grain and Oilseed",
            "3113": "Sugar and Confectionery",
            "3114": "Fruit and Vegetable",
            "3115": "Dairy Product",
            "3116": "Animal Slaughtering",
            "3117": "Seafood",
            "3118": "Bakeries",
            "3119": "Other Food"
        }
        
        # Calculate cluster scores based on business count per type
        cluster_scores = {
            "logistics": 0,
            "manufacturing": 0,
            "biosciences": 0,
            "technology": 0,
            "animal_health": 0,
            "professional_services": 0,
            "community_services": 0
        }
        
        # Track which NAICS codes contribute to multiple categories
        multi_category_naics = {}
        
        # Score each cluster type based on NAICS composition
        for naics, count in naics_counts.items():
            # Give higher weight to 4-digit matches (more specific)
            weight = 1.5 if len(naics) == 4 else 1.0
            matches = []
            
            # Check all categories (not elif) to handle overlaps
            if naics in logistics_codes:
                cluster_scores["logistics"] += count * weight
                matches.append("logistics")
                
            if naics in manufacturing_codes:
                cluster_scores["manufacturing"] += count * weight
                matches.append("manufacturing")
                
            if naics in biosciences_codes:
                cluster_scores["biosciences"] += count * weight
                matches.append("biosciences")
                
            if naics in technology_codes:
                cluster_scores["technology"] += count * weight
                matches.append("technology")
                
            if naics in animal_health_codes:
                cluster_scores["animal_health"] += count * weight
                matches.append("animal_health")
                
            if naics in professional_services_codes:
                cluster_scores["professional_services"] += count * weight
                matches.append("professional_services")
                
            if naics in community_services_codes:
                cluster_scores["community_services"] += count * weight
                matches.append("community_services")
            
            # Track multi-category codes
            if len(matches) > 1:
                multi_category_naics[naics] = matches
        
        # IMPROVED: Better tie-breaking and threshold logic
        total_weighted_businesses = sum(cluster_scores.values()) if cluster_scores else 1
        max_score = max(cluster_scores.values()) if cluster_scores else 0
        
        # Special handling for known KC strengths
        kc_strength_boost = {
            'logistics': 1.3,  # KC is a major logistics hub
            'biosciences': 1.2,  # Animal Health Corridor
            'manufacturing': 1.1,  # Strong manufacturing base
        }
        
        # Apply KC strength boosts
        for cluster_type, boost in kc_strength_boost.items():
            if cluster_type in cluster_scores:
                original_score = cluster_scores[cluster_type]
                cluster_scores[cluster_type] *= boost
                logger.info(f"Applied KC boost to {cluster_type}: {original_score:.0f} -> {cluster_scores[cluster_type]:.0f}")
        
        # Recalculate after boosts
        max_score = max(cluster_scores.values()) if cluster_scores else 0
        
        # Get all types with high scores (within 20% of max to allow more diversity)
        high_scoring_types = {
            type_: score for type_, score in cluster_scores.items() 
            if score >= max_score * 0.8
        }
        
        # If we have a clear winner (50% or more ahead of second place)
        sorted_scores = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_scores) > 1:
            first_score = sorted_scores[0][1]
            second_score = sorted_scores[1][1]
            
            # Check if any single type has more than 25% of weighted businesses (lowered threshold)
            if first_score > total_weighted_businesses * 0.25:
                dominant_type = sorted_scores[0][0]
                logger.info(f"Clear dominant cluster type '{dominant_type}' with scores: {cluster_scores}")
                return dominant_type
            
            # Or if it's 25% ahead of second place (lowered from 50%)
            if first_score > second_score * 1.25:  # 25% lead
                dominant_type = sorted_scores[0][0]
                logger.info(f"Clear dominant cluster type '{dominant_type}' with scores: {cluster_scores}")
                return dominant_type
            
            # Special case: if logistics or biosciences is top-2 and has meaningful presence
            if len(sorted_scores) >= 2:
                top_two_types = [sorted_scores[0][0], sorted_scores[1][0]]
                for strategic_type in ['logistics', 'biosciences']:
                    if strategic_type in top_two_types:
                        type_score = cluster_scores.get(strategic_type, 0)
                        if type_score > total_weighted_businesses * 0.15:  # At least 15% presence
                            logger.info(f"Strategic cluster type '{strategic_type}' recognized with score: {type_score}")
                            return strategic_type
        
        # IMPROVED: Handle ties and mixed clusters
        if len(high_scoring_types) > 1 or (max_score < total_weighted_businesses * 0.3):
            # Multiple high-scoring types - this is a mixed cluster
            
            # Check if service sectors are strong
            service_score = cluster_scores.get("professional_services", 0) + cluster_scores.get("community_services", 0)
            manufacturing_score = cluster_scores.get("manufacturing", 0)
            
            # Special cases for common combinations
            if "professional_services" in high_scoring_types and "community_services" in high_scoring_types:
                logger.info(f"Mixed cluster identified as 'service_hub' with scores: {cluster_scores}")
                return "service_hub"  # Mixed service cluster
                
            if service_score > manufacturing_score * 0.8:  # Services are competitive
                if "technology" in high_scoring_types:
                    logger.info(f"Mixed cluster identified as 'tech_services' with scores: {cluster_scores}")
                    return "tech_services"
                else:
                    logger.info(f"Mixed cluster identified as 'mixed_services' with scores: {cluster_scores}")
                    return "mixed_services"
            
            if "technology" in high_scoring_types and "manufacturing" in high_scoring_types:
                logger.info(f"Mixed cluster identified as 'advanced_manufacturing' with scores: {cluster_scores}")
                return "advanced_manufacturing"  # Tech-enabled manufacturing
            
            if "biosciences" in high_scoring_types and "technology" in high_scoring_types:
                logger.info(f"Mixed cluster identified as 'biotech' with scores: {cluster_scores}")
                return "biotech"  # Biotechnology cluster
            
            if "logistics" in high_scoring_types and "manufacturing" in high_scoring_types:
                logger.info(f"Mixed cluster identified as 'supply_chain' with scores: {cluster_scores}")
                return "supply_chain"  # Integrated supply chain cluster
            
            if "professional_services" in high_scoring_types and "technology" in high_scoring_types:
                logger.info(f"Mixed cluster identified as 'tech_services' with scores: {cluster_scores}")
                return "tech_services"  # Technology-enabled professional services
            
            if "community_services" in high_scoring_types and any(x in high_scoring_types for x in ["biosciences", "technology"]):
                logger.info(f"Mixed cluster identified as 'knowledge_services' with scores: {cluster_scores}")
                return "knowledge_services"  # Knowledge-based community services
            
            # Look at business characteristics for tie-breaking
            avg_employees = businesses['employees'].mean()
            avg_score = businesses.get('composite_score', businesses.get('total_score', 0)).mean()
            avg_patents = businesses.get('patent_count', 0).mean()
            
            # High patent count suggests technology or biosciences
            if avg_patents > 2:
                if "technology" in high_scoring_types:
                    logger.info(f"Tie-broken to 'technology' based on patents with scores: {cluster_scores}")
                    return "technology"
                elif "biosciences" in high_scoring_types:
                    logger.info(f"Tie-broken to 'biosciences' based on patents with scores: {cluster_scores}")
                    return "biosciences"
            
            # Large employers often indicate manufacturing or logistics
            if avg_employees > 100:
                if "manufacturing" in high_scoring_types:
                    logger.info(f"Tie-broken to 'manufacturing' based on employment with scores: {cluster_scores}")
                    return "manufacturing"
                elif "logistics" in high_scoring_types:
                    logger.info(f"Tie-broken to 'logistics' based on employment with scores: {cluster_scores}")
                    return "logistics"
            
            # High innovation scores suggest technology or biosciences
            if avg_score > 70:
                if "technology" in high_scoring_types:
                    logger.info(f"Tie-broken to 'technology' based on innovation score with scores: {cluster_scores}")
                    return "technology"
                elif "biosciences" in high_scoring_types:
                    logger.info(f"Tie-broken to 'biosciences' based on innovation score with scores: {cluster_scores}")
                    return "biosciences"
            
            # If still tied, return "mixed" with descriptor
            top_types = list(high_scoring_types.keys())[:2]
            mixed_type = f"{top_types[0]}_{top_types[1]}"
            logger.info(f"Mixed cluster identified as '{mixed_type}' with scores: {cluster_scores}")
            return mixed_type
        
        # Single dominant type
        dominant_type = max(cluster_scores.items(), key=lambda x: x[1])[0]
        
        # If the dominant type represents less than 25% of businesses, it's truly mixed
        if max_score < len(businesses) * 0.25:
            logger.info(f"No clear dominant type (< 25% threshold), returning 'mixed' with scores: {cluster_scores}")
            return "mixed"
        
        logger.info(f"Cluster type determined as '{dominant_type}' with scores: {cluster_scores}")
        return dominant_type
        
        return dominant_type
    
    def _calculate_cluster_scores(self, businesses: pd.DataFrame, 
                                cluster_data: Dict) -> Dict:
        """Calculate comprehensive scores for a cluster"""
        scores = {}
        
        # Natural assets score (based on location and resources)
        cluster_type = cluster_data["type"]
        if cluster_type == "logistics":
            scores["natural_assets_score"] = 90  # KC has excellent rail/highway
        elif cluster_type == "biosciences":
            scores["natural_assets_score"] = 70  # Good but not exceptional
        else:
            scores["natural_assets_score"] = 75
        
        # Infrastructure score
        central_counties = ["Jackson County", "Johnson County"]
        central_businesses = businesses[businesses["county"].str.contains(
            "|".join(central_counties), na=False)].shape[0]
        scores["infrastructure_score"] = min(100, 60 + (central_businesses / len(businesses)) * 40)
        
        # Workforce score
        total_employees = cluster_data["total_employees"]
        scores["workforce_score"] = min(100, 50 + (total_employees / 100) * 5)
        
        # Innovation score (already calculated)
        scores["innovation_capacity_score"] = min(100, cluster_data["innovation_score"] / 10)
        
        # Market access score
        if cluster_type in ["logistics", "manufacturing"]:
            scores["market_access_score"] = 85
        else:
            scores["market_access_score"] = 75
        
        # Geopolitical stability (KC is stable)
        scores["geopolitical_score"] = 80
        
        # Resilience score
        industry_diversity = len(businesses["naics_code"].str[:3].unique())
        scores["resilience_score"] = min(100, 50 + industry_diversity * 10)
        
        # Total score (weighted) using configurable weights
        scores["total_score"] = (
            scores["natural_assets_score"] * self.cluster_weights.get("natural_assets", 0.20) +
            scores["infrastructure_score"] * self.cluster_weights.get("infrastructure", 0.20) +
            scores["workforce_score"] * self.cluster_weights.get("workforce", 0.15) +
            scores["innovation_capacity_score"] * self.cluster_weights.get("innovation", 0.15) +
            scores["market_access_score"] * self.cluster_weights.get("market_access", 0.15) +
            scores["geopolitical_score"] * self.cluster_weights.get("geopolitical", 0.10) +
            scores["resilience_score"] * self.cluster_weights.get("resilience", 0.05)
        )
        
        # Longevity score
        scores["longevity_score"] = self._calculate_longevity_score(businesses, cluster_data)
        
        return scores
    
    def _calculate_network_metrics(self, businesses: pd.DataFrame) -> Dict:
        """Calculate network metrics for business relationships"""
        # Get top businesses by size/importance
        businesses_sorted = businesses.sort_values('employees', ascending=False)
        
        # Get central businesses (top 10 or all if less)
        n_central = min(10, len(businesses_sorted))
        central_businesses = []
        
        for idx, business in businesses_sorted.head(n_central).iterrows():
            business_name = business.get('name', business.get('company_name', f'Business {idx}'))
            # Truncate long names and ensure they're strings
            if pd.notna(business_name):
                business_name = str(business_name)[:40]
            else:
                business_name = f'Unknown Business {idx}'
            central_businesses.append(business_name)
        
        # Calculate network density based on industry relationships
        naics_codes = businesses['naics_code'].str[:3].value_counts()
        unique_industries = len(naics_codes)
        total_businesses = len(businesses)
        
        # Higher density if businesses are in related industries
        industry_concentration = naics_codes.iloc[0] / total_businesses if len(naics_codes) > 0 else 0
        network_density = min(0.8, 0.3 + (industry_concentration * 0.5))
        
        # Calculate clustering coefficient (businesses in same sub-industries)
        avg_clustering = 0
        for naics, count in naics_codes.items():
            if count > 1:
                # Businesses in same industry are more likely to cluster
                avg_clustering += (count / total_businesses) * 0.8
        
        # Calculate synergy score based on complementary industries
        synergy_score = self._calculate_synergies(businesses)
        
        # Network resilience based on diversity
        resilience = min(100, 40 + (unique_industries * 10))
        
        return {
            'central_businesses': central_businesses,
            'network_density': round(network_density, 2),
            'avg_clustering': round(avg_clustering, 2),
            'synergy_score': min(100, synergy_score),
            'network_resilience': resilience,
            'total_connections': int(total_businesses * (total_businesses - 1) * network_density / 2)
        }
    
    def _generate_cluster_name(self, businesses: pd.DataFrame, cluster_type: str, index: int) -> str:
        """Generate a descriptive name for the cluster based on its composition"""
        # Get geographic concentration with more variation
        county_counts = businesses['county'].value_counts()
        
        # Get top counties and their percentages
        total_businesses = len(businesses)
        if not county_counts.empty:
            top_county_pct = county_counts.iloc[0] / total_businesses
            
            # If one county dominates (>60%), use it
            if top_county_pct > 0.6:
                top_county = county_counts.index[0]
            # If top 2 counties have similar counts, use both
            elif len(county_counts) > 1 and county_counts.iloc[1] / total_businesses > 0.25:
                top_county = f"{county_counts.index[0]}-{county_counts.index[1]}"
            # If distributed across many counties, use regional descriptor
            elif len(county_counts) > 3:
                top_county = "Greater KC Metro"
            else:
                top_county = county_counts.index[0]
        else:
            top_county = "Regional"
        
        # Get industry specifics with more detail
        naics_counts = businesses['naics_code'].str[:4].value_counts()
        
        # Also look at prominent businesses for naming inspiration
        top_businesses = businesses.nlargest(3, 'composite_score') if 'composite_score' in businesses.columns else businesses.head(3)
        prominent_names = [b.get('name', '') for _, b in top_businesses.iterrows() if b.get('name')]
        
        # Check if cluster has a major anchor company
        largest_employer = businesses.nlargest(1, 'employees').iloc[0] if not businesses.empty else None
        if largest_employer is not None and largest_employer.get('employees', 0) > total_businesses * 50:  # Major employer
            anchor_name = largest_employer.get('name', '')
            if anchor_name and len(anchor_name) < 30:
                return f"{anchor_name} Innovation District"
        
        # Map cluster types to descriptive names
        type_descriptors = {
            "logistics": "Transportation & Warehousing",
            "manufacturing": "Advanced Manufacturing",
            "technology": "Technology & Innovation",
            "biosciences": "Life Sciences & Healthcare",
            "animal_health": "Animal Health & AgTech",
            "mixed": "Diversified Business",
            "mixed_services": "Mixed Services",
            "service_hub": "Service Hub",
            "tech_services": "Technology Services",
            "advanced_manufacturing": "Advanced Manufacturing",
            "biotech": "Biotechnology",
            "supply_chain": "Supply Chain",
            "knowledge_services": "Knowledge Services"
        }
        
        # Get specific industry focus
        industry_focus = ""
        if cluster_type == "logistics" and "484" in naics_counts.index[:3].str[:3].values:
            industry_focus = "Freight & Distribution"
        elif cluster_type == "manufacturing":
            if "332" in naics_counts.index[:3].str[:3].values:
                industry_focus = "Metal Fabrication"
            elif "333" in naics_counts.index[:3].str[:3].values:
                industry_focus = "Machinery & Equipment"
            elif "334" in naics_counts.index[:3].str[:3].values:
                industry_focus = "Electronics Manufacturing"
            elif "336" in naics_counts.index[:3].str[:3].values:
                industry_focus = "Transportation Equipment"
            elif "335" in naics_counts.index[:3].str[:3].values:
                industry_focus = "Electrical Equipment"
            elif "326" in naics_counts.index[:3].str[:3].values:
                industry_focus = "Plastics & Rubber"
            elif "327" in naics_counts.index[:3].str[:3].values:
                industry_focus = "Nonmetallic Mineral Products"
            else:
                # Use index to differentiate generic manufacturing clusters
                industry_focus = f"Industrial Sector {index}"
        elif cluster_type == "technology":
            if "5112" in naics_counts.index[:3].values:
                industry_focus = "Software Development"
            elif "5415" in naics_counts.index[:3].values:
                industry_focus = "IT Services"
            else:
                industry_focus = "Digital Innovation"
        elif cluster_type == "biosciences":
            if "3254" in naics_counts.index[:3].values:
                industry_focus = "Pharmaceutical"
            elif "3391" in naics_counts.index[:3].values:
                industry_focus = "Medical Devices"
            else:
                industry_focus = "Healthcare Innovation"
        elif cluster_type == "mixed_services":
            # Analyze the mix to provide specific focus
            top_naics = naics_counts.index[:5].tolist()
            if any(code.startswith("541") for code in top_naics):
                industry_focus = "Professional Services"
            elif any(code.startswith("611") for code in top_naics):
                industry_focus = "Education & Training"
            elif any(code.startswith("621") for code in top_naics):
                industry_focus = "Healthcare Services"
            else:
                industry_focus = f"Services Cluster {index}"
        elif cluster_type in ["service_hub", "tech_services", "knowledge_services"]:
            # These already have descriptive names
            industry_focus = ""
        
        # Build descriptive name with more variation
        name_templates = [
            "{geo} {focus} Hub",
            "{geo} {focus} District",
            "{geo} {focus} Corridor",
            "{geo} {focus} Center",
            "{geo} {focus} Park",
            "{geo} {focus} Campus"
        ]
        
        # Select template based on cluster characteristics
        if cluster_type in ["logistics", "supply_chain"]:
            template = name_templates[2]  # Corridor
        elif cluster_type in ["technology", "biosciences"]:
            template = name_templates[5]  # Campus
        elif cluster_type in ["manufacturing", "advanced_manufacturing"]:
            template = name_templates[4]  # Park
        elif "mixed" in cluster_type or "service" in cluster_type:
            template = name_templates[index % 3]  # Vary between Hub, District, Center
        else:
            template = name_templates[3]  # Center
        
        # Handle geographic variations
        if top_county == "Regional":
            geo_name = "KC Regional"
        elif "-" in top_county:  # Multi-county
            geo_name = top_county.split("-")[0]  # Use first county
        elif top_county == "Greater KC Metro":
            geo_name = ["Metro KC", "Greater Kansas City", "KC Area"][index % 3]
        else:
            geo_name = top_county
        
        # Get focus descriptor
        if industry_focus:
            focus = industry_focus
        else:
            descriptor = type_descriptors.get(cluster_type, "Business")
            # Add index-based variation for similar clusters
            if cluster_type == "mixed_services":
                variations = [descriptor, "Integrated Business", "Commercial Services", "Enterprise Services"]
                focus = variations[index % len(variations)]
            else:
                focus = descriptor
        
        return template.format(geo=geo_name, focus=focus)
    
    def _calculate_longevity_score(self, businesses: pd.DataFrame, 
                                  cluster_data: Dict) -> float:
        """Calculate cluster longevity score (0-10 scale)"""
        score = 5.0  # Base score
        
        # Infrastructure quality boost
        cluster_type = cluster_data["type"]
        if cluster_type == "logistics":
            score += 2.0  # KC's rail infrastructure is long-term asset
        
        # Workforce stability
        from datetime import datetime as _dt
        avg_business_age = _dt.now().year - businesses["year_established"].mean()
        if avg_business_age > 10:
            score += 1.0
        
        # Market growth trends
        growth_industries = ["493", "3254", "5415"]  # High-growth NAICS
        growth_businesses = businesses[businesses["naics_code"].str[:3].isin(growth_industries)]
        if len(growth_businesses) / len(businesses) > 0.5:
            score += 1.5
        
        # Risk factors reduction
        if cluster_data["risk_score"] < 30:
            score += 0.5
        
        return min(10, max(0, score))











