"""Strategic Cluster Formation Module - Industry-First Approach"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class StrategicClusterFormer:
    """Forms economic clusters based on strategic industry alignment and KC strengths"""
    
    def __init__(self, config):
        self.config = config
        self.pareto_threshold = getattr(config, 'PARETO_THRESHOLD', 0.95)
        self.university_data = None
        
        # Define Kansas City's strategic cluster types based on regional strengths
        self.cluster_definitions = {
            "logistics": {
                "naics_codes": ["484", "493", "488", "492", "4931"],  # Trucking, Warehousing, Rail, Couriers
                "description": "Logistics and Distribution",
                "kc_strengths": ["Second-largest rail network", "Central US location", "I-35/I-70 junction"],
                "required_infrastructure": ["rail_access", "highway_access"],
                "synergies": ["manufacturing", "animal_health"]
            },
            "manufacturing": {
                "naics_codes": ["332", "333", "336", "334", "335"],  # Metal, Machinery, Auto, Electronics
                "description": "Advanced Manufacturing",
                "kc_strengths": ["Ford Plant", "Honeywell", "Strong industrial base"],
                "required_infrastructure": ["rail_access", "skilled_trades"],
                "synergies": ["logistics", "technology"]
            },
            "biosciences": {
                "naics_codes": ["3254", "3391", "5417", "6215", "3345"],  # Pharma, Medical, R&D, Labs
                "description": "Biosciences and Healthcare",
                "kc_strengths": ["KU Medical Center", "Stowers Institute", "Children's Mercy"],
                "required_infrastructure": ["university_partnership", "stem_workforce"],
                "synergies": ["animal_health", "technology"]
            },
            "animal_health": {
                "naics_codes": ["3253", "1152", "3111", "4245", "5413"],  # Ag chemicals, Animal production, Feed
                "description": "Animal Health Corridor",
                "kc_strengths": ["Global animal health capital", "Zoetis", "Boehringer Ingelheim"],
                "required_infrastructure": ["university_partnership", "broadband"],
                "synergies": ["biosciences", "logistics"]
            },
            "technology": {
                "naics_codes": ["5415", "5416", "5418", "5112", "5182"],  # IT, Data, Internet, Software
                "description": "Technology and Innovation",
                "kc_strengths": ["Google Fiber", "Cerner", "Growing startup ecosystem"],
                "required_infrastructure": ["broadband", "stem_workforce"],
                "synergies": ["biosciences", "manufacturing"]
            }
        }
        
        # KC-specific scoring multipliers
        self.kc_asset_multipliers = {
            "rail_access": {"logistics": 1.5, "manufacturing": 1.3},
            "highway_access": {"logistics": 1.4, "manufacturing": 1.2},
            "university_partnership": {"biosciences": 1.4, "technology": 1.3, "animal_health": 1.3},
            "river_access": {"manufacturing": 1.2, "logistics": 1.1},
            "airport_proximity": {"logistics": 1.3, "technology": 1.2},
            "stem_workforce": {"technology": 1.4, "biosciences": 1.3}
        }
    
    def set_university_data(self, university_data: List[Dict]):
        """Set university research data for cluster analysis"""
        self.university_data = university_data
        logger.info(f"Set university data with {len(university_data) if university_data else 0} universities")
    
    def form_strategic_clusters(self, businesses: pd.DataFrame, selected_clusters: List[str] = None) -> List[Dict]:
        """Form clusters based on strategic industry alignment"""
        logger.info("Starting strategic cluster formation")
        
        # If no clusters selected, use all
        if selected_clusters is None:
            selected_clusters = list(self.cluster_definitions.keys())
        
        clusters = []
        
        for cluster_type, definition in self.cluster_definitions.items():
            # Skip if not selected
            if cluster_type not in selected_clusters:
                logger.info(f"Skipping {cluster_type} cluster (not selected)")
                continue
            logger.info(f"Forming {cluster_type} cluster with NAICS codes: {definition['naics_codes']}")
            
            # 1. Select businesses in this industry
            industry_businesses = self._select_industry_businesses(
                businesses, 
                definition["naics_codes"]
            )
            
            if len(industry_businesses) < self.config.MIN_CLUSTER_SIZE:
                logger.warning(f"Not enough businesses for {cluster_type} cluster: {len(industry_businesses)}")
                continue
            
            # 2. Score businesses within industry context
            scored_businesses = self._score_businesses_for_cluster(
                industry_businesses,
                cluster_type,
                definition
            )
            
            # 3. Apply business-level Pareto filter (top 95% within industry)
            elite_businesses = self._apply_business_pareto_filter(scored_businesses)
            
            if len(elite_businesses) < self.config.MIN_CLUSTER_SIZE:
                logger.warning(f"Not enough elite businesses for {cluster_type}: {len(elite_businesses)}")
                continue
            
            # 4. Optimize cluster composition
            optimized_cluster = self._optimize_cluster_composition(
                elite_businesses,
                cluster_type,
                definition
            )
            
            # 5. Calculate cluster-level scores
            cluster_data = self._evaluate_strategic_cluster(
                optimized_cluster,
                cluster_type,
                definition
            )
            
            clusters.append(cluster_data)
            logger.info(f"Successfully created {cluster_type} cluster with {cluster_data['business_count']} businesses")
        
        # Log summary of clusters created
        logger.info(f"Total clusters created: {len(clusters)}")
        for cluster in clusters:
            logger.info(f"  - {cluster['name']} (type: {cluster['type']}) with {cluster['business_count']} businesses")
        
        # 6. Apply cluster-level Pareto filtering
        filtered_clusters = self._apply_cluster_pareto_filter(clusters)
        
        # 7. Add cross-cluster synergies
        self._evaluate_cluster_synergies(filtered_clusters)
        
        return filtered_clusters
    
    def _select_industry_businesses(self, businesses: pd.DataFrame, naics_codes: List[str]) -> pd.DataFrame:
        """Select businesses matching industry NAICS codes"""
        # Match on 3, 4, or full NAICS code prefix
        mask = pd.Series([False] * len(businesses), index=businesses.index)
        
        for code in naics_codes:
            # Check if any business NAICS code starts with this prefix
            mask |= businesses['naics_code'].str.startswith(code)
        
        selected = businesses[mask].copy()
        logger.info(f"Selected {len(selected)} businesses from {len(businesses)} total")
        
        # Debug: Show sample of selected businesses
        if len(selected) > 0:
            sample_naics = selected['naics_code'].head(5).tolist()
            logger.debug(f"Sample NAICS codes selected: {sample_naics}")
        else:
            logger.warning(f"No businesses matched NAICS codes: {naics_codes}")
            
        return selected
    
    def _score_businesses_for_cluster(self, businesses: pd.DataFrame, 
                                    cluster_type: str, 
                                    definition: Dict) -> pd.DataFrame:
        """Score businesses within their industry cluster context"""
        scored = businesses.copy()
        
        # Base scoring (Innovation, Market, Competition)
        # Already done in business_scorer, but adjust for cluster context
        
        # Apply KC asset multipliers
        for _, business in scored.iterrows():
            multiplier = 1.0
            
            # Check infrastructure alignment
            for infra in definition.get("required_infrastructure", []):
                if self._business_has_infrastructure(business, infra):
                    asset_mult = self.kc_asset_multipliers.get(infra, {}).get(cluster_type, 1.0)
                    multiplier *= asset_mult
            
            # Geographic bonus for concentration
            if cluster_type == "logistics" and business.get("county") in ["Jackson County", "Clay County"]:
                multiplier *= 1.2  # Near rail hubs
            elif cluster_type == "biosciences" and business.get("county") == "Johnson County":
                multiplier *= 1.15  # Near KU Med
            
            # Apply university research strength bonus
            uni_multiplier = self._get_university_research_multiplier(business, cluster_type)
            multiplier *= uni_multiplier
            
            # Apply multiplier to composite score
            scored.loc[scored.index == business.name, 'cluster_adjusted_score'] = (
                business['composite_score'] * multiplier
            )
        
        return scored
    
    def _get_university_research_multiplier(self, business: pd.Series, cluster_type: str) -> float:
        """Calculate multiplier based on nearby university research strength"""
        if not self.university_data:
            return 1.0
        
        multiplier = 1.0
        county = business.get("county", "")
        
        # Map counties to nearby universities
        county_universities = {
            "Johnson County": ["University of Kansas"],
            "Jackson County": ["University of Missouri-Kansas City"],
            "Wyandotte County": ["University of Kansas", "Kansas State University"],
            "Clay County": ["University of Missouri-Kansas City"]
        }
        
        nearby_unis = county_universities.get(county, [])
        if not nearby_unis:
            return multiplier
        
        # Calculate research strength in relevant cluster
        total_funding = 0
        total_grants = 0
        
        for uni in self.university_data:
            if uni["name"] in nearby_unis:
                # Check if university has research in this cluster area
                research_clusters = uni.get("research_clusters", {})
                cluster_funding = research_clusters.get("funding", {}).get(cluster_type, 0)
                cluster_grants = research_clusters.get("counts", {}).get(cluster_type, 0)
                
                total_funding += cluster_funding
                total_grants += cluster_grants
        
        # Apply multiplier based on research strength
        if total_funding > 10000000:  # $10M+ in relevant research
            multiplier *= 1.25
        elif total_funding > 5000000:  # $5M+ in relevant research
            multiplier *= 1.15
        elif total_funding > 1000000:  # $1M+ in relevant research
            multiplier *= 1.05
        
        if total_grants > 10:  # Strong research activity
            multiplier *= 1.1
        
        logger.debug(f"University research multiplier for {cluster_type} in {county}: {multiplier:.2f} "
                    f"(funding: ${total_funding:,}, grants: {total_grants})")
        
        return multiplier
    
    def _business_has_infrastructure(self, business: pd.Series, infra_type: str) -> bool:
        """Check if business location has required infrastructure"""
        # Simplified - in reality would check actual infrastructure data
        county_infrastructure = {
            "Jackson County": ["rail_access", "highway_access", "broadband", "airport_proximity"],
            "Clay County": ["highway_access", "broadband"],
            "Johnson County": ["highway_access", "broadband", "university_partnership"],
            "Wyandotte County": ["rail_access", "highway_access", "river_access"],
        }
        
        county = business.get("county", "")
        return infra_type in county_infrastructure.get(county, [])
    
    def _apply_business_pareto_filter(self, businesses: pd.DataFrame) -> pd.DataFrame:
        """Apply Pareto threshold within industry"""
        if businesses.empty:
            return businesses
        
        top_score = businesses['cluster_adjusted_score'].max()
        threshold = top_score * (1 - self.pareto_threshold)
        
        filtered = businesses[businesses['cluster_adjusted_score'] >= threshold]
        logger.info(f"Business Pareto filter: {len(businesses)} -> {len(filtered)} (threshold: {threshold:.2f})")
        
        return filtered
    
    def _optimize_cluster_composition(self, businesses: pd.DataFrame, 
                                    cluster_type: str,
                                    definition: Dict) -> pd.DataFrame:
        """Optimize the mix of businesses in the cluster"""
        # Sort by adjusted score
        businesses = businesses.sort_values('cluster_adjusted_score', ascending=False)
        
        # Target composition based on cluster type
        if cluster_type == "logistics":
            # Want mix of trucking, warehousing, rail
            composition = {
                "484": 0.4,  # Trucking
                "493": 0.3,  # Warehousing
                "488": 0.2,  # Rail
                "other": 0.1
            }
        elif cluster_type == "biosciences":
            # Want R&D heavy with some manufacturing
            composition = {
                "5417": 0.4,  # R&D
                "3254": 0.3,  # Pharma manufacturing
                "3391": 0.2,  # Medical equipment
                "other": 0.1
            }
        else:
            # Default balanced composition
            composition = None
        
        if composition:
            selected = self._select_by_composition(businesses, composition)
        else:
            # Take top performers up to max cluster size
            max_size = min(self.config.MAX_CLUSTER_SIZE, len(businesses))
            selected = businesses.head(max_size)
        
        return selected
    
    def _select_by_composition(self, businesses: pd.DataFrame, composition: Dict) -> pd.DataFrame:
        """Select businesses to match target composition"""
        selected_indices = []
        remaining = businesses.copy()
        
        target_size = min(self.config.MAX_CLUSTER_SIZE, len(businesses))
        
        for naics_prefix, target_pct in composition.items():
            if naics_prefix == "other":
                continue
                
            target_count = int(target_size * target_pct)
            matching = remaining[remaining['naics_code'].str.startswith(naics_prefix)]
            
            if len(matching) > 0:
                take = min(target_count, len(matching))
                selected_indices.extend(matching.head(take).index.tolist())
                remaining = remaining.drop(matching.head(take).index)
        
        # Fill remainder with top scored businesses
        remainder = target_size - len(selected_indices)
        if remainder > 0 and len(remaining) > 0:
            selected_indices.extend(remaining.head(remainder).index.tolist())
        
        return businesses.loc[selected_indices]
    
    def _evaluate_strategic_cluster(self, businesses: pd.DataFrame, 
                                  cluster_type: str,
                                  definition: Dict) -> Dict:
        """Evaluate cluster-level strategic metrics"""
        
        # Calculate base metrics
        total_employees = businesses['employees'].sum()
        total_revenue = businesses['revenue_estimate'].sum()
        avg_score = businesses['cluster_adjusted_score'].mean()
        
        # Economic impact with multipliers
        # Following IMPLAN/RIMS II methodology for economic impact
        base_multiplier = self._get_cluster_multiplier(cluster_type)
        # Apply industry-specific output multipliers (not revenue multipliers)
        # GDP impact = Direct Output * Output Multiplier * GDP/Output Ratio
        gdp_output_ratios = {
            "logistics": 0.45,      # Transportation/warehousing GDP/output
            "manufacturing": 0.35,  # Manufacturing GDP/output
            "technology": 0.65,     # Information sector GDP/output
            "biosciences": 0.55,    # Pharma/biotech GDP/output
            "animal_health": 0.50,  # Specialized manufacturing
            "mixed": 0.45
        }
        gdp_ratio = gdp_output_ratios.get(cluster_type, 0.45)
        gdp_impact = total_revenue * base_multiplier * gdp_ratio
        
        # Job creation using BLS employment multipliers
        # Employment multipliers are typically lower than output multipliers
        employment_multipliers = {
            "logistics": 1.8,       # 0.8 indirect jobs per direct job
            "manufacturing": 1.6,   # 0.6 indirect jobs per direct job  
            "technology": 2.2,      # 1.2 indirect jobs per direct job
            "biosciences": 2.0,     # 1.0 indirect jobs per direct job
            "animal_health": 1.9,   # 0.9 indirect jobs per direct job
            "mixed": 1.7
        }
        emp_multiplier = employment_multipliers.get(cluster_type, 1.7)
        direct_jobs = total_employees
        indirect_jobs = int(direct_jobs * (emp_multiplier - 1))
        total_jobs = direct_jobs + indirect_jobs
        
        # Innovation metrics
        total_patents = businesses['patent_count'].sum()
        total_sbir = businesses['sbir_awards'].sum()
        innovation_score = (total_patents * 10 + total_sbir * 20) / len(businesses)
        
        # Strategic alignment with KC assets
        strategic_score = self._calculate_strategic_alignment(
            businesses, cluster_type, definition
        )
        
        # Synergy potential
        synergy_score = self._calculate_synergy_potential(
            businesses, definition.get("synergies", [])
        )
        
        # Risk assessment
        risk_score = self._calculate_cluster_risk(businesses, cluster_type)
        
        # Overall cluster score (weighted)
        cluster_score = (
            strategic_score * 0.25 +
            avg_score * 0.20 +
            (gdp_impact / 1e9) * 10 * 0.20 +  # Normalize GDP to 0-100 scale
            innovation_score * 0.15 +
            synergy_score * 0.15 +
            (100 - risk_score) * 0.05
        )
        
        return {
            "type": cluster_type,
            "name": definition["description"],
            "businesses": businesses.to_dict("records"),
            "business_count": len(businesses),
            "cluster_score": cluster_score,
            "strategic_score": strategic_score,
            "metrics": {
                "total_employees": total_employees,
                "total_revenue": total_revenue,
                "projected_gdp_impact": gdp_impact,
                "projected_jobs": total_jobs,
                "direct_jobs": direct_jobs,
                "indirect_jobs": indirect_jobs,
                "innovation_score": innovation_score,
                "synergy_score": synergy_score,
                "risk_score": risk_score
            },
            "kc_strengths_utilized": definition["kc_strengths"],
            "required_infrastructure": definition["required_infrastructure"],
            "synergy_clusters": definition["synergies"]
        }
    
    def _get_cluster_multiplier(self, cluster_type: str) -> float:
        """Get economic multiplier for cluster type"""
        multipliers = {
            "logistics": 2.8,
            "manufacturing": 2.5,
            "biosciences": 2.3,
            "animal_health": 2.2,
            "technology": 2.0
        }
        return multipliers.get(cluster_type, 2.0)
    
    def _calculate_strategic_alignment(self, businesses: pd.DataFrame, 
                                     cluster_type: str, 
                                     definition: Dict) -> float:
        """Calculate how well cluster aligns with KC strategic assets"""
        score = 50.0  # Base score
        
        # Infrastructure alignment
        for infra in definition["required_infrastructure"]:
            businesses_with_infra = sum(
                self._business_has_infrastructure(row, infra) 
                for _, row in businesses.iterrows()
            )
            if businesses_with_infra / len(businesses) > 0.7:
                score += 10
        
        # Geographic concentration in strategic areas
        strategic_counties = {
            "logistics": ["Jackson County", "Clay County", "Wyandotte County"],
            "biosciences": ["Johnson County", "Jackson County"],
            "technology": ["Jackson County", "Johnson County"]
        }
        
        if cluster_type in strategic_counties:
            strategic_businesses = businesses[
                businesses['county'].isin(strategic_counties[cluster_type])
            ]
            if len(strategic_businesses) / len(businesses) > 0.6:
                score += 15
        
        # Size and scale bonus
        if len(businesses) >= 50:
            score += 10
        elif len(businesses) >= 30:
            score += 5
        
        return min(100, score)
    
    def _calculate_synergy_potential(self, businesses: pd.DataFrame, 
                                   synergy_clusters: List[str]) -> float:
        """Calculate potential for synergies with other clusters"""
        # Simplified - would need actual cluster data
        base_score = 50.0
        
        # Industry diversity within cluster
        unique_naics = businesses['naics_code'].str[:3].nunique()
        if unique_naics > 3:
            base_score += 10
        
        # Size distribution (mix of large and small)
        emp_std = businesses['employees'].std()
        if emp_std > 50:
            base_score += 10
        
        # Number of potential synergy clusters
        base_score += len(synergy_clusters) * 5
        
        return min(100, base_score)
    
    def _calculate_cluster_risk(self, businesses: pd.DataFrame, 
                              cluster_type: str) -> float:
        """Calculate cluster risk score"""
        risk_score = 0
        
        # Industry concentration risk
        top_naics = businesses['naics_code'].str[:3].value_counts().iloc[0]
        concentration = top_naics / len(businesses)
        if concentration > 0.7:
            risk_score += 30
        elif concentration > 0.5:
            risk_score += 15
        
        # Age risk (too many new businesses)
        avg_age = 2025 - businesses['year_established'].mean()
        if avg_age < 3:
            risk_score += 20
        elif avg_age < 5:
            risk_score += 10
        
        # Size risk
        small_businesses = len(businesses[businesses['employees'] < 20])
        if small_businesses / len(businesses) > 0.7:
            risk_score += 15
        
        # Market volatility by cluster type
        volatility_scores = {
            "manufacturing": 20,
            "technology": 15,
            "logistics": 10,
            "biosciences": 5,
            "animal_health": 5
        }
        risk_score += volatility_scores.get(cluster_type, 10)
        
        return min(100, risk_score)
    
    def _apply_cluster_pareto_filter(self, clusters: List[Dict]) -> List[Dict]:
        """Apply Pareto filter at cluster level - keep clusters within threshold of top"""
        if not clusters:
            return clusters
        
        # Sort by cluster score
        clusters.sort(key=lambda x: x['cluster_score'], reverse=True)
        
        top_score = clusters[0]['cluster_score']
        threshold = top_score * (1 - self.pareto_threshold)
        
        filtered = [c for c in clusters if c['cluster_score'] >= threshold]
        
        logger.info(f"Cluster Pareto filter: {len(clusters)} -> {len(filtered)} clusters")
        logger.info(f"Top cluster: {clusters[0]['name']} (score: {top_score:.2f})")
        logger.info(f"Threshold: {threshold:.2f}")
        
        return filtered
    
    def _evaluate_cluster_synergies(self, clusters: List[Dict]):
        """Evaluate synergies between formed clusters"""
        for i, cluster in enumerate(clusters):
            synergy_scores = {}
            
            for j, other_cluster in enumerate(clusters):
                if i == j:
                    continue
                
                if other_cluster['type'] in cluster['synergy_clusters']:
                    # Calculate actual synergy score based on complementarity
                    synergy = self._calculate_cluster_pair_synergy(cluster, other_cluster)
                    synergy_scores[other_cluster['type']] = synergy
            
            cluster['realized_synergies'] = synergy_scores
            cluster['total_synergy_score'] = sum(synergy_scores.values())
    
    def _calculate_cluster_pair_synergy(self, cluster1: Dict, cluster2: Dict) -> float:
        """Calculate synergy between two clusters"""
        synergy = 0
        
        # Supply chain linkages
        supply_chain_pairs = [
            ("manufacturing", "logistics"),
            ("biosciences", "animal_health"),
            ("technology", "manufacturing")
        ]
        
        pair = tuple(sorted([cluster1['type'], cluster2['type']]))
        if pair in supply_chain_pairs:
            synergy += 30
        
        # Geographic proximity
        cluster1_counties = {b['county'] for b in cluster1['businesses']}
        cluster2_counties = {b['county'] for b in cluster2['businesses']}
        
        overlap = len(cluster1_counties & cluster2_counties)
        if overlap > 0:
            synergy += overlap * 10
        
        # Scale complementarity
        size_diff = abs(cluster1['business_count'] - cluster2['business_count'])
        if 10 <= size_diff <= 30:
            synergy += 15  # Good mix of large and small
        
        return min(100, synergy)