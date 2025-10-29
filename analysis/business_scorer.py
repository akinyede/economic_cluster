"""Business scoring module for KC Cluster Prediction Tool"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models import Business

logger = logging.getLogger(__name__)

class BusinessScorer:
    """Evaluates individual businesses for cluster potential"""
    
    def __init__(self):
        self.config = Config()
        self.scaler = StandardScaler()
        self.market_growth_rates = {
            "484": 0.06,  # Trucking - 6% growth
            "493": 0.08,  # Warehousing - 8% growth (e-commerce driven)
            "3254": 0.07,  # Pharmaceutical - 7% growth
            "5415": 0.09,  # Computer systems design - 9% growth
            "332": 0.04,  # Fabricated metal - 4% growth
            "3253": 0.05,  # Pesticide/agricultural chemical - 5% growth
        }
        # Use configurable scoring weights
        self.scoring_weights = self.config.BUSINESS_SCORING_WEIGHTS
        
    def calculate_innovation_score(self, business: Dict) -> float:
        """Calculate innovation score based on patents, SBIR awards, and R&D"""
        score = 0.0
        import numpy as _np

        # Paper: diminishing returns for patents via log(1+x)
        # Allocate up to 40 points to patents component
        patent_count = max(0, int(business.get("patent_count", 0)))
        # Normalize log1p against a reasonable cap (e.g., 50 patents)
        patents_norm = 0.0
        if patent_count > 0:
            patents_norm = min(1.0, _np.log1p(patent_count) / _np.log1p(50))
        patents_points = 40.0 * patents_norm
        score += patents_points

        # SBIR/STTR awards (0-40 points), diminishing returns
        sbir_awards = max(0, int(business.get("sbir_awards", 0)))
        sbir_norm = 0.0
        if sbir_awards > 0:
            sbir_norm = min(1.0, _np.log1p(sbir_awards) / _np.log1p(10))
        sbir_points = 40.0 * sbir_norm
        score += sbir_points

        # R&D intensity / technical workforce proxy (0-20 points)
        naics = (business.get("naics_code", "") or "").strip()
        rd_points = 0.0
        if naics.startswith("5417"):  # Scientific R&D services
            rd_points = 20.0
        elif naics.startswith("3254"):  # Pharmaceutical / biotech
            rd_points = 15.0
        elif naics.startswith("5415") or naics.startswith("5112"):  # Computer systems / software
            rd_points = 10.0
        score += rd_points

        # University partnerships (small bonus, capped by 100)
        if business.get("university_partner"):
            score += 5.0

        return float(min(100.0, max(0.0, score)))
    
    def calculate_market_potential_score(self, business: Dict) -> float:
        """Calculate market potential based on industry growth and demand"""
        score = 0.0
        import numpy as _np

        naics = (business.get("naics_code", "") or "").strip()
        naics_prefix = naics[:3] if len(naics) >= 3 else naics

        # Industry growth (0-40): normalized by max configured growth
        growth_rate = float(self.market_growth_rates.get(naics_prefix, 0.05))
        max_growth = max(self.market_growth_rates.values()) if self.market_growth_rates else 0.10
        growth_points = 40.0 * (growth_rate / max_growth) if max_growth > 0 else 20.0
        score += growth_points

        # Business scale (0-30): log revenue and employees for diminishing returns
        employees = max(1, int(business.get("employees", 10)))
        revenue = float(business.get("revenue_estimate", 0.0))

        # Normalize log revenue to ~[0,1] with 10B cap
        revenue_points = 0.0
        if revenue > 0:
            revenue_points = 15.0 * min(1.0, _np.log1p(revenue) / _np.log1p(10_000_000_000))

        employee_points = 15.0 * min(1.0, _np.log1p(employees) / _np.log1p(100_000))
        score += (revenue_points + employee_points)

        # Momentum (0-30): favor younger in high-growth, but capped
        year_established = int(business.get("year_established", 2020))
        business_age = max(0, 2025 - year_established)
        # Scale: new (<=5y): 1.0; 6-10y: 0.67; older: 0.33
        if business_age <= 5:
            age_factor = 1.0
        elif business_age <= 10:
            age_factor = 2.0 / 3.0
        else:
            age_factor = 1.0 / 3.0
        momentum_points = min(30.0, 30.0 * age_factor * (growth_rate / max(0.10, max_growth)))
        score += momentum_points

        return float(min(100.0, max(0.0, score)))
    
    def calculate_competition_score(self, business: Dict, 
                                   market_data: Optional[Dict] = None) -> float:
        """Calculate competitive position score based on relative market position"""
        score = 0.0
        
        naics = business.get("naics_code", "")
        naics_prefix = naics[:3] if len(naics) >= 3 else naics
        employees = business.get("employees", 10)
        revenue = business.get("revenue_estimate", 0)
        
        # Market position (0-40 points)
        # Based on relative size within industry
        if market_data and 'industry_stats' in market_data:
            industry_stats = market_data['industry_stats'].get(naics_prefix, {})
            median_employees = industry_stats.get('median_employees', 20)
            median_revenue = industry_stats.get('median_revenue', 500_000)
            
            # Employee position score
            employee_ratio = employees / median_employees if median_employees > 0 else 1
            if employee_ratio >= 5:  # 5x median
                score += 20
            elif employee_ratio >= 2:  # 2x median
                score += 15
            elif employee_ratio >= 1:  # At or above median
                score += 10
            else:
                score += 5 * employee_ratio  # Below median
            
            # Revenue position score
            revenue_ratio = revenue / median_revenue if median_revenue > 0 else 1
            if revenue_ratio >= 5:
                score += 20
            elif revenue_ratio >= 2:
                score += 15
            elif revenue_ratio >= 1:
                score += 10
            else:
                score += 5 * revenue_ratio
        else:
            # Fallback to absolute metrics if no market data
            if employees >= 100:
                score += 20
            elif employees >= 50:
                score += 15
            elif employees >= 20:
                score += 10
            else:
                score += 5
            
            if revenue >= 10_000_000:
                score += 20
            elif revenue >= 1_000_000:
                score += 15
            elif revenue >= 500_000:
                score += 10
            else:
                score += 5
        
        # Innovation differentiation (0-30 points)
        patent_count = business.get("patent_count", 0)
        sbir_awards = business.get("sbir_awards", 0)
        
        innovation_score = 0
        if patent_count > 5:
            innovation_score += 20
        elif patent_count > 0:
            innovation_score += 10 + (patent_count * 2)
        
        if sbir_awards > 0:
            innovation_score += min(10, sbir_awards * 5)
        
        score += min(30, innovation_score)
        
        # Market saturation factor (0-30 points)
        # Less saturated markets score higher
        if market_data:
            industry_count = market_data.get(f"count_{naics_prefix}", 100)
            if industry_count < 50:
                score += 30  # Low competition
            elif industry_count < 100:
                score += 20  # Moderate competition
            elif industry_count < 200:
                score += 10  # High competition
            else:
                score += 5   # Very high competition
        
        # Market share estimation (simplified)
        if market_data:
            industry_businesses = market_data.get(f"count_{naics[:3]}", 100)
            if industry_businesses < 50:
                score += 10  # Less saturated market
            elif industry_businesses > 200:
                score -= 10  # Highly competitive
                
            # Adjust for real-time market conditions if available
            if 'market_scores' in market_data:
                # Map NAICS to cluster type
                cluster_type = self._get_cluster_type(naics)
                if cluster_type and cluster_type in market_data['market_scores']:
                    market_score = market_data['market_scores'][cluster_type]
                    # Adjust competition score based on market favorability
                    score += (market_score - 0.5) * 20  # +/- 10 points
                    
            # Adjust for geopolitical risks if available
            if 'geopolitical_risks' in market_data:
                cluster_type = self._get_cluster_type(naics)
                if cluster_type and cluster_type in market_data['geopolitical_risks']:
                    geo_risk = market_data['geopolitical_risks'][cluster_type]
                    # Higher risk reduces competitive advantage
                    if geo_risk > 0.7:
                        score -= 15  # High risk penalty
                    elif geo_risk > 0.5:
                        score -= 8   # Moderate risk penalty
                    elif geo_risk < 0.3:
                        score += 5   # Low risk bonus
        
        return min(100, max(0, score))
    
    def _calculate_context_aware_threshold(self, df: pd.DataFrame, 
                                          is_quick_mode: bool = False,
                                          market_data: Optional[Dict] = None) -> float:
        """Calculate context-aware dynamic Pareto threshold
        
        This method ensures enough businesses pass for viable cluster formation
        while maintaining quality standards. It considers:
        1. Analysis mode (Quick vs Full)
        2. Economic targets
        3. Score distribution
        4. Minimum cluster requirements
        
        Returns:
            float: Dynamic threshold between 0 and 1 (e.g., 0.3 means include top 70%)
        """
        scores = df['composite_score'].values
        total_businesses = len(df)
        
        # Mode-specific base thresholds
        if is_quick_mode:
            # Quick mode: Be more inclusive since we have fewer businesses
            base_threshold = 0.30  # Include top 70%
            min_businesses_needed = 100  # Need at least 100 for 3-5 clusters
            logger.info(f"Quick mode: Starting with base threshold {base_threshold:.2f}")
        else:
            # Full mode: Can be more selective but still need critical mass
            base_threshold = 0.45  # Include top 55%
            min_businesses_needed = 500  # Need at least 500 for 5-8 clusters
            logger.info(f"Full mode: Starting with base threshold {base_threshold:.2f}")
        
        # Economic target-based adjustment
        if market_data and 'economic_targets' in market_data:
            gdp_target = market_data['economic_targets'].get('gdp_growth', 2.87e9)
            job_target = market_data['economic_targets'].get('direct_jobs', 1000)
            
            # Estimate requirements based on average business metrics
            avg_revenue = df['revenue_estimate'].mean() if 'revenue_estimate' in df else 1e6
            avg_employees = df['employees'].mean() if 'employees' in df else 15
            
            # Use realistic multipliers
            businesses_for_gdp = gdp_target / (avg_revenue * 2.3)  # 2.3x multiplier
            businesses_for_jobs = job_target / (avg_employees * 1.6)  # 1.6x multiplier
            
            target_businesses = max(businesses_for_gdp, businesses_for_jobs) * 1.5  # 50% buffer
            target_percentage = target_businesses / total_businesses
            
            logger.info(f"Economic targets suggest need for ~{int(target_businesses)} businesses ({target_percentage*100:.1f}% of total)")
            
            # Adjust threshold if we need more businesses
            if target_percentage > (1 - base_threshold):
                economic_threshold = max(0.20, 1 - target_percentage * 1.2)  # Don't go below 20%
                logger.info(f"Adjusting threshold for economic targets: {base_threshold:.2f} -> {economic_threshold:.2f}")
                base_threshold = economic_threshold
        
        # Score distribution analysis
        percentiles = np.percentile(scores, [99, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1])
        
        # Find natural breaks in score distribution
        score_gaps = []
        for i in range(len(percentiles) - 1):
            gap = percentiles[i] - percentiles[i+1]
            percentile_level = [99, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1][i]
            score_gaps.append((gap, percentile_level))
        
        # Find largest gap (natural break)
        score_gaps.sort(reverse=True)
        largest_gap_percentile = score_gaps[0][1]
        
        logger.info(f"Score distribution shows natural break at {largest_gap_percentile}th percentile")
        logger.info(f"  Top 1%: {percentiles[0]:.1f}")
        logger.info(f"  Top 5%: {percentiles[1]:.1f}")
        logger.info(f"  Top 10%: {percentiles[2]:.1f}")
        logger.info(f"  Top 20%: {percentiles[3]:.1f}")
        
        # Consider natural break but don't be too restrictive
        natural_threshold = (100 - largest_gap_percentile) / 100
        if natural_threshold < base_threshold:
            # Natural break is more selective, average them
            distribution_threshold = (base_threshold + natural_threshold) / 2
        else:
            # Natural break is less selective, use base
            distribution_threshold = base_threshold
        
        # Ensure minimum viable businesses
        businesses_at_threshold = int(total_businesses * (1 - distribution_threshold))
        
        if businesses_at_threshold < min_businesses_needed:
            # Need to be more inclusive
            required_percentage = min_businesses_needed / total_businesses
            if required_percentage < 0.95:  # Don't include more than 95%
                final_threshold = 1 - required_percentage * 1.1  # 10% buffer
                logger.info(f"Adjusting for minimum viability: {distribution_threshold:.2f} -> {final_threshold:.2f}")
            else:
                final_threshold = 0.05  # Include top 95% max
                logger.warning(f"Not enough businesses for ideal clustering. Including top 95%")
        else:
            final_threshold = distribution_threshold
        
        # Sanity checks
        final_threshold = max(0.05, min(0.70, final_threshold))  # Between 5% and 70%
        
        # Log decision summary
        expected_businesses = int(total_businesses * (1 - final_threshold))
        logger.info(f"Dynamic threshold decision summary:")
        logger.info(f"  Mode: {'Quick' if is_quick_mode else 'Full'}")
        logger.info(f"  Total businesses: {total_businesses}")
        logger.info(f"  Base threshold: {base_threshold:.2f}")
        logger.info(f"  Natural break threshold: {natural_threshold:.2f}")
        logger.info(f"  Final threshold: {final_threshold:.2f}")
        logger.info(f"  Expected businesses passing: {expected_businesses}")
        logger.info(f"  Minimum needed: {min_businesses_needed}")
        
        return final_threshold
    
    def _get_cluster_type(self, naics_code: str) -> Optional[str]:
        """Map NAICS code to cluster type"""
        naics_prefix = naics_code[:3] if len(naics_code) >= 3 else naics_code
        
        # Based on config.py CLUSTER_NAICS_CODES
        cluster_mapping = {
            "484": "logistics", "488": "logistics", "492": "logistics", "493": "logistics",
            "325": "biosciences", "339": "biosciences", "621": "biosciences",
            "511": "technology", "518": "technology", "519": "technology", "541": "technology",
            "311": "manufacturing", "312": "manufacturing", "332": "manufacturing", 
            "333": "manufacturing", "336": "manufacturing"
        }
        
        # Special handling for 541 subcategories
        if naics_code.startswith("5417"):  # Scientific R&D
            return "biosciences"
        
        # Check specific prefixes
        if naics_prefix in cluster_mapping:
            return cluster_mapping[naics_prefix]
            
        # Check broader technology classification
        if naics_code.startswith("5415"):  # Computer services
            return "technology"
        elif naics_code.startswith("5417"):  # Scientific R&D
            return "biosciences"
            
        return None
    
    def calculate_resource_fit_score(self, business: Dict, 
                                   infrastructure: Dict) -> float:
        """Calculate how well business aligns with regional resources"""
        score = 0.0
        naics = business.get("naics_code", "")
        naics_prefix = naics[:3] if len(naics) >= 3 else naics
        
        # Industry-specific infrastructure needs (0-40 points)
        infra_score = 0
        
        # Define infrastructure requirements by industry
        industry_needs = {
            "484": ["rail_capacity", "highway_access"],  # Trucking
            "493": ["rail_capacity", "warehouse_space"],  # Warehousing
            "325": ["university_research", "lab_space"],  # Chemicals/Pharma
            "541": ["broadband_speed", "tech_workforce"],  # Professional/Tech
            "333": ["skilled_manufacturing", "industrial_power"],  # Machinery
            "511": ["broadband_speed", "tech_workforce"]  # Software
        }
        
        needs = industry_needs.get(naics_prefix, [])
        if needs:
            met_needs = 0
            for need in needs:
                if infrastructure.get(need, 0) > 0:
                    met_needs += 1
            infra_score = (met_needs / len(needs)) * 40 if needs else 20
        else:
            # Generic infrastructure score
            infra_score = 20
        
        score += infra_score
        
        # Workforce fit (0-30 points)
        employees = business.get("employees", 10)
        workforce_available = infrastructure.get("workforce_availability", 10000)
        workforce_skilled = infrastructure.get("skilled_workforce_pct", 0.30)
        
        # Scale workforce needs
        if employees > 100 and workforce_available > 50000:
            workforce_score = 30
        elif employees > 50 and workforce_available > 20000:
            workforce_score = 25
        elif workforce_available > 10000:
            workforce_score = 20
        else:
            workforce_score = 10
        
        # Adjust for skill match
        if naics_prefix in ["541", "511", "325"] and workforce_skilled > 0.40:
            workforce_score *= 1.2
        
        score += min(30, workforce_score)
        
        # Cost competitiveness (0-30 points)
        # Based on regional cost advantages
        cost_index = infrastructure.get("cost_index", 100)  # 100 = national average
        if cost_index < 90:
            cost_score = 30  # Very competitive
        elif cost_index < 95:
            cost_score = 25  # Competitive
        elif cost_index < 100:
            cost_score = 20  # Slightly competitive
        elif cost_index < 105:
            cost_score = 15  # Average
        else:
            cost_score = 10  # Above average
        
        score += cost_score
        
        return min(100, score)
    
    def apply_regional_strategic_multiplier(self, score: float, business: Dict) -> float:
        """
        Apply Kansas City regional strategic advantages based on NAICS
        From Akinyede & Caruso (2025) paper
        """
        naics = (business.get("naics_code", "") or "").strip()
        
        multiplier = 1.0
        
        # Check for strategic sectors using Config multipliers
        if naics.startswith("3254"):  # Pharmaceutical/Biotech
            multiplier = 1.25  # biosciences
        elif naics.startswith("112"):  # Animal production
            multiplier = 1.30  # animal_health (highest in KC)
        elif naics.startswith("54"):  # Professional/Tech services
            multiplier = 1.20  # technology
        elif naics.startswith("484") or naics.startswith("493"):  # Transport/Warehouse
            multiplier = 1.15  # logistics
        elif naics.startswith("52"):  # Finance/Insurance
            multiplier = 1.15  # fintech
        elif naics.startswith("31") or naics.startswith("32") or naics.startswith("33"):  # Manufacturing
            multiplier = 1.10  # manufacturing
        
        return score * multiplier
    
    def calculate_composite_score(self, business: Dict, 
                                market_data: Optional[Dict] = None,
                                infrastructure: Optional[Dict] = None) -> Tuple[float, Dict]:
        """Calculate weighted composite score for business"""
        
        # Calculate individual scores
        innovation = self.calculate_innovation_score(business)
        market_potential = self.calculate_market_potential_score(business)
        competition = self.calculate_competition_score(business, market_data)
        
        # Calculate resource fit if infrastructure data available
        resource_fit = 0
        if infrastructure:
            resource_fit = self.calculate_resource_fit_score(business, infrastructure)

            # Fixed share for resource-fit; re-normalize remaining weights
            resource_weight = 0.15
            base_weights = {
                "innovation": self.scoring_weights.get("innovation", 0.30),
                "market_potential": self.scoring_weights.get("market_potential", 0.40),
                "competition": self.scoring_weights.get("competition", 0.30),
            }

            # Normalise so that innovation+market_potential+competition = 1.0
            base_total = sum(base_weights.values()) or 1.0
            norm_weights = {k: v / base_total for k, v in base_weights.items()}

            # Scale the normalised weights so that they occupy (1-resource_weight) share
            norm_weights = {k: v * (1 - resource_weight) for k, v in norm_weights.items()}

            composite = (
                innovation * norm_weights["innovation"] +
                market_potential * norm_weights["market_potential"] +
                competition * norm_weights["competition"] +
                resource_fit * resource_weight
            )
        else:
            # Use configurable weights
            composite = (
                innovation * self.scoring_weights.get("innovation", 0.30) +
                market_potential * self.scoring_weights.get("market_potential", 0.40) +
                competition * self.scoring_weights.get("competition", 0.30)
            )
        
        # Apply regional strategic multiplier (from Akinyede & Caruso 2025)
        composite_with_multiplier = self.apply_regional_strategic_multiplier(composite, business)
        
        scores = {
            "innovation_score": innovation,
            "market_potential_score": market_potential,
            "competition_score": competition,
            "resource_fit_score": resource_fit,
            "composite_score": composite,
            "composite_score_with_multiplier": composite_with_multiplier
        }
        
        return composite_with_multiplier, scores
    
    def get_industry_thresholds(self, naics_code: str) -> Dict[str, float]:
        """Get industry-specific thresholds based on typical business characteristics"""
        # Define industry profiles based on real-world characteristics
        industry_profiles = {
            # Transportation often has owner-operators and small fleets
            "484": {"min_employees": 1, "min_revenue": 50000},    # Trucking
            "488": {"min_employees": 5, "min_revenue": 500000},   # Rail (needs more capital)
            "492": {"min_employees": 1, "min_revenue": 35000},    # Couriers/messengers
            "493": {"min_employees": 3, "min_revenue": 100000},   # Warehousing
            
            # Services can be very small
            "541": {"min_employees": 1, "min_revenue": 50000},    # Professional services
            "611": {"min_employees": 1, "min_revenue": 25000},    # Education
            "624": {"min_employees": 1, "min_revenue": 30000},    # Social assistance
            "812": {"min_employees": 1, "min_revenue": 25000},    # Personal services
            
            # Manufacturing typically needs more scale
            "311": {"min_employees": 5, "min_revenue": 200000},   # Food manufacturing
            "332": {"min_employees": 5, "min_revenue": 150000},   # Fabricated metal
            "333": {"min_employees": 8, "min_revenue": 300000},   # Machinery
            "334": {"min_employees": 10, "min_revenue": 500000},  # Computer/electronic
            
            # Biosciences/Healthcare
            "325": {"min_employees": 10, "min_revenue": 500000},  # Chemical/pharma
            "339": {"min_employees": 5, "min_revenue": 200000},   # Medical equipment
            "621": {"min_employees": 3, "min_revenue": 150000},   # Healthcare
            
            # Technology
            "511": {"min_employees": 2, "min_revenue": 75000},    # Software publishing
            "518": {"min_employees": 3, "min_revenue": 100000},   # Data processing
            "5415": {"min_employees": 1, "min_revenue": 50000},   # Computer systems design
        }
        
        # Get first 3 digits of NAICS for lookup
        naics_3 = str(naics_code)[:3] if naics_code else ""
        
        # Return industry-specific or default thresholds
        return industry_profiles.get(naics_3, {
            "min_employees": self.config.MIN_EMPLOYEES,
            "min_revenue": self.config.MIN_REVENUE
        })

    def apply_business_filters(self, businesses: List[Dict]) -> Tuple[List[Dict], Dict]:
        """Apply business filtering criteria with industry-specific thresholds"""
        filtered = businesses.copy()  # Start with all businesses
        current_year = 2025
        initial_count = len(businesses)
        
        # Log filter criteria
        logger.info(f"Starting with {initial_count} businesses")
        logger.info(f"Default filter criteria - Min employees: {self.config.MIN_EMPLOYEES}, Max employees: {self.config.MAX_EMPLOYEES}")
        logger.info(f"Default filter criteria - Min revenue: {self.config.MIN_REVENUE}")
        logger.info(f"Using industry-specific thresholds where applicable")
        logger.info(f"Filter criteria - Min age: {self.config.MIN_BUSINESS_AGE}")
        logger.info(f"Filter criteria - Geographic focus: {getattr(self.config, 'GEOGRAPHIC_FOCUS', 'both')}")
        
        filter_reasons = {
            'employees': 0,
            'revenue': 0,
            'age': 0,
            'naics': 0,
            'geography': 0,
            'patents': 0,
            'sbir': 0,
            'growth': 0
        }
        
        # Track filtering at each stage
        filter_stages = []
        
        # Stage 1: Employee count filter
        stage_count = len(filtered)
        new_filtered = []
        industry_filtered = {}
        
        for business in filtered:
            naics_code = business.get("naics_code", "")
            thresholds = self.get_industry_thresholds(naics_code)
            employees = business.get("employees", 0)
            
            if employees < thresholds["min_employees"] or employees > self.config.MAX_EMPLOYEES:
                filter_reasons['employees'] += 1
                naics_3 = str(naics_code)[:3]
                industry_filtered[naics_3] = industry_filtered.get(naics_3, 0) + 1
            else:
                new_filtered.append(business)
        
        removed = stage_count - len(new_filtered)
        filter_stages.append(('employees', removed, len(new_filtered)))
        if removed / initial_count > 0.9:
            logger.warning(f"Employee filter removed {removed} businesses ({removed/initial_count*100:.1f}% of total)")
        filtered = new_filtered
        
        # Stage 2: Revenue filter
        stage_count = len(filtered)
        new_filtered = []
        
        for business in filtered:
            naics_code = business.get("naics_code", "")
            thresholds = self.get_industry_thresholds(naics_code)
            revenue = business.get("revenue_estimate", 0)
            
            if revenue < thresholds["min_revenue"]:
                filter_reasons['revenue'] += 1
                if filter_reasons['revenue'] <= 5:
                    logger.debug(f"Business filtered by revenue: {business.get('name', 'Unknown')} (NAICS: {naics_code}) - Revenue: ${revenue:,.0f} < ${thresholds['min_revenue']:,.0f}")
            else:
                new_filtered.append(business)
        
        removed = stage_count - len(new_filtered)
        filter_stages.append(('revenue', removed, len(new_filtered)))
        if removed / initial_count > 0.9:
            logger.warning(f"Revenue filter removed {removed} businesses ({removed/initial_count*100:.1f}% of total)")
        filtered = new_filtered
        
        # Stage 3: Age filter
        stage_count = len(filtered)
        new_filtered = []
        
        for business in filtered:
            year_established = business.get("year_established", current_year)
            age = current_year - year_established
            
            if age < self.config.MIN_BUSINESS_AGE:
                filter_reasons['age'] += 1
            else:
                new_filtered.append(business)
        
        removed = stage_count - len(new_filtered)
        filter_stages.append(('age', removed, len(new_filtered)))
        if removed / initial_count > 0.9:
            logger.warning(f"Age filter removed {removed} businesses ({removed/initial_count*100:.1f}% of total)")
        filtered = new_filtered
        
        # Stage 4: NAICS exclusion filter
        stage_count = len(filtered)
        new_filtered = []
        
        for business in filtered:
            naics = business.get("naics_code", "")
            if any(naics.startswith(excluded) for excluded in self.config.EXCLUDED_NAICS):
                filter_reasons['naics'] += 1
            else:
                new_filtered.append(business)
        
        removed = stage_count - len(new_filtered)
        filter_stages.append(('naics', removed, len(new_filtered)))
        filtered = new_filtered
        
        # Stage 5: Geographic filter
        if hasattr(self.config, 'GEOGRAPHIC_FOCUS') and self.config.GEOGRAPHIC_FOCUS not in ["both", "all"]:
            stage_count = len(filtered)
            new_filtered = []
            
            for business in filtered:
                county = business.get("county", "").lower()
                passes_geo = False
                
                if self.config.GEOGRAPHIC_FOCUS == "urban":
                    urban_counties = ["jackson", "clay", "platte", "johnson", "wyandotte"]
                    passes_geo = any(uc in county for uc in urban_counties)
                elif self.config.GEOGRAPHIC_FOCUS == "suburban":
                    suburban_counties = ["cass", "leavenworth", "miami"]
                    passes_geo = any(sc in county for sc in suburban_counties)
                elif self.config.GEOGRAPHIC_FOCUS == "rural":
                    rural_counties = ["bates", "caldwell", "clinton", "lafayette", "ray", "linn"]
                    passes_geo = any(rc in county for rc in rural_counties)
                
                if passes_geo:
                    new_filtered.append(business)
                else:
                    filter_reasons['geography'] += 1
            
            removed = stage_count - len(new_filtered)
            filter_stages.append(('geography', removed, len(new_filtered)))
            if removed / initial_count > 0.9:
                logger.warning(f"Geographic filter removed {removed} businesses ({removed/initial_count*100:.1f}% of total)")
            filtered = new_filtered
        
        # Stage 6: Patent filter
        if hasattr(self.config, 'REQUIRE_PATENTS') and self.config.REQUIRE_PATENTS:
            stage_count = len(filtered)
            new_filtered = [b for b in filtered if b.get("patent_count", 0) > 0]
            removed = stage_count - len(new_filtered)
            filter_reasons['patents'] = removed
            filter_stages.append(('patents', removed, len(new_filtered)))
            if removed / initial_count > 0.9:
                logger.warning(f"Patent filter removed {removed} businesses ({removed/initial_count*100:.1f}% of total)")
            filtered = new_filtered
        
        # Stage 7: SBIR filter
        if hasattr(self.config, 'REQUIRE_SBIR') and self.config.REQUIRE_SBIR:
            stage_count = len(filtered)
            new_filtered = [b for b in filtered if b.get("sbir_awards", 0) > 0]
            removed = stage_count - len(new_filtered)
            filter_reasons['sbir'] = removed
            filter_stages.append(('sbir', removed, len(new_filtered)))
            if removed / initial_count > 0.9:
                logger.warning(f"SBIR filter removed {removed} businesses ({removed/initial_count*100:.1f}% of total)")
            filtered = new_filtered
        
        # Stage 8: Growth rate filter
        if hasattr(self.config, 'MIN_GROWTH_RATE') and self.config.MIN_GROWTH_RATE > 0:
            stage_count = len(filtered)
            new_filtered = []
            
            for business in filtered:
                if "growth_rate" in business:
                    if business.get("growth_rate", 0) >= self.config.MIN_GROWTH_RATE:
                        new_filtered.append(business)
                    else:
                        filter_reasons['growth'] += 1
                else:
                    new_filtered.append(business)  # No growth data, don't filter
            
            removed = stage_count - len(new_filtered)
            filter_stages.append(('growth', removed, len(new_filtered)))
            if removed / initial_count > 0.9:
                logger.warning(f"Growth rate filter removed {removed} businesses ({removed/initial_count*100:.1f}% of total)")
            filtered = new_filtered
        
        logger.info(f"Filtered businesses: {len(businesses)} -> {len(filtered)}")
        logger.info(f"Filter reasons: {filter_reasons}")
        
        # Log stage-by-stage filtering
        logger.info("Filter stages:")
        for stage_name, removed, remaining in filter_stages:
            if removed > 0:
                logger.info(f"  {stage_name}: removed {removed}, remaining {remaining}")
        
        # Log industry-specific filtering impact
        if industry_filtered:
            logger.info("Industry-specific filtering impact:")
            for naics, count in sorted(industry_filtered.items(), key=lambda x: x[1], reverse=True)[:10]:
                logger.info(f"  NAICS {naics}: {count} businesses filtered")
        
        # Add filter stages to the filter reasons for detailed reporting
        filter_reasons['stages'] = filter_stages
        
        return filtered, filter_reasons
    
    def _stratified_sample_top_n(self, businesses: List[Dict], 
                                 sample_size: int,
                                 market_data: Optional[Dict] = None,
                                 infrastructure: Optional[Dict] = None) -> List[Dict]:
        """Apply stratified sampling with top-N scoring
        
        Maintains geographic and industry distribution while selecting
        highest-potential businesses within each stratum.
        """
        import pandas as pd
        from collections import defaultdict
        
        # Quick score all businesses for stratified sampling
        logger.info("Quick-scoring all businesses for stratified sampling...")
        
        # Add quick scores to businesses
        for business in businesses:
            # Use simplified scoring for speed
            quick_score = self._calculate_quick_score(business)
            business['_quick_score'] = quick_score
        
        # Define strategic industries for KC with guaranteed minimum representation
        strategic_industries = {
            'logistics': {'codes': ['484', '488', '493', '423'], 'min_pct': 0.08},  # 8% min
            'biosciences': {'codes': ['325', '339', '621', '5417'], 'min_pct': 0.10},  # 10% min
            'technology': {'codes': ['511', '518', '519', '5415'], 'min_pct': 0.08},  # 8% min
            'manufacturing': {'codes': ['311', '312', '332', '333', '336'], 'min_pct': 0.06},  # 6% min
        }
        
        sampled_businesses = []
        used_indices = set()
        
        # First, ensure strategic industries have minimum representation
        for industry_name, industry_info in strategic_industries.items():
            min_count = int(sample_size * industry_info['min_pct'])
            industry_businesses = []
            
            # Find businesses in this strategic industry
            for business in businesses:
                naics = business.get('naics_code', '')
                if any(naics.startswith(code) for code in industry_info['codes']):
                    industry_businesses.append(business)
            
            if industry_businesses:
                # Sort by quick score and take top N
                industry_businesses.sort(key=lambda x: x['_quick_score'], reverse=True)
                sample_count = min(min_count, len(industry_businesses))
                
                for business in industry_businesses[:sample_count]:
                    if id(business) not in used_indices:
                        sampled_businesses.append(business)
                        used_indices.add(id(business))
                
                logger.info(f"  {industry_name}: sampled {sample_count} businesses")
        
        # Calculate remaining sample size
        remaining_sample_size = sample_size - len(sampled_businesses)
        
        if remaining_sample_size > 0:
            # Create strata based on county and industry (first 3 digits of NAICS)
            strata = defaultdict(list)
            for business in businesses:
                if id(business) not in used_indices:
                    county = business.get('county', 'Unknown')
                    naics_prefix = business.get('naics_code', '000')[:3]
                    stratum_key = f"{county}_{naics_prefix}"
                    strata[stratum_key].append(business)
            
            # Proportional allocation for remaining
            total_remaining = sum(len(s) for s in strata.values())
            
            for stratum_key, stratum_businesses in strata.items():
                if remaining_sample_size <= 0:
                    break
                    
                # Calculate this stratum's proportion
                stratum_proportion = len(stratum_businesses) / total_remaining if total_remaining > 0 else 0
                stratum_sample_size = max(1, int(remaining_sample_size * stratum_proportion))
                
                # Sort by quick score and take top N from this stratum
                stratum_businesses.sort(key=lambda x: x['_quick_score'], reverse=True)
                sampled = stratum_businesses[:stratum_sample_size]
                sampled_businesses.extend(sampled)
                remaining_sample_size -= len(sampled)
        
        # If we're over sample_size, trim to exact size (keeping highest scores)
        if len(sampled_businesses) > sample_size:
            sampled_businesses.sort(key=lambda x: x['_quick_score'], reverse=True)
            sampled_businesses = sampled_businesses[:sample_size]
        
        # Clean up temporary score
        for business in sampled_businesses:
            business.pop('_quick_score', None)
        
        # Log sampling statistics
        sampled_counties = defaultdict(int)
        sampled_industries = defaultdict(int)
        strategic_counts = {name: 0 for name in strategic_industries}
        
        for business in sampled_businesses:
            sampled_counties[business.get('county', 'Unknown')] += 1
            naics = business.get('naics_code', '000')
            sampled_industries[naics[:3]] += 1
            
            # Count strategic industries
            for industry_name, industry_info in strategic_industries.items():
                if any(naics.startswith(code) for code in industry_info['codes']):
                    strategic_counts[industry_name] += 1
        
        logger.info(f"Sampled {len(sampled_businesses)} businesses:")
        logger.info(f"  - Counties represented: {len(sampled_counties)}")
        logger.info(f"  - Industries represented: {len(sampled_industries)}")
        logger.info(f"  - Strategic industry representation:")
        for industry_name, count in strategic_counts.items():
            pct = count / len(sampled_businesses) * 100 if sampled_businesses else 0
            logger.info(f"    - {industry_name}: {count} ({pct:.1f}%)")
        
        return sampled_businesses
    
    def _calculate_quick_score(self, business: Dict) -> float:
        """Calculate a quick score for stratified sampling
        
        This is a simplified version of the full composite score,
        optimized for speed when processing large datasets.
        """
        score = 0.0
        
        # Revenue indicator (0-35 points) - normalized
        revenue = business.get('revenue_estimate', 0)
        if revenue > 10_000_000:  # $10M+
            score += 35
        elif revenue > 5_000_000:  # $5M+
            score += 30
        elif revenue > 1_000_000:  # $1M+
            score += 25
        elif revenue > 500_000:  # $500K+
            score += 20
        elif revenue > 100_000:  # $100K+
            score += 15
        else:
            score += 10
        
        # Employee count (0-25 points) - normalized
        employees = business.get('employees', 0)
        if employees > 100:
            score += 25
        elif employees > 50:
            score += 20
        elif employees > 20:
            score += 15
        elif employees > 10:
            score += 10
        else:
            score += 5
        
        # Industry growth potential (0-20 points) - data-driven
        naics = business.get('naics_code', '')[:3]
        # Use actual growth rates
        growth_rate = self.market_growth_rates.get(naics, 0.05)
        score += min(20, growth_rate * 200)  # Convert to 0-20 scale
        
        # Innovation indicators (0-20 points)
        innovation_score = 0
        if business.get('patent_count', 0) > 0:
            innovation_score += 10
        if business.get('sbir_awards', 0) > 0:
            innovation_score += 10
        score += innovation_score
        
        return score
    
    def rank_businesses(self, businesses: List[Dict], 
                       market_data: Optional[Dict] = None,
                       infrastructure: Optional[Dict] = None,
                       sample_size: Optional[int] = None,
                       progress_callback: Optional[Callable[[int, str], None]] = None) -> pd.DataFrame:
        """Rank businesses by composite score and apply Pareto filter
        
        Args:
            businesses: List of business dictionaries
            market_data: Market conditions data
            infrastructure: Infrastructure data
            sample_size: Optional limit on number of businesses to process (for quick mode)
        """
        
        # First, ensure all businesses have revenue estimates
        logger.info(f"Processing {len(businesses)} businesses for ranking")
        revenue_added = 0
        for business in businesses:
            if "revenue_estimate" not in business or business["revenue_estimate"] == 0:
                business["revenue_estimate"] = self.estimate_business_revenue(business)
                revenue_added += 1
        logger.info(f"Added revenue estimates to {revenue_added} businesses")
        
        # Apply business filters after revenue estimation
        filtered_businesses, filter_reasons = self.apply_business_filters(businesses)
        
        # Handle case where all businesses are filtered out
        if not filtered_businesses:
            logger.warning("All businesses filtered out! Returning empty DataFrame")
            # Create a DataFrame with filter reasons to pass back
            empty_df = pd.DataFrame()
            empty_df.attrs['filter_reasons'] = filter_reasons
            empty_df.attrs['total_businesses'] = len(businesses)
            return empty_df
        
        # Apply stratified sampling if sample_size is specified (Quick mode)
        if sample_size and len(filtered_businesses) > sample_size:
            logger.info(f"Applying stratified sampling: {len(filtered_businesses)} -> {sample_size}")
            filtered_businesses = self._stratified_sample_top_n(
                filtered_businesses, sample_size, market_data, infrastructure
            )
        
        # Process businesses in batches to manage memory
        batch_size = 1000
        scored_businesses = []
        
        for i in range(0, len(filtered_businesses), batch_size):
            batch = filtered_businesses[i:i + batch_size]
            batch_scored = []
            
            for business in batch:
                composite, scores = self.calculate_composite_score(
                    business, market_data, infrastructure
                )
                
                business_record = business.copy()
                business_record.update(scores)
                batch_scored.append(business_record)
            
            scored_businesses.extend(batch_scored)
            
            # Log and emit progress for large datasets
            if len(filtered_businesses) > batch_size:
                processed = min(i + batch_size, len(filtered_businesses))
                progress = min(100, int(processed / len(filtered_businesses) * 100))
                logger.info(f"Scored {processed}/{len(filtered_businesses)} businesses ({progress}%)")
                if progress_callback:
                    try:
                        progress_callback(progress, f"Scoring businesses... ({processed}/{len(filtered_businesses)})")
                    except Exception:
                        pass
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(scored_businesses)
        
        # Sort by composite score
        df = df.sort_values("composite_score", ascending=False)
        
        # Apply dynamic Pareto filter based on context
        if len(df) > 0:
            # Determine if we're in Quick or Full mode
            # Prefer explicit signal from market_data (set by main.py), fallback to size heuristic
            is_quick_mode = False
            try:
                if market_data and (market_data.get('quick_mode') or market_data.get('analysis_mode') == 'quick'):
                    is_quick_mode = True
                else:
                    is_quick_mode = len(df) < 1000
            except Exception:
                is_quick_mode = len(df) < 1000
            
            # Calculate dynamic threshold based on mode and targets
            dynamic_threshold = self._calculate_context_aware_threshold(
                df, 
                is_quick_mode=is_quick_mode,
                market_data=market_data
            )
            
            top_score = df.iloc[0]["composite_score"]
            threshold = top_score * (1 - dynamic_threshold)
            df["passes_threshold"] = df["composite_score"] >= threshold
            
            # Log threshold details
            passing_count = len(df[df["passes_threshold"]])
            mode_str = "Quick" if is_quick_mode else "Full"
            logger.info(f"{mode_str} Mode - Dynamic Pareto threshold: {threshold:.1f} (top score: {top_score:.1f}, dynamic threshold: {dynamic_threshold*100:.0f}%)")
            logger.info(f"Businesses passing threshold: {passing_count}/{len(df)} ({passing_count/len(df)*100:.1f}%)")
            logger.info(f"Original config threshold would have been: {self.config.PARETO_THRESHOLD*100:.0f}%")
        else:
            df["passes_threshold"] = False
        
        # Store filter summary in DataFrame attributes
        df.attrs['filter_reasons'] = filter_reasons
        df.attrs['total_businesses'] = len(businesses)
        df.attrs['filtered_count'] = len(filtered_businesses)
        df.attrs['sample_size'] = sample_size
        
        return df
    
    def generate_business_insights(self, df: pd.DataFrame) -> Dict:
        """Generate insights from scored businesses"""
        
        # Handle empty dataframe
        if df.empty:
            return {
                "total_businesses": 0,
                "passing_threshold": 0,
                "avg_composite_score": 0,
                "top_industries": [],
                "innovation_leaders": [],
                "high_growth_potential": []
            }
        
        insights = {
            "total_businesses": len(df),
            "passing_threshold": len(df[df["passes_threshold"]]) if "passes_threshold" in df.columns else 0,
            "avg_composite_score": df["composite_score"].mean() if "composite_score" in df.columns else 0,
            "top_industries": [],
            "innovation_leaders": [],
            "high_growth_potential": []
        }
        
        # Top industries by average score
        if "composite_score" in df.columns and "naics_code" in df.columns:
            industry_scores = df.groupby("naics_code")["composite_score"].agg(["mean", "count"])
            industry_scores = industry_scores[industry_scores["count"] >= 3]
            industry_scores = industry_scores.sort_values("mean", ascending=False)
            insights["top_industries"] = industry_scores.head(5).to_dict("index")
        
        # Innovation leaders
        if "innovation_score" in df.columns:
            innovation_leaders = df.nlargest(5, "innovation_score")[
                ["name", "innovation_score", "patent_count", "sbir_awards"]
            ]
            insights["innovation_leaders"] = innovation_leaders.to_dict("records")
        
        # High growth potential
        if "market_potential_score" in df.columns:
            growth_potential = df.nlargest(5, "market_potential_score")[
                ["name", "market_potential_score", "naics_code", "employees"]
            ]
            insights["high_growth_potential"] = growth_potential.to_dict("records")
        
        return insights
    
    def estimate_business_revenue(self, business: Dict) -> float:
        """Estimate annual revenue based on employees and industry.

        Robust to missing or non-string NAICS codes and non-numeric employees.
        """
        # Employees: coerce to int with a conservative default
        employees_raw = business.get("employees", 10)
        try:
            employees = int(employees_raw) if employees_raw is not None else 10
        except (TypeError, ValueError):
            employees = 10

        # NAICS: normalize to a digit string (e.g., 541511 -> "541511")
        naics_raw = business.get("naics_code", "")
        naics_str = "".join(ch for ch in str(naics_raw) if ch.isdigit())
        
        # Industry-specific revenue per employee (based on research)
        revenue_per_employee = {
            "484": 216528,   # Trucking
            "493": 189000,   # Warehousing
            "3254": 450000,  # Pharmaceutical
            "5415": 165000,  # Computer systems
            "332": 195000,   # Fabricated metal
            "3253": 380000,  # Agricultural chemical
        }
        
        # Get industry average or use general average
        # Prefer 4-digit mapping if present, else fall back to 3-digit
        if len(naics_str) >= 4 and naics_str[:4] in revenue_per_employee:
            key = naics_str[:4]
        else:
            key = naics_str[:3] if len(naics_str) >= 3 else naics_str
        rpe = revenue_per_employee.get(key, 180000)
        
        # Adjust for business size (economies of scale)
        if employees > 100:
            rpe *= 1.1
        elif employees < 20:
            rpe *= 0.9
        
        return employees * rpe
