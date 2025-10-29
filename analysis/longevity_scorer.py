"""Longevity scoring for cluster resilience in changing environments"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
import requests

logger = logging.getLogger(__name__)

class LongevityScorer:
    """Score cluster longevity considering political, market, and environmental changes"""
    
    def __init__(self):
        # Risk factors and weights
        self.risk_factors = {
            'political': {
                'weight': 0.25,
                'components': [
                    'trade_policy_exposure',
                    'regulatory_change_risk',
                    'government_dependency',
                    'political_stability'
                ]
            },
            'market': {
                'weight': 0.30,
                'components': [
                    'demand_volatility',
                    'commodity_price_risk',
                    'technology_disruption',
                    'competitive_intensity'
                ]
            },
            'financial': {
                'weight': 0.20,
                'components': [
                    'capital_intensity',
                    'credit_availability',
                    'interest_rate_sensitivity',
                    'currency_risk'
                ]
            },
            'operational': {
                'weight': 0.15,
                'components': [
                    'supply_chain_risk',
                    'workforce_availability',
                    'infrastructure_dependency',
                    'climate_vulnerability'
                ]
            },
            'strategic': {
                'weight': 0.10,
                'components': [
                    'innovation_capacity',
                    'market_diversification',
                    'strategic_partnerships',
                    'adaptability'
                ]
            }
        }
        
        # Cluster-specific risk profiles
        self.cluster_risk_profiles = {
            'logistics': {
                'trade_policy_exposure': 0.8,  # High - international trade dependent
                'technology_disruption': 0.7,   # High - autonomous vehicles
                'workforce_availability': 0.6,  # Medium - driver shortage
                'infrastructure_dependency': 0.9,  # Very high - rail/road critical
                'commodity_price_risk': 0.7,    # High - fuel prices
                'climate_vulnerability': 0.4     # Medium - weather disruptions
            },
            'biosciences': {
                'regulatory_change_risk': 0.9,  # Very high - FDA regulations
                'government_dependency': 0.6,    # High - research grants
                'capital_intensity': 0.9,        # Very high - R&D costs
                'innovation_capacity': 0.9,      # Very high - required
                'technology_disruption': 0.3,    # Low - creates disruption
                'market_diversification': 0.7    # High - global markets
            },
            'technology': {
                'technology_disruption': 0.5,    # Medium - both risk and opportunity
                'competitive_intensity': 0.9,    # Very high - rapid change
                'innovation_capacity': 0.9,      # Very high - required
                'workforce_availability': 0.7,   # High - talent competition
                'adaptability': 0.8,            # High - must pivot quickly
                'capital_intensity': 0.6         # Medium - varies by segment
            },
            'manufacturing': {
                'trade_policy_exposure': 0.7,   # High - tariffs impact
                'commodity_price_risk': 0.8,    # High - raw materials
                'infrastructure_dependency': 0.7, # High - utilities/transport
                'workforce_availability': 0.5,   # Medium - skilled trades
                'technology_disruption': 0.6,    # Medium - automation
                'regulatory_change_risk': 0.5    # Medium - environmental regs
            },
            'animal_health': {
                'regulatory_change_risk': 0.8,   # High - USDA/FDA oversight
                'market_diversification': 0.8,   # High - global markets
                'innovation_capacity': 0.7,      # High - new products needed
                'climate_vulnerability': 0.6,    # Medium - disease patterns
                'trade_policy_exposure': 0.6,    # Medium - export markets
                'strategic_partnerships': 0.8    # High - pharma ties
            }
        }
        
    def calculate_longevity_score(self, cluster: Dict, 
                                 market_conditions: Optional[Dict] = None,
                                 political_climate: Optional[Dict] = None) -> Dict:
        """
        Calculate comprehensive longevity score for a cluster
        
        Returns score 0-100 where:
        - 80-100: Very resilient, likely to thrive for 10+ years
        - 60-79: Resilient, good for 5-10 years
        - 40-59: Moderate risk, 3-5 year outlook
        - 20-39: High risk, 1-3 year outlook
        - 0-19: Very high risk, immediate concerns
        """
        cluster_type = cluster.get('type', 'mixed')
        
        # Get base risk profile
        risk_profile = self.cluster_risk_profiles.get(
            cluster_type, 
            self._get_default_risk_profile()
        )
        
        # Calculate risk scores by category
        risk_scores = {}
        
        for category, config in self.risk_factors.items():
            score = self._calculate_category_score(
                category, 
                config['components'],
                risk_profile,
                cluster,
                market_conditions,
                political_climate
            )
            risk_scores[category] = score
            
        # Calculate weighted total (inverse for longevity)
        total_risk = sum(
            risk_scores[cat] * self.risk_factors[cat]['weight'] 
            for cat in risk_scores
        )
        
        # Convert risk to longevity (0-100 scale)
        longevity_score = (1 - total_risk) * 100
        
        # Calculate time horizon
        time_horizon = self._estimate_time_horizon(longevity_score)
        
        # Generate detailed assessment
        assessment = self._generate_assessment(
            cluster, 
            risk_scores, 
            longevity_score,
            time_horizon
        )
        
        return {
            'score': round(longevity_score, 1),
            'grade': self._get_grade(longevity_score),
            'time_horizon': time_horizon,
            'risk_breakdown': risk_scores,
            'key_risks': self._identify_key_risks(risk_scores, risk_profile),
            'mitigation_strategies': self._suggest_mitigations(cluster_type, risk_scores),
            'assessment': assessment
        }
    
    def _calculate_category_score(self, category: str, components: List[str],
                                 risk_profile: Dict, cluster: Dict,
                                 market_conditions: Optional[Dict],
                                 political_climate: Optional[Dict]) -> float:
        """Calculate risk score for a category"""
        scores = []
        
        for component in components:
            # Base score from profile
            base_score = risk_profile.get(component, 0.5)
            
            # Adjust based on current conditions
            if category == 'political' and political_climate:
                base_score = self._adjust_political_risk(component, base_score, political_climate)
            elif category == 'market' and market_conditions:
                base_score = self._adjust_market_risk(component, base_score, market_conditions)
                
            # Adjust based on cluster characteristics
            base_score = self._adjust_for_cluster(component, base_score, cluster)
            
            scores.append(base_score)
            
        return np.mean(scores) if scores else 0.5
    
    def _adjust_political_risk(self, component: str, base: float, climate: Dict) -> float:
        """Adjust risk based on current political climate"""
        if component == 'trade_policy_exposure':
            # Check for trade tensions
            if climate.get('trade_tensions', False):
                return min(base * 1.3, 1.0)
        elif component == 'regulatory_change_risk':
            # Check for regulatory activity
            if climate.get('regulatory_activity', 'normal') == 'high':
                return min(base * 1.2, 1.0)
        elif component == 'political_stability':
            # Check election cycle
            if climate.get('election_year', False):
                return min(base * 1.1, 1.0)
                
        return base
    
    def _adjust_market_risk(self, component: str, base: float, conditions: Dict) -> float:
        """Adjust risk based on current market conditions"""
        if component == 'demand_volatility':
            # Check market volatility
            vix = conditions.get('volatility_index', 20)
            if vix > 30:
                return min(base * 1.2, 1.0)
        elif component == 'commodity_price_risk':
            # Check commodity trends
            if conditions.get('commodity_trend', 'stable') == 'volatile':
                return min(base * 1.3, 1.0)
        elif component == 'interest_rate_sensitivity':
            # Check rate environment
            if conditions.get('rate_trend', 'stable') == 'rising':
                return min(base * 1.2, 1.0)
                
        return base
    
    def _adjust_for_cluster(self, component: str, base: float, cluster: Dict) -> float:
        """Adjust risk based on cluster characteristics"""
        # Size adjustment - larger clusters more resilient
        business_count = cluster.get('business_count', 0)
        if business_count > 100:
            base *= 0.9  # 10% risk reduction
        elif business_count < 20:
            base *= 1.1  # 10% risk increase
            
        # Diversity adjustment
        if component in ['market_diversification', 'adaptability']:
            # Check industry diversity within cluster
            businesses = cluster.get('businesses', [])
            unique_naics = len(set(b.get('naics_code', '')[:3] for b in businesses[:50]))
            if unique_naics > 10:
                base *= 0.8  # More diverse = lower risk
                
        return min(max(base, 0), 1)  # Keep in 0-1 range
    
    def _get_default_risk_profile(self) -> Dict:
        """Get default risk profile for unknown cluster types"""
        return {component: 0.5 for components in self.risk_factors.values() 
                for component in components['components']}
    
    def _estimate_time_horizon(self, score: float) -> str:
        """Estimate viable time horizon based on score"""
        if score >= 80:
            return "10+ years"
        elif score >= 60:
            return "5-10 years"
        elif score >= 40:
            return "3-5 years"
        elif score >= 20:
            return "1-3 years"
        else:
            return "<1 year"
    
    def _get_grade(self, score: float) -> str:
        """Convert score to letter grade"""
        if score >= 90:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 80:
            return "A-"
        elif score >= 75:
            return "B+"
        elif score >= 70:
            return "B"
        elif score >= 65:
            return "B-"
        elif score >= 60:
            return "C+"
        elif score >= 55:
            return "C"
        elif score >= 50:
            return "C-"
        elif score >= 40:
            return "D"
        else:
            return "F"
    
    def _identify_key_risks(self, risk_scores: Dict, risk_profile: Dict) -> List[Dict]:
        """Identify the most significant risks"""
        key_risks = []
        
        # Find highest risk categories
        sorted_risks = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)
        
        for category, score in sorted_risks[:3]:  # Top 3 risks
            if score > 0.6:  # Significant risk threshold
                # Find specific high-risk components
                components = self.risk_factors[category]['components']
                high_components = [
                    comp for comp in components 
                    if risk_profile.get(comp, 0.5) > 0.7
                ]
                
                key_risks.append({
                    'category': category,
                    'severity': 'high' if score > 0.8 else 'medium',
                    'score': score,
                    'components': high_components
                })
                
        return key_risks
    
    def _suggest_mitigations(self, cluster_type: str, risk_scores: Dict) -> List[Dict]:
        """Suggest mitigation strategies for identified risks"""
        mitigations = []
        
        # Political risk mitigations
        if risk_scores.get('political', 0) > 0.6:
            mitigations.append({
                'risk': 'Political/Regulatory',
                'strategy': 'Diversify customer base across states/countries',
                'priority': 'high',
                'timeline': '6-12 months'
            })
            
        # Market risk mitigations
        if risk_scores.get('market', 0) > 0.6:
            mitigations.append({
                'risk': 'Market Volatility',
                'strategy': 'Implement hedging strategies for commodity exposure',
                'priority': 'high',
                'timeline': '3-6 months'
            })
            
        # Technology risk mitigations
        if cluster_type == 'logistics' and risk_scores.get('market', 0) > 0.5:
            mitigations.append({
                'risk': 'Technology Disruption',
                'strategy': 'Invest in automation and autonomous vehicle partnerships',
                'priority': 'medium',
                'timeline': '1-3 years'
            })
            
        # Financial risk mitigations
        if risk_scores.get('financial', 0) > 0.6:
            mitigations.append({
                'risk': 'Financial',
                'strategy': 'Establish diverse funding sources and credit facilities',
                'priority': 'high',
                'timeline': '3-6 months'
            })
            
        return mitigations
    
    def _generate_assessment(self, cluster: Dict, risk_scores: Dict, 
                           longevity_score: float, time_horizon: str) -> str:
        """Generate narrative assessment"""
        cluster_name = cluster.get('name', 'This cluster')
        cluster_type = cluster.get('type', 'mixed')
        
        # Determine overall outlook
        if longevity_score >= 70:
            outlook = "strong"
            recommendation = "recommended for long-term investment"
        elif longevity_score >= 50:
            outlook = "moderate"
            recommendation = "suitable for medium-term initiatives with risk management"
        else:
            outlook = "challenging"
            recommendation = "requires significant risk mitigation or short-term focus"
            
        assessment = f"{cluster_name} shows {outlook} longevity prospects with a {time_horizon} viable horizon. "
        
        # Add specific risks
        highest_risk = max(risk_scores.items(), key=lambda x: x[1])
        assessment += f"The primary risk factor is {highest_risk[0]} (score: {highest_risk[1]:.2f}). "
        
        # Add strengths
        lowest_risk = min(risk_scores.items(), key=lambda x: x[1])
        assessment += f"The cluster shows resilience in {lowest_risk[0]} (score: {lowest_risk[1]:.2f}). "
        
        # Final recommendation
        assessment += f"Overall, this {cluster_type} cluster is {recommendation}."
        
        return assessment
    
    def generate_longevity_report(self, clusters: List[Dict],
                                 market_conditions: Optional[Dict] = None) -> Dict:
        """Generate comprehensive longevity report for all clusters"""
        # Get current conditions if not provided
        if not market_conditions:
            market_conditions = self._get_current_market_conditions()
            
        political_climate = self._get_political_climate()
        
        report = {
            'assessment_date': datetime.now().isoformat(),
            'market_conditions': market_conditions,
            'political_climate': political_climate,
            'cluster_scores': {},
            'rankings': [],
            'recommendations': []
        }
        
        # Score each cluster
        for cluster in clusters:
            cluster_name = cluster.get('name', 'Unknown')
            longevity = self.calculate_longevity_score(
                cluster, 
                market_conditions, 
                political_climate
            )
            report['cluster_scores'][cluster_name] = longevity
            
            report['rankings'].append({
                'cluster': cluster_name,
                'score': longevity['score'],
                'grade': longevity['grade'],
                'horizon': longevity['time_horizon']
            })
            
        # Sort rankings
        report['rankings'].sort(key=lambda x: x['score'], reverse=True)
        
        # Generate portfolio recommendations
        report['recommendations'] = self._generate_portfolio_recommendations(report['rankings'])
        
        return report
    
    def _get_current_market_conditions(self) -> Dict:
        """Get current market conditions (simplified)"""
        # In production, this would fetch real data
        return {
            'volatility_index': 22,  # VIX
            'rate_trend': 'stable',
            'commodity_trend': 'rising',
            'economic_growth': 'moderate',
            'inflation': 'elevated'
        }
    
    def _get_political_climate(self) -> Dict:
        """Get current political climate indicators"""
        # In production, this would analyze news/policy
        current_year = datetime.now().year
        return {
            'election_year': current_year % 4 == 0,
            'trade_tensions': True,  # Ongoing US-China issues
            'regulatory_activity': 'moderate',
            'policy_uncertainty': 'elevated'
        }
    
    def _generate_portfolio_recommendations(self, rankings: List[Dict]) -> List[str]:
        """Generate strategic recommendations based on longevity analysis"""
        recommendations = []
        
        # Identify best long-term bets
        long_term = [r for r in rankings if r['score'] >= 70]
        if long_term:
            recommendations.append(
                f"Prioritize {long_term[0]['cluster']} for long-term development (score: {long_term[0]['score']})"
            )
            
        # Identify risky clusters
        risky = [r for r in rankings if r['score'] < 50]
        if risky:
            recommendations.append(
                f"Implement risk mitigation for {risky[0]['cluster']} or consider short-term focus only"
            )
            
        # Portfolio balance
        if len(rankings) > 3:
            recommendations.append(
                "Diversify investments across multiple clusters to balance risk"
            )
            
        return recommendations