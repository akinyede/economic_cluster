"""Revenue projection system for individual businesses and clusters"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)

class RevenueProjector:
    """Project revenue growth for businesses and clusters"""
    
    def __init__(self):
        # Industry-specific growth models
        self.growth_models = {
            'logistics': {
                'base_growth': 0.067,  # 6.7% industry CAGR
                'size_factors': {
                    'small': 1.2,      # <50 employees grow faster
                    'medium': 1.0,     # 50-500 employees at industry rate
                    'large': 0.8       # >500 employees grow slower
                },
                'maturity_curve': 'sigmoid'  # S-curve growth
            },
            'technology': {
                'base_growth': 0.085,
                'size_factors': {
                    'small': 1.5,      # Startups can grow very fast
                    'medium': 1.2,
                    'large': 0.9
                },
                'maturity_curve': 'exponential'  # Early exponential growth
            },
            'biosciences': {
                'base_growth': 0.072,
                'size_factors': {
                    'small': 0.8,      # Slow early due to R&D
                    'medium': 1.3,     # Accelerate with products
                    'large': 1.0
                },
                'maturity_curve': 'delayed_sigmoid'  # Slow start, then rapid
            },
            'manufacturing': {
                'base_growth': 0.038,
                'size_factors': {
                    'small': 1.1,
                    'medium': 1.0,
                    'large': 0.95
                },
                'maturity_curve': 'linear'  # Steady growth
            },
            'animal_health': {
                'base_growth': 0.058,
                'size_factors': {
                    'small': 1.2,
                    'medium': 1.1,
                    'large': 0.9
                },
                'maturity_curve': 'sigmoid'
            }
        }
        
        # Revenue per employee benchmarks by industry
        self.revenue_per_employee = {
            'logistics': {
                'small': 180_000,
                'medium': 220_000,
                'large': 250_000
            },
            'technology': {
                'small': 150_000,
                'medium': 300_000,
                'large': 500_000
            },
            'biosciences': {
                'small': 200_000,
                'medium': 400_000,
                'large': 600_000
            },
            'manufacturing': {
                'small': 150_000,
                'medium': 200_000,
                'large': 280_000
            },
            'animal_health': {
                'small': 250_000,
                'medium': 350_000,
                'large': 450_000
            }
        }
        
    def project_business_revenue(self, business: Dict, years: int = 5) -> Dict:
        """Project revenue for an individual business"""
        
        # Extract business characteristics
        current_revenue = business.get('revenue_estimate', 0)
        employees = business.get('employees', 1)
        years_in_business = business.get('years_in_business', 1)
        industry = self._determine_industry(business)
        size_category = self._categorize_size(employees)
        
        # Get growth model
        model = self.growth_models.get(industry, self.growth_models['manufacturing'])
        
        # Calculate base growth rate
        base_growth = model['base_growth']
        size_factor = model['size_factors'].get(size_category, 1.0)
        
        # Adjust for business maturity
        maturity_factor = self._calculate_maturity_factor(
            years_in_business, 
            model['maturity_curve']
        )
        
        # Adjust for business quality (composite score)
        quality_factor = self._calculate_quality_factor(business)
        
        # Combined growth rate
        adjusted_growth = base_growth * size_factor * maturity_factor * quality_factor
        
        # Generate projections
        projections = {
            'current_revenue': current_revenue,
            'current_employees': employees,
            'growth_rate': adjusted_growth,
            'yearly_projections': {},
            'confidence_intervals': {}
        }
        
        # Project year by year
        for year in range(1, years + 1):
            # Revenue projection
            if model['maturity_curve'] == 'exponential':
                projected_revenue = current_revenue * (1 + adjusted_growth) ** year
            elif model['maturity_curve'] == 'sigmoid':
                projected_revenue = self._sigmoid_growth(
                    current_revenue, adjusted_growth, year, years_in_business
                )
            elif model['maturity_curve'] == 'delayed_sigmoid':
                projected_revenue = self._delayed_sigmoid_growth(
                    current_revenue, adjusted_growth, year, years_in_business
                )
            else:  # linear
                projected_revenue = current_revenue * (1 + adjusted_growth * year)
                
            projections['yearly_projections'][2024 + year] = {
                'revenue': projected_revenue,
                'employees': self._project_employees(employees, projected_revenue, industry),
                'growth_rate': adjusted_growth * maturity_factor
            }
            
            # Calculate confidence intervals
            std_dev = projected_revenue * 0.15  # 15% standard deviation
            projections['confidence_intervals'][2024 + year] = {
                'lower_bound': projected_revenue - 1.96 * std_dev,  # 95% CI
                'upper_bound': projected_revenue + 1.96 * std_dev
            }
            
        # Add summary metrics
        projections['total_growth'] = (
            projections['yearly_projections'][2024 + years]['revenue'] / 
            current_revenue - 1
        ) * 100
        
        projections['cagr'] = (
            (projections['yearly_projections'][2024 + years]['revenue'] / 
             current_revenue) ** (1/years) - 1
        ) * 100
        
        return projections
    
    def _determine_industry(self, business: Dict) -> str:
        """Determine industry from NAICS code"""
        naics = str(business.get('naics_code', ''))
        
        # Industry mapping
        if naics.startswith(('484', '488', '493')):
            return 'logistics'
        elif naics.startswith(('518', '541')):
            return 'technology'
        elif naics.startswith(('325', '339', '621')):
            return 'biosciences'
        elif naics.startswith(('311', '312', '313', '314', '315', '316', '321', '322', '323', '324', '326', '327', '331', '332', '333', '334', '335', '336', '337')):
            return 'manufacturing'
        elif naics.startswith('3254'):
            return 'animal_health'
        else:
            return 'manufacturing'  # Default
            
    def _categorize_size(self, employees: int) -> str:
        """Categorize business size"""
        if employees < 50:
            return 'small'
        elif employees < 500:
            return 'medium'
        else:
            return 'large'
            
    def _calculate_maturity_factor(self, years_in_business: int, curve: str) -> float:
        """Calculate growth adjustment based on business maturity"""
        if curve == 'exponential':
            # Young companies grow faster
            if years_in_business < 3:
                return 1.5
            elif years_in_business < 7:
                return 1.2
            elif years_in_business < 15:
                return 1.0
            else:
                return 0.8
                
        elif curve == 'sigmoid':
            # S-curve: slow-fast-slow
            if years_in_business < 3:
                return 0.8
            elif years_in_business < 10:
                return 1.3
            else:
                return 0.9
                
        elif curve == 'delayed_sigmoid':
            # Biotech pattern: very slow, then rapid
            if years_in_business < 5:
                return 0.5
            elif years_in_business < 12:
                return 1.5
            else:
                return 1.0
                
        else:  # linear
            return 1.0
            
    def _calculate_quality_factor(self, business: Dict) -> float:
        """Adjust growth based on business quality metrics"""
        composite_score = business.get('composite_score', 50)
        
        # Higher quality businesses grow faster
        if composite_score >= 80:
            return 1.3
        elif composite_score >= 65:
            return 1.15
        elif composite_score >= 50:
            return 1.0
        elif composite_score >= 35:
            return 0.85
        else:
            return 0.7
            
    def _sigmoid_growth(self, current: float, rate: float, year: int, 
                       maturity: int) -> float:
        """Calculate sigmoid (S-curve) growth"""
        # Logistic function parameters
        L = current * 10  # Max capacity (10x current)
        k = rate * 2      # Steepness
        x0 = 10          # Midpoint (years)
        
        x = maturity + year
        return L / (1 + np.exp(-k * (x - x0)))
        
    def _delayed_sigmoid_growth(self, current: float, rate: float, year: int,
                              maturity: int) -> float:
        """Calculate delayed sigmoid growth (biotech pattern)"""
        # Similar to sigmoid but shifted right
        L = current * 20  # Higher max for biotech
        k = rate * 1.5
        x0 = 15          # Later inflection point
        
        x = maturity + year
        return L / (1 + np.exp(-k * (x - x0)))
        
    def _project_employees(self, current_employees: int, projected_revenue: float,
                         industry: str) -> int:
        """Project employee count based on revenue"""
        # Use industry benchmarks
        size_category = self._categorize_size(current_employees)
        rev_per_emp = self.revenue_per_employee[industry][size_category]
        
        projected_employees = int(projected_revenue / rev_per_emp)
        
        # Smooth the transition
        return int(current_employees * 0.7 + projected_employees * 0.3)
        
    def project_cluster_revenue(self, cluster: Dict, years: int = 5) -> Dict:
        """Project revenue for entire cluster"""
        businesses = cluster.get('businesses', [])
        cluster_type = cluster.get('type', 'mixed')
        
        # Project each business
        business_projections = []
        for business in businesses:
            projection = self.project_business_revenue(business, years)
            business_projections.append(projection)
            
        # Aggregate projections
        cluster_projection = {
            'cluster_name': cluster.get('name', 'Unknown'),
            'cluster_type': cluster_type,
            'business_count': len(businesses),
            'current_revenue': sum(p['current_revenue'] for p in business_projections),
            'yearly_totals': {},
            'growth_metrics': {},
            'top_growers': []
        }
        
        # Calculate yearly totals
        for year in range(2024 + 1, 2024 + years + 1):
            total_revenue = sum(
                p['yearly_projections'][year]['revenue'] 
                for p in business_projections
            )
            total_employees = sum(
                p['yearly_projections'][year]['employees']
                for p in business_projections
            )
            
            cluster_projection['yearly_totals'][year] = {
                'revenue': total_revenue,
                'employees': total_employees,
                'revenue_per_employee': total_revenue / total_employees if total_employees > 0 else 0
            }
            
        # Calculate growth metrics
        final_revenue = cluster_projection['yearly_totals'][2024 + years]['revenue']
        cluster_projection['growth_metrics'] = {
            'total_growth': (final_revenue / cluster_projection['current_revenue'] - 1) * 100,
            'cagr': ((final_revenue / cluster_projection['current_revenue']) ** (1/years) - 1) * 100,
            'revenue_potential': final_revenue,
            'job_creation': (cluster_projection['yearly_totals'][2024 + years]['employees'] - 
                           sum(b.get('employees', 0) for b in businesses))
        }
        
        # Identify top growth businesses
        growth_rates = []
        for i, (business, projection) in enumerate(zip(businesses, business_projections)):
            growth_rates.append({
                'business': business.get('name', f'Business {i}'),
                'current_revenue': projection['current_revenue'],
                'projected_revenue': projection['yearly_projections'][2024 + years]['revenue'],
                'growth_rate': projection['total_growth']
            })
            
        cluster_projection['top_growers'] = sorted(
            growth_rates, 
            key=lambda x: x['growth_rate'], 
            reverse=True
        )[:10]
        
        # Add market share analysis
        cluster_projection['market_analysis'] = self._analyze_market_share(
            cluster_type, cluster_projection
        )
        
        # Add fields expected by tests
        avg_growth_rate = cluster_projection['growth_metrics']['cagr']
        cluster_projection['projected_growth_rate'] = avg_growth_rate
        cluster_projection['revenue_potential_score'] = min(cluster_projection['current_revenue'] / 1e6, 100)
        cluster_projection['confidence_level'] = self._determine_confidence_level(businesses, avg_growth_rate)
        
        return cluster_projection
        
    def _analyze_market_share(self, cluster_type: str, projection: Dict) -> Dict:
        """Analyze potential market share"""
        # Market sizes (simplified)
        market_sizes = {
            'logistics': 36_000_000_000,    # KC metro logistics market
            'technology': 12_000_000_000,   # KC metro tech market
            'biosciences': 8_000_000_000,   # KC metro bio market
            'manufacturing': 45_000_000_000, # KC metro manufacturing
            'animal_health': 2_000_000_000   # KC metro animal health
        }
        
        market_size = market_sizes.get(cluster_type, 10_000_000_000)
        current_share = (projection['current_revenue'] / market_size) * 100
        future_share = (projection['growth_metrics']['revenue_potential'] / market_size) * 100
        
        return {
            'addressable_market': market_size,
            'current_market_share': current_share,
            'projected_market_share': future_share,
            'share_gain': future_share - current_share,
            'market_position': self._assess_market_position(future_share)
        }
        
    def _assess_market_position(self, market_share: float) -> str:
        """Assess market position based on share"""
        if market_share >= 15:
            return 'Market Leader'
        elif market_share >= 10:
            return 'Major Player'
        elif market_share >= 5:
            return 'Significant Presence'
        elif market_share >= 1:
            return 'Established Player'
        else:
            return 'Emerging Player'
    
    def _determine_confidence_level(self, businesses: List[Dict], growth_rate: float) -> str:
        """Determine confidence level based on data quality and projections"""
        if len(businesses) >= 10 and growth_rate > 0 and growth_rate < 20:
            return 'High'
        elif len(businesses) >= 5 and growth_rate > -5 and growth_rate < 30:
            return 'Medium'
        else:
            return 'Low'
            
    def generate_revenue_report(self, clusters: List[Dict]) -> Dict:
        """Generate comprehensive revenue projection report"""
        report = {
            'total_current_revenue': 0,
            'total_projected_revenue': 0,
            'by_cluster': {},
            'growth_rankings': [],
            'market_opportunities': [],
            'investment_priorities': []
        }
        
        # Project each cluster
        for cluster in clusters:
            cluster_name = cluster.get('name', 'Unknown')
            projection = self.project_cluster_revenue(cluster)
            
            report['by_cluster'][cluster_name] = projection
            report['total_current_revenue'] += projection['current_revenue']
            report['total_projected_revenue'] += projection['growth_metrics']['revenue_potential']
            
            # Track rankings
            report['growth_rankings'].append({
                'cluster': cluster_name,
                'cagr': projection['growth_metrics']['cagr'],
                'revenue_potential': projection['growth_metrics']['revenue_potential'],
                'market_share_gain': projection['market_analysis']['share_gain']
            })
            
        # Sort rankings
        report['growth_rankings'].sort(key=lambda x: x['cagr'], reverse=True)
        
        # Identify opportunities
        report['market_opportunities'] = self._identify_opportunities(report)
        
        # Investment priorities
        report['investment_priorities'] = self._prioritize_investments(report)
        
        # Summary metrics
        report['summary'] = {
            'total_growth': ((report['total_projected_revenue'] / 
                            report['total_current_revenue']) - 1) * 100,
            'weighted_cagr': self._calculate_weighted_cagr(report),
            'total_job_creation': sum(
                p['growth_metrics']['job_creation'] 
                for p in report['by_cluster'].values()
            )
        }
        
        return report
        
    def _identify_opportunities(self, report: Dict) -> List[Dict]:
        """Identify market opportunities from projections"""
        opportunities = []
        
        # High growth clusters
        for ranking in report['growth_rankings'][:3]:
            if ranking['cagr'] > 10:
                opportunities.append({
                    'type': 'high_growth',
                    'cluster': ranking['cluster'],
                    'opportunity': f"{ranking['cagr']:.1f}% CAGR growth potential",
                    'action': 'Accelerate investment and scaling'
                })
                
        # Market share gains
        for cluster_name, projection in report['by_cluster'].items():
            share_gain = projection['market_analysis']['share_gain']
            if share_gain > 2:
                opportunities.append({
                    'type': 'market_expansion',
                    'cluster': cluster_name,
                    'opportunity': f"{share_gain:.1f}% market share gain possible",
                    'action': 'Focus on competitive positioning'
                })
                
        return opportunities
        
    def _prioritize_investments(self, report: Dict) -> List[Dict]:
        """Prioritize investment recommendations"""
        priorities = []
        
        for cluster_name, projection in report['by_cluster'].items():
            # Calculate investment score
            score = (
                projection['growth_metrics']['cagr'] * 0.4 +
                projection['market_analysis']['share_gain'] * 10 * 0.3 +
                min(projection['growth_metrics']['job_creation'] / 1000, 10) * 0.3
            )
            
            priorities.append({
                'cluster': cluster_name,
                'score': score,
                'revenue_potential': projection['growth_metrics']['revenue_potential'],
                'investment_focus': self._determine_investment_focus(projection)
            })
            
        # Sort by score
        priorities.sort(key=lambda x: x['score'], reverse=True)
        
        return priorities
        
    def _determine_investment_focus(self, projection: Dict) -> str:
        """Determine where to focus investment"""
        top_growers = projection.get('top_growers', [])
        if not top_growers:
            return 'General cluster development'
            
        # If top businesses dominate growth
        top_3_growth = sum(g['growth_rate'] for g in top_growers[:3])
        total_growth = sum(g['growth_rate'] for g in top_growers)
        
        if top_3_growth / total_growth > 0.5:
            return f"Focus on top performers: {', '.join(g['business'] for g in top_growers[:3])}"
        else:
            return 'Broad-based support across cluster'
            
    def _calculate_weighted_cagr(self, report: Dict) -> float:
        """Calculate revenue-weighted CAGR"""
        weighted_sum = 0
        total_revenue = report['total_current_revenue']
        
        for projection in report['by_cluster'].values():
            weight = projection['current_revenue'] / total_revenue
            weighted_sum += projection['growth_metrics']['cagr'] * weight
            
        return weighted_sum