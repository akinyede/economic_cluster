"""Enhanced market analysis with real-time data and competitive intelligence"""

import requests
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import yfinance as yf

try:  # Optional dependency for live FRED data
    from fredapi import Fred  # type: ignore
    _FRED_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    Fred = None  # type: ignore
    _FRED_AVAILABLE = False

logger = logging.getLogger(__name__)

class EnhancedMarketAnalyzer:
    """Comprehensive market analysis for cluster opportunities"""
    
    def __init__(self, fred_api_key: Optional[str] = None):
        if _FRED_AVAILABLE and fred_api_key:
            self.fred = Fred(api_key=fred_api_key)
        else:
            self.fred = None
            if not _FRED_AVAILABLE:
                logger.warning("fredapi package not available; market analyzer will use static indicators")
        
        # Industry mapping to market data
        self.industry_mapping = {
            'logistics': {
                'fred_series': ['DTCTLVEUSC', 'PPITRN'],  # Transport services index
                'competitors': ['FDX', 'UPS', 'XPO', 'JBHT'],
                'market_size': 1_800_000_000_000,  # $1.8T US logistics market
                'growth_rate': 0.067  # 6.7% CAGR
            },
            'technology': {
                'fred_series': ['BOGZ1FA318013353Q'],  # Tech sector output
                'competitors': ['MSFT', 'GOOGL', 'CRM', 'NOW'],
                'market_size': 1_600_000_000_000,
                'growth_rate': 0.085
            },
            'biosciences': {
                'fred_series': ['PCU3254325403'],  # Pharma price index
                'competitors': ['JNJ', 'PFE', 'AMGN', 'GILD'],
                'market_size': 550_000_000_000,
                'growth_rate': 0.072
            },
            'manufacturing': {
                'fred_series': ['IPMAN', 'PPIACO'],  # Manufacturing indices
                'competitors': ['CAT', 'DE', 'MMM', 'HON'],
                'market_size': 2_300_000_000_000,
                'growth_rate': 0.038
            },
            'animal_health': {
                'fred_series': ['PCU311111311111'],  # Animal food mfg
                'competitors': ['ZTS', 'IDXX', 'ELAN', 'PETQ'],
                'market_size': 45_000_000_000,
                'growth_rate': 0.058
            }
        }
        
    def analyze_market_opportunity(self, cluster_type: str, businesses: List[Dict]) -> Dict:
        """Comprehensive market opportunity analysis"""
        
        if cluster_type not in self.industry_mapping:
            cluster_type = 'manufacturing'  # Default
            
        industry = self.industry_mapping[cluster_type]
        
        analysis = {
            'market_size': industry['market_size'],
            'growth_rate': industry['growth_rate'],
            'market_trends': self._get_market_trends(cluster_type),
            'competitive_landscape': self._analyze_competition(cluster_type),
            'demand_forecast': self._forecast_demand(cluster_type),
            'addressable_market': self._calculate_addressable_market(cluster_type, businesses),
            'market_share_potential': self._estimate_market_share(cluster_type, businesses),
            'entry_barriers': self._assess_entry_barriers(cluster_type),
            'opportunities': self._identify_opportunities(cluster_type),
            'threats': self._identify_threats(cluster_type)
        }
        
        # Calculate overall market opportunity score
        score = 0.0
        
        # Growth rate component (0-30 points)
        score += min(industry['growth_rate'] * 300, 30)
        
        # Market size component (0-20 points)
        if industry['market_size'] > 1e12:  # > $1T
            score += 20
        elif industry['market_size'] > 1e11:  # > $100B
            score += 15
        elif industry['market_size'] > 1e10:  # > $10B
            score += 10
        else:
            score += 5
            
        # Addressable market component (0-20 points)
        addressable = analysis['addressable_market']['obtainable_market']
        if addressable > 1e9:  # > $1B
            score += 20
        elif addressable > 1e8:  # > $100M
            score += 15
        elif addressable > 1e7:  # > $10M
            score += 10
        else:
            score += 5
            
        # Competition component (0-15 points)
        concentration = analysis['competitive_landscape']['market_concentration']
        if concentration < 1500:  # Low concentration
            score += 15
        elif concentration < 2500:  # Moderate
            score += 10
        else:  # High concentration
            score += 5
            
        # Opportunities vs threats (0-15 points)
        opp_count = len(analysis['opportunities'])
        threat_count = len(analysis['threats'])
        if opp_count > threat_count * 2:
            score += 15
        elif opp_count > threat_count:
            score += 10
        else:
            score += 5
            
        analysis['market_opportunity_score'] = min(score, 100)
        analysis['overall_score'] = analysis['market_opportunity_score']
        analysis['growth_potential'] = min(industry['growth_rate'] * 1000, 100)
        analysis['market_size_index'] = min(industry['market_size'] / 1e11, 100)  # Normalized to 100B
        
        return analysis
    
    def _get_market_trends(self, cluster_type: str) -> Dict:
        """Get current market trends from FRED and other sources"""
        trends = {
            'price_indices': {},
            'demand_indicators': {},
            'supply_indicators': {}
        }
        
        if self.fred and cluster_type in self.industry_mapping:
            for series_id in self.industry_mapping[cluster_type]['fred_series']:
                try:
                    # Get last 5 years of data
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=5*365)
                    
                    data = self.fred.get_series(series_id, start_date, end_date)
                    if not data.empty:
                        trends['price_indices'][series_id] = {
                            'current': float(data.iloc[-1]),
                            'year_ago': float(data.iloc[-252]) if len(data) > 252 else float(data.iloc[0]),
                            'trend': 'increasing' if data.iloc[-1] > data.iloc[-252] else 'decreasing'
                        }
                except Exception as e:
                    logger.warning(f"Could not fetch FRED series {series_id}: {e}")
        
        # Add specific trend indicators
        if cluster_type == 'logistics':
            trends['demand_indicators']['ecommerce_growth'] = 0.11  # 11% annual
            trends['demand_indicators']['freight_volume'] = 'increasing'
            trends['supply_indicators']['driver_shortage'] = 80_000  # Truck drivers needed
            
        elif cluster_type == 'biosciences':
            trends['demand_indicators']['aging_population'] = 0.17  # 17% over 65
            trends['demand_indicators']['rd_spending'] = 'increasing'
            trends['supply_indicators']['clinical_trials'] = 'accelerating'
            
        return trends
    
    def _analyze_competition(self, cluster_type: str) -> Dict:
        """Analyze competitive landscape using stock market data"""
        competition = {
            'major_players': [],
            'market_concentration': 0,
            'competitive_intensity': 'medium'
        }
        
        if cluster_type in self.industry_mapping:
            tickers = self.industry_mapping[cluster_type]['competitors']
            
            for ticker in tickers[:4]:  # Top 4 competitors
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    competition['major_players'].append({
                        'ticker': ticker,
                        'name': info.get('longName', ticker),
                        'market_cap': info.get('marketCap', 0),
                        'revenue': info.get('totalRevenue', 0),
                        'employees': info.get('fullTimeEmployees', 0),
                        'profit_margin': info.get('profitMargins', 0)
                    })
                except Exception as e:
                    logger.warning(f"Could not fetch data for {ticker}: {e}")
            
            # Calculate market concentration (HHI)
            if competition['major_players']:
                total_revenue = sum(p['revenue'] for p in competition['major_players'])
                if total_revenue > 0:
                    shares = [(p['revenue'] / total_revenue) * 100 for p in competition['major_players']]
                    competition['market_concentration'] = sum(s**2 for s in shares)
                    
                    # Interpret concentration
                    if competition['market_concentration'] > 2500:
                        competition['competitive_intensity'] = 'high concentration'
                    elif competition['market_concentration'] > 1500:
                        competition['competitive_intensity'] = 'moderate concentration'
                    else:
                        competition['competitive_intensity'] = 'low concentration'
        
        return competition
    
    def _forecast_demand(self, cluster_type: str, years: int = 5) -> Dict:
        """Forecast demand based on growth rates and trends"""
        industry = self.industry_mapping.get(cluster_type, {})
        base_market = industry.get('market_size', 1_000_000_000)
        growth_rate = industry.get('growth_rate', 0.05)
        
        forecast = {
            'current_market': base_market,
            'forecast_years': {}
        }
        
        for year in range(1, years + 1):
            forecast['forecast_years'][2024 + year] = base_market * (1 + growth_rate) ** year
            
        forecast['total_growth'] = (forecast['forecast_years'][2024 + years] / base_market - 1) * 100
        forecast['cagr'] = growth_rate * 100
        
        return forecast
    
    def _calculate_addressable_market(self, cluster_type: str, businesses: List[Dict]) -> Dict:
        """Calculate specific addressable market for the cluster"""
        industry = self.industry_mapping.get(cluster_type, {})
        total_market = industry.get('market_size', 1_000_000_000)
        
        # Estimate based on business characteristics
        total_revenue = sum(b.get('revenue_estimate', 0) for b in businesses)
        avg_business_size = total_revenue / len(businesses) if businesses else 1_000_000
        
        # Addressable market factors
        geographic_factor = 0.02  # KC metro is ~2% of US economy
        capability_factor = 0.5   # Can address 50% of opportunities
        competition_factor = 0.3  # Can win 30% of deals
        
        addressable = {
            'total_market': total_market,
            'geographic_market': total_market * geographic_factor,
            'serviceable_market': total_market * geographic_factor * capability_factor,
            'obtainable_market': total_market * geographic_factor * capability_factor * competition_factor,
            'market_penetration_needed': 0.01  # 1% to be successful
        }
        
        return addressable
    
    def _estimate_market_share(self, cluster_type: str, businesses: List[Dict]) -> Dict:
        """Estimate potential market share"""
        addressable = self._calculate_addressable_market(cluster_type, businesses)
        
        # Current position
        total_revenue = sum(b.get('revenue_estimate', 0) for b in businesses)
        
        share = {
            'current_share': (total_revenue / addressable['geographic_market']) * 100 if addressable['geographic_market'] > 0 else 0,
            'potential_share': 5.0,  # Target 5% of geographic market
            'revenue_potential': addressable['geographic_market'] * 0.05,
            'growth_required': ((addressable['geographic_market'] * 0.05) / total_revenue - 1) * 100 if total_revenue > 0 else float('inf')
        }
        
        return share
    
    def _assess_entry_barriers(self, cluster_type: str) -> Dict:
        """Assess barriers to entry for the market"""
        barriers = {
            'capital_requirements': 'medium',
            'regulatory': 'low',
            'technology': 'medium',
            'brand_loyalty': 'low',
            'economies_of_scale': 'medium',
            'overall_difficulty': 'medium'
        }
        
        # Adjust by cluster type
        if cluster_type == 'biosciences':
            barriers['regulatory'] = 'high'
            barriers['capital_requirements'] = 'high'
            barriers['technology'] = 'high'
            barriers['overall_difficulty'] = 'high'
            
        elif cluster_type == 'logistics':
            barriers['capital_requirements'] = 'high'
            barriers['economies_of_scale'] = 'high'
            barriers['overall_difficulty'] = 'medium-high'
            
        elif cluster_type == 'technology':
            barriers['technology'] = 'high'
            barriers['brand_loyalty'] = 'medium'
            
        return barriers
    
    def _identify_opportunities(self, cluster_type: str) -> List[Dict]:
        """Identify specific market opportunities"""
        opportunities = []
        
        base_opportunities = [
            {
                'type': 'market_growth',
                'description': f"{cluster_type.title()} market growing at {self.industry_mapping.get(cluster_type, {}).get('growth_rate', 0.05)*100:.1f}% annually",
                'impact': 'high',
                'timeline': '1-3 years'
            }
        ]
        
        # Cluster-specific opportunities
        if cluster_type == 'logistics':
            opportunities.extend([
                {
                    'type': 'ecommerce_growth',
                    'description': 'E-commerce driving 11% annual growth in last-mile delivery',
                    'impact': 'high',
                    'timeline': 'immediate'
                },
                {
                    'type': 'supply_chain_reshoring',
                    'description': 'Manufacturing reshoring creating new logistics demand',
                    'impact': 'medium',
                    'timeline': '2-5 years'
                }
            ])
            
        elif cluster_type == 'biosciences':
            opportunities.extend([
                {
                    'type': 'aging_population',
                    'description': 'Aging demographics driving healthcare demand',
                    'impact': 'high',
                    'timeline': '5-10 years'
                },
                {
                    'type': 'personalized_medicine',
                    'description': 'Precision medicine creating new market segments',
                    'impact': 'high',
                    'timeline': '3-7 years'
                }
            ])
            
        opportunities.extend(base_opportunities)
        return opportunities
    
    def _identify_threats(self, cluster_type: str) -> List[Dict]:
        """Identify market threats"""
        threats = []
        
        base_threats = [
            {
                'type': 'economic_downturn',
                'description': 'Potential recession could reduce demand',
                'probability': 'medium',
                'impact': 'high'
            }
        ]
        
        # Cluster-specific threats
        if cluster_type == 'logistics':
            threats.extend([
                {
                    'type': 'automation',
                    'description': 'Autonomous vehicles could disrupt employment',
                    'probability': 'high',
                    'impact': 'high',
                    'timeline': '5-10 years'
                }
            ])
            
        elif cluster_type == 'manufacturing':
            threats.extend([
                {
                    'type': 'trade_policy',
                    'description': 'Tariffs and trade restrictions',
                    'probability': 'medium',
                    'impact': 'medium'
                }
            ])
            
        threats.extend(base_threats)
        return threats
    
    def generate_market_report(self, clusters: List[Dict]) -> Dict:
        """Generate comprehensive market analysis report for all clusters"""
        report = {
            'total_market_opportunity': 0,
            'by_cluster': {},
            'key_opportunities': [],
            'key_threats': [],
            'recommendations': []
        }
        
        for cluster in clusters:
            cluster_type = cluster.get('type', 'mixed')
            businesses = cluster.get('businesses', [])
            
            analysis = self.analyze_market_opportunity(cluster_type, businesses)
            report['by_cluster'][cluster.get('name', 'Unknown')] = analysis
            report['total_market_opportunity'] += analysis['addressable_market']['obtainable_market']
            
            # Aggregate opportunities and threats
            report['key_opportunities'].extend(analysis['opportunities'][:2])
            report['key_threats'].extend(analysis['threats'][:1])
            
        # Generate recommendations
        report['recommendations'] = self._generate_market_recommendations(report)
        
        return report
    
    def _generate_market_recommendations(self, report: Dict) -> List[str]:
        """Generate strategic recommendations based on market analysis"""
        recommendations = []
        
        # Find highest growth opportunity
        best_growth = max(report['by_cluster'].items(), 
                         key=lambda x: x[1]['growth_rate'])
        recommendations.append(
            f"Prioritize {best_growth[0]} - highest growth rate at {best_growth[1]['growth_rate']*100:.1f}%"
        )
        
        # Find largest addressable market
        best_market = max(report['by_cluster'].items(),
                         key=lambda x: x[1]['addressable_market']['obtainable_market'])
        recommendations.append(
            f"Focus resources on {best_market[0]} - largest addressable market at ${best_market[1]['addressable_market']['obtainable_market']:,.0f}"
        )
        
        # Risk mitigation
        if any(t['probability'] == 'high' for t in report['key_threats']):
            recommendations.append(
                "Develop risk mitigation strategies for identified high-probability threats"
            )
            
        return recommendations
