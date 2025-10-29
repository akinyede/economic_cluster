"""Geopolitical risk analysis module for KC Cluster Prediction Tool"""
import requests
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeopoliticalAnalyzer:
    """Analyze geopolitical factors affecting business clusters"""
    
    def __init__(self):
        self.config = Config()
        self.census_api_key = self.config.CENSUS_API_KEY
        
    def fetch_trade_data(self) -> Dict:
        """Fetch international trade data from Census API"""
        trade_data = {}
        
        if not self.census_api_key:
            logger.warning("Census API key not found, using simulated data")
            return self._get_simulated_trade_data()
        
        # Census International Trade API endpoints
        base_url = "https://api.census.gov/data/timeseries/intltrade/exports/hs"
        
        try:
            # Get latest available data (usually 2-3 months behind)
            current_date = datetime.now()
            # Try last 3 months to find available data
            for i in range(3):
                test_date = current_date - timedelta(days=30 * i)
                year = test_date.year
                month = str(test_date.month).zfill(2)
                
                params = {
                    'get': 'ALL_VAL_MO,CTY_CODE,CTY_NAME',
                    'time': f'{year}-{month}',
                    'key': self.census_api_key
                }
                
                response = requests.get(base_url, params=params, timeout=self.config.API_TIMEOUT_SHORT)
                
                if response.status_code == 200:
                    data = response.json()
                    if len(data) > 1:  # Has data beyond header
                        trade_data['export_data'] = self._process_census_trade_data(data)
                        trade_data['data_date'] = f'{year}-{month}'
                        break
                        
        except Exception as e:
            logger.error(f"Error fetching trade data: {e}")
            return self._get_simulated_trade_data()
            
        # Get top trading partners and their stability
        if 'export_data' in trade_data:
            trade_data['top_partners'] = self._analyze_trading_partners(trade_data['export_data'])
            
        return trade_data
    
    def _process_census_trade_data(self, raw_data: List) -> Dict:
        """Process raw Census trade data"""
        if not raw_data or len(raw_data) < 2:
            return {}
            
        headers = raw_data[0]
        data_rows = raw_data[1:]
        
        # Create DataFrame for easier processing
        df = pd.DataFrame(data_rows, columns=headers)
        
        # Convert values to numeric
        if 'ALL_VAL_MO' in df.columns:
            df['ALL_VAL_MO'] = pd.to_numeric(df['ALL_VAL_MO'], errors='coerce')
            
        # Aggregate by country
        country_exports = {}
        for _, row in df.iterrows():
            country = row.get('CTY_NAME', 'Unknown')
            value = row.get('ALL_VAL_MO', 0)
            if country and country != 'Unknown' and value > 0:
                if country not in country_exports:
                    country_exports[country] = 0
                country_exports[country] += value
                
        # Sort by export value
        sorted_exports = sorted(country_exports.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_exports': sum(country_exports.values()),
            'country_exports': dict(sorted_exports[:20]),  # Top 20 partners
            'num_partners': len(country_exports)
        }
    
    def _analyze_trading_partners(self, export_data: Dict) -> List[Dict]:
        """Analyze trading partner stability and risks"""
        partners = []
        
        # Country risk scores (simplified - in production, use external API)
        risk_scores = {
            'Canada': 0.1,
            'Mexico': 0.3,
            'China': 0.6,
            'Japan': 0.2,
            'Germany': 0.2,
            'United Kingdom': 0.2,
            'South Korea': 0.3,
            'Netherlands': 0.2,
            'Brazil': 0.4,
            'India': 0.4,
            'France': 0.2,
            'Italy': 0.3,
            'Taiwan': 0.5,
            'Singapore': 0.2,
            'Belgium': 0.2
        }
        
        country_exports = export_data.get('country_exports', {})
        total_exports = export_data.get('total_exports', 1)
        
        for country, export_value in list(country_exports.items())[:10]:
            partners.append({
                'country': country,
                'export_value': export_value,
                'export_share': export_value / total_exports if total_exports > 0 else 0,
                'risk_score': risk_scores.get(country, 0.5),  # Default medium risk
                'stability': 'High' if risk_scores.get(country, 0.5) < 0.3 else 'Medium' if risk_scores.get(country, 0.5) < 0.6 else 'Low'
            })
            
        return partners
    
    def fetch_supply_chain_risks(self) -> Dict:
        """Analyze supply chain vulnerabilities"""
        risks = {
            'port_congestion': self._get_port_congestion_index(),
            'shipping_costs': self._get_shipping_cost_index(),
            'critical_materials': self._get_critical_materials_status(),
            'trade_restrictions': self._get_trade_restrictions()
        }
        
        return risks
    
    def _get_port_congestion_index(self) -> Dict:
        """Get port congestion data (simulated for now)"""
        # In production, could use MarineTraffic API or similar
        return {
            'west_coast': {
                'congestion_level': 0.7,  # 0-1 scale
                'avg_wait_days': 5.2,
                'trend': 'improving'
            },
            'east_coast': {
                'congestion_level': 0.4,
                'avg_wait_days': 2.1,
                'trend': 'stable'
            },
            'gulf_coast': {
                'congestion_level': 0.3,
                'avg_wait_days': 1.5,
                'trend': 'stable'
            }
        }
    
    def _get_shipping_cost_index(self) -> Dict:
        """Get shipping cost indices"""
        # Could integrate with Freightos Baltic Index API
        return {
            'container_20ft': {
                'current_price': 2850,
                'monthly_change': -0.05,
                'yearly_change': -0.42
            },
            'container_40ft': {
                'current_price': 3200,
                'monthly_change': -0.03,
                'yearly_change': -0.38
            },
            'air_freight': {
                'price_per_kg': 4.2,
                'monthly_change': 0.02,
                'yearly_change': -0.15
            }
        }
    
    def _get_critical_materials_status(self) -> Dict:
        """Track critical materials availability"""
        # Critical materials for different clusters
        return {
            'semiconductors': {
                'availability': 'constrained',
                'lead_time_weeks': 26,
                'price_trend': 'increasing',
                'affected_clusters': ['technology', 'manufacturing']
            },
            'rare_earth_metals': {
                'availability': 'moderate',
                'lead_time_weeks': 12,
                'price_trend': 'stable',
                'affected_clusters': ['technology', 'biosciences']
            },
            'lithium': {
                'availability': 'tight',
                'lead_time_weeks': 16,
                'price_trend': 'increasing',
                'affected_clusters': ['manufacturing', 'technology']
            },
            'steel': {
                'availability': 'good',
                'lead_time_weeks': 4,
                'price_trend': 'decreasing',
                'affected_clusters': ['manufacturing', 'logistics']
            }
        }
    
    def _get_trade_restrictions(self) -> Dict:
        """Get current trade restrictions and tariffs"""
        return {
            'active_tariffs': [
                {
                    'country': 'China',
                    'products': 'Electronics, machinery',
                    'rate': '25%',
                    'impact': 'high'
                },
                {
                    'country': 'EU',
                    'products': 'Steel, aluminum',
                    'rate': '10%',
                    'impact': 'medium'
                }
            ],
            'sanctions': [
                {
                    'country': 'Russia',
                    'scope': 'Technology, energy',
                    'impact': 'medium'
                }
            ],
            'trade_agreements': [
                {
                    'agreement': 'USMCA',
                    'partners': ['Canada', 'Mexico'],
                    'benefit': 'Reduced tariffs, streamlined logistics'
                },
                {
                    'agreement': 'KORUS',
                    'partners': ['South Korea'],
                    'benefit': 'Technology cooperation'
                }
            ]
        }
    
    def calculate_geopolitical_risk_score(self, cluster_type: str, trade_data: Dict, supply_chain_risks: Dict) -> float:
        """Calculate overall geopolitical risk score for a cluster type"""
        base_score = 0.5  # Neutral baseline
        
        # Trade concentration risk
        if 'top_partners' in trade_data:
            top_partners = trade_data['top_partners']
            if top_partners:
                # High concentration with risky partners increases risk
                concentration = sum(p['export_share'] for p in top_partners[:3])
                avg_risk = sum(p['risk_score'] * p['export_share'] for p in top_partners) / sum(p['export_share'] for p in top_partners)
                
                trade_risk = concentration * avg_risk
                base_score += trade_risk * 0.3
        
        # Supply chain risks by cluster
        if cluster_type == 'logistics':
            # Port congestion affects logistics heavily
            port_risk = max(
                supply_chain_risks['port_congestion']['west_coast']['congestion_level'],
                supply_chain_risks['port_congestion']['east_coast']['congestion_level']
            )
            base_score += port_risk * 0.2
            
            # Shipping costs
            if supply_chain_risks['shipping_costs']['container_20ft']['monthly_change'] > 0:
                base_score += 0.1
                
        elif cluster_type == 'technology':
            # Semiconductor availability
            if supply_chain_risks['critical_materials']['semiconductors']['availability'] == 'constrained':
                base_score += 0.2
            
            # Trade restrictions with China
            base_score += 0.15  # Due to electronics tariffs
            
        elif cluster_type == 'manufacturing':
            # Multiple material dependencies
            materials = ['semiconductors', 'steel', 'lithium']
            material_risk = 0
            for material in materials:
                if material in supply_chain_risks['critical_materials']:
                    if supply_chain_risks['critical_materials'][material]['availability'] in ['constrained', 'tight']:
                        material_risk += 0.1
            base_score += material_risk
            
        elif cluster_type == 'biosciences':
            # Less affected by trade issues, but some material dependencies
            if supply_chain_risks['critical_materials']['rare_earth_metals']['availability'] != 'good':
                base_score += 0.1
        
        return min(1.0, max(0.0, base_score))
    
    def get_geopolitical_insights(self, cluster_type: str, risk_score: float, trade_data: Dict, supply_chain_risks: Dict) -> List[str]:
        """Generate insights based on geopolitical analysis"""
        insights = []
        
        # Risk level insights
        if risk_score > 0.7:
            insights.append(f"High geopolitical risk ({risk_score:.2f}) suggests careful supply chain planning")
        elif risk_score > 0.4:
            insights.append(f"Moderate geopolitical risk ({risk_score:.2f}) - monitor trade developments")
        else:
            insights.append(f"Low geopolitical risk ({risk_score:.2f}) provides stable operating environment")
        
        # Trade partner insights
        if 'top_partners' in trade_data and trade_data['top_partners']:
            risky_partners = [p for p in trade_data['top_partners'][:5] if p['risk_score'] > 0.5]
            if risky_partners:
                countries = ', '.join(p['country'] for p in risky_partners)
                insights.append(f"High trade exposure to volatile markets: {countries}")
        
        # Cluster-specific insights
        if cluster_type == 'logistics':
            west_congestion = supply_chain_risks['port_congestion']['west_coast']['congestion_level']
            if west_congestion > 0.6:
                insights.append(f"West Coast port congestion ({west_congestion:.1%}) may impact logistics operations")
                
        elif cluster_type == 'technology':
            semi_status = supply_chain_risks['critical_materials']['semiconductors']
            if semi_status['availability'] == 'constrained':
                insights.append(f"Semiconductor shortage with {semi_status['lead_time_weeks']}-week lead times affects tech manufacturing")
                
        elif cluster_type == 'manufacturing':
            materials_at_risk = [
                name for name, data in supply_chain_risks['critical_materials'].items()
                if cluster_type in data.get('affected_clusters', []) and data['availability'] != 'good'
            ]
            if materials_at_risk:
                insights.append(f"Critical material constraints: {', '.join(materials_at_risk)}")
        
        # Trade agreement benefits
        agreements = supply_chain_risks.get('trade_restrictions', {}).get('trade_agreements', [])
        relevant_agreements = [
            a for a in agreements 
            if cluster_type == 'logistics' or (cluster_type == 'technology' and 'Technology' in a.get('benefit', ''))
        ]
        if relevant_agreements:
            insights.append(f"Leverage trade agreements: {', '.join(a['agreement'] for a in relevant_agreements)}")
        
        return insights
    
    def enhance_with_geopolitical_data(self, market_data: Dict) -> Dict:
        """Enhance market data with geopolitical risk analysis"""
        logger.info("Enhancing market data with geopolitical risk analysis...")
        
        # Fetch all geopolitical data
        geo_data = self.fetch_all_geopolitical_data()
        
        # Add to market data
        market_data['geopolitical_risks'] = geo_data['risk_scores']
        market_data['trade_data'] = geo_data['trade_data']
        market_data['supply_chain_risks'] = geo_data['supply_chain_risks']
        
        # Merge insights
        if 'insights' not in market_data:
            market_data['insights'] = []
            
        # Add geopolitical insights for each cluster
        for cluster_type, cluster_insights in geo_data['insights'].items():
            for insight in cluster_insights:
                market_data['insights'].append(f"[{cluster_type.upper()}] {insight}")
        
        # Update timestamp
        market_data['geopolitical_timestamp'] = geo_data['timestamp']
        
        logger.info("Geopolitical risk enhancement complete")
        return market_data
    
    def fetch_all_geopolitical_data(self) -> Dict:
        """Fetch all geopolitical risk data"""
        logger.info("Fetching geopolitical risk data...")
        
        geopolitical_data = {
            'timestamp': datetime.now().isoformat(),
            'trade_data': self.fetch_trade_data(),
            'supply_chain_risks': self.fetch_supply_chain_risks(),
            'risk_scores': {},
            'insights': {}
        }
        
        # Calculate risk scores for each cluster
        for cluster_type in ['logistics', 'biosciences', 'technology', 'manufacturing']:
            risk_score = self.calculate_geopolitical_risk_score(
                cluster_type,
                geopolitical_data['trade_data'],
                geopolitical_data['supply_chain_risks']
            )
            geopolitical_data['risk_scores'][cluster_type] = risk_score
            
            # Generate insights
            geopolitical_data['insights'][cluster_type] = self.get_geopolitical_insights(
                cluster_type,
                risk_score,
                geopolitical_data['trade_data'],
                geopolitical_data['supply_chain_risks']
            )
        
        logger.info("Geopolitical risk data fetch complete")
        return geopolitical_data
    
    def _get_simulated_trade_data(self) -> Dict:
        """Return simulated trade data for testing"""
        return {
            'export_data': {
                'total_exports': 150000000000,  # $150B
                'country_exports': {
                    'Canada': 30000000000,
                    'Mexico': 25000000000,
                    'China': 20000000000,
                    'Japan': 15000000000,
                    'Germany': 10000000000,
                    'United Kingdom': 8000000000,
                    'South Korea': 7000000000,
                    'Netherlands': 6000000000,
                    'Brazil': 5000000000,
                    'India': 4000000000
                },
                'num_partners': 150
            },
            'data_date': datetime.now().strftime('%Y-%m'),
            'top_partners': [
                {'country': 'Canada', 'export_value': 30000000000, 'export_share': 0.2, 'risk_score': 0.1, 'stability': 'High'},
                {'country': 'Mexico', 'export_value': 25000000000, 'export_share': 0.167, 'risk_score': 0.3, 'stability': 'Medium'},
                {'country': 'China', 'export_value': 20000000000, 'export_share': 0.133, 'risk_score': 0.6, 'stability': 'Low'},
                {'country': 'Japan', 'export_value': 15000000000, 'export_share': 0.1, 'risk_score': 0.2, 'stability': 'High'},
                {'country': 'Germany', 'export_value': 10000000000, 'export_share': 0.067, 'risk_score': 0.2, 'stability': 'High'}
            ]
        }