"""Real-time market monitoring module for KC Cluster Prediction Tool"""
import requests
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketMonitor:
    """Monitor real-time market conditions affecting clusters"""
    
    def __init__(self):
        self.config = Config()
        self.fred_api_key = self.config.FRED_API_KEY
        self.eia_api_key = self.config.EIA_API_KEY
        self.alpha_vantage_key = self.config.ALPHA_VANTAGE_API_KEY
        
    def fetch_economic_indicators(self) -> Dict:
        """Fetch key economic indicators from FRED"""
        indicators = {}
        
        # FRED series IDs for key indicators
        series_ids = {
            'gdp_growth': 'A191RL1Q225SBEA',  # Real GDP growth
            'unemployment': 'UNRATE',          # Unemployment rate
            'inflation': 'CPIAUCSL',           # CPI inflation
            'interest_rate': 'DFF',            # Federal funds rate
            'industrial_production': 'INDPRO',  # Industrial production index
            'business_confidence': 'BSCICP03USM665S',  # Business confidence
            'consumer_sentiment': 'UMCSENT',   # Consumer sentiment
        }
        
        if not self.fred_api_key:
            logger.warning("FRED API key not found, using simulated data")
            return self._get_simulated_economic_data()
        
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        for name, series_id in series_ids.items():
            try:
                params = {
                    'series_id': series_id,
                    'api_key': self.fred_api_key,
                    'file_type': 'json',
                    'limit': 1,
                    'sort_order': 'desc'
                }
                
                response = requests.get(base_url, params=params, timeout=self.config.API_TIMEOUT_SHORT)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'observations' in data and data['observations']:
                        latest = data['observations'][0]
                        indicators[name] = {
                            'value': float(latest['value']),
                            'date': latest['date'],
                            'series_id': series_id
                        }
                    
            except Exception as e:
                logger.error(f"Error fetching {name}: {e}")
                
        return indicators
    
    def fetch_commodity_prices(self) -> Dict:
        """Fetch commodity prices from EIA v2 API"""
        commodities = {}
        
        if not self.eia_api_key:
            logger.warning("EIA API key not found, using simulated data")
            return self._get_simulated_commodity_data()
        
        # EIA v2 API endpoints and parameters
        eia_queries = {
            'diesel': {
                'route': 'petroleum/pri/gnd/data/',
                'params': {
                    'api_key': self.eia_api_key,
                    'frequency': 'weekly',
                    'data[0]': 'value',
                    'facets[series][]': 'EMD_EPD2D_PTE_NUS_DPG',
                    'sort[0][column]': 'period',
                    'sort[0][direction]': 'desc',
                    'offset': 0,
                    'length': 1
                },
                'units': 'USD/gallon'
            },
            'crude_oil': {
                'route': 'petroleum/pri/spt/data/',
                'params': {
                    'api_key': self.eia_api_key,
                    'frequency': 'daily',
                    'data[0]': 'value',
                    'facets[series][]': 'RWTC',
                    'sort[0][column]': 'period',
                    'sort[0][direction]': 'desc',
                    'offset': 0,
                    'length': 1
                },
                'units': 'USD/barrel'
            },
            'natural_gas': {
                'route': 'natural-gas/pri/fut/data/',
                'params': {
                    'api_key': self.eia_api_key,
                    'frequency': 'daily',
                    'data[0]': 'value',
                    'sort[0][column]': 'period',
                    'sort[0][direction]': 'desc',
                    'offset': 0,
                    'length': 1
                },
                'units': 'USD/MMBtu'
            },
            'electricity': {
                'route': 'electricity/retail-sales/data/',
                'params': {
                    'api_key': self.eia_api_key,
                    'frequency': 'monthly',
                    'data[0]': 'price',
                    'facets[sectorid][]': 'ALL',
                    'facets[stateid][]': 'US',
                    'sort[0][column]': 'period',
                    'sort[0][direction]': 'desc',
                    'offset': 0,
                    'length': 1
                },
                'units': 'cents/kWh'
            }
        }
        
        base_url = "https://api.eia.gov/v2/"
        
        # Prioritize most important commodities and use longer timeout
        priority_order = ['diesel', 'crude_oil', 'electricity', 'natural_gas']
        
        for name in priority_order:
            if name not in eia_queries:
                continue
                
            query = eia_queries[name]
            try:
                url = base_url + query['route']
                logger.info(f"Fetching {name} from EIA (this may take 30-60 seconds)...")
                
                # Use extended timeout for EIA
                response = requests.get(url, params=query['params'], timeout=self.config.API_TIMEOUT_EXTENDED)
                
                if response.status_code == 200:
                    data = response.json()
                    if 'response' in data and 'data' in data['response'] and data['response']['data']:
                        latest = data['response']['data'][0]
                        value = latest.get('value') or latest.get('price', 0)
                        commodities[name] = {
                            'value': float(value),
                            'date': latest.get('period', datetime.now().strftime('%Y-%m-%d')),
                            'units': query['units']
                        }
                        logger.info(f"✓ {name}: ${value} {query['units']}")
                    else:
                        logger.warning(f"No data returned for {name}")
                        # Use fallback value
                        commodities[name] = self._get_simulated_commodity_data().get(name, {})
                else:
                    logger.error(f"EIA API error for {name}: HTTP {response.status_code}")
                    # Use fallback value
                    commodities[name] = self._get_simulated_commodity_data().get(name, {})
                        
            except requests.exceptions.Timeout:
                logger.error(f"Timeout fetching {name} after {self.config.API_TIMEOUT_EXTENDED}s - using cached value")
                commodities[name] = self._get_simulated_commodity_data().get(name, {})
            except Exception as e:
                logger.error(f"Error fetching {name}: {e}")
                commodities[name] = self._get_simulated_commodity_data().get(name, {})
                
        return commodities
    
    def fetch_industry_trends(self) -> Dict:
        """Fetch industry-specific market trends"""
        trends = {}
        
        # Cluster-specific indicators from FRED
        cluster_indicators = {
            'logistics': {
                'truck_tonnage': 'TRUCKD11',       # Truck tonnage index
                'rail_freight': 'RAILFRTCARLOADSD11', # Rail freight carloads
            },
            'biosciences': {
                'healthcare_employment': 'CES6562000001', # Healthcare employment
            },
            'technology': {
                'tech_employment': 'CES5051700001', # Tech sector employment
            },
            'manufacturing': {
                'manufacturing_production': 'IPMAN',   # Manufacturing production index
                'capacity_utilization': 'TCU',         # Capacity utilization
                'durable_goods': 'DGORDER',           # Durable goods orders
            }
        }
        
        # Fetch FRED indicators
        for cluster, indicators in cluster_indicators.items():
            trends[cluster] = {}
            for name, series_id in indicators.items():
                if series_id:
                    value = self._fetch_fred_series(series_id)
                    if value:
                        trends[cluster][name] = value
        
        # Add sector performance from Alpha Vantage if available
        if self.alpha_vantage_key:
            sector_data = self._fetch_sector_performance()
            if sector_data:
                # Map sectors to clusters
                trends['technology']['sector_performance'] = sector_data.get('Information Technology', 0)
                trends['biosciences']['sector_performance'] = sector_data.get('Health Care', 0)
                trends['manufacturing']['sector_performance'] = sector_data.get('Industrials', 0)
                trends['logistics']['sector_performance'] = sector_data.get('Industrials', 0)
                        
        return trends
    
    def _fetch_sector_performance(self) -> Optional[Dict]:
        """Fetch sector performance from Alpha Vantage"""
        try:
            # Try the SECTOR function instead
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'SECTOR',
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params, timeout=self.config.API_TIMEOUT_SHORT)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for different response formats
                if 'Rank A: Real-Time Performance' in data:
                    performances = {}
                    for sector, perf in data['Rank A: Real-Time Performance'].items():
                        # Convert percentage string to float
                        try:
                            performances[sector] = float(perf.strip('%'))
                        except:
                            performances[sector] = 0.0
                    return performances
                elif 'Meta Data' in data:
                    # If API structure changed, return default values
                    logger.warning("Alpha Vantage API structure changed, using defaults")
                    return {
                        'Information Technology': 0.5,
                        'Health Care': 0.3,
                        'Industrials': 0.2
                    }
                else:
                    # Log the response for debugging
                    logger.debug(f"Unexpected Alpha Vantage response: {list(data.keys())}")
                    
        except Exception as e:
            logger.error(f"Error fetching sector performance: {e}")
            
        # Return default values if API fails
        return {
            'Information Technology': 0.0,
            'Health Care': 0.0,
            'Industrials': 0.0
        }
    
    def _fetch_fred_series(self, series_id: str) -> Optional[Dict]:
        """Helper to fetch a single FRED series"""
        if not self.fred_api_key:
            return None
            
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'limit': 1,
                'sort_order': 'desc'
            }
            
            response = requests.get(url, params=params, timeout=self.config.API_TIMEOUT_SHORT)
            
            if response.status_code == 200:
                data = response.json()
                if 'observations' in data and data['observations']:
                    latest = data['observations'][0]
                    return {
                        'value': float(latest['value']),
                        'date': latest['date']
                    }
                    
        except Exception as e:
            logger.error(f"Error fetching FRED series {series_id}: {e}")
            
        return None
    
    def enhance_market_data(self, existing_market_data: Dict) -> Dict:
        """Enhance existing market data with real-time indicators"""
        # Keep existing growth rates
        enhanced_data = existing_market_data.copy()
        
        # Add real-time indicators
        enhanced_data['economic_indicators'] = self.fetch_economic_indicators()
        enhanced_data['commodity_prices'] = self.fetch_commodity_prices()
        enhanced_data['industry_trends'] = self.fetch_industry_trends()
        enhanced_data['timestamp'] = datetime.now().isoformat()
        
        # Update growth rates based on real-time data if available
        if enhanced_data['economic_indicators']:
            # Adjust growth rates based on economic conditions
            gdp_growth = enhanced_data['economic_indicators'].get('gdp_growth', {}).get('value', 2.5)
            adjustment_factor = gdp_growth / 2.5  # Baseline GDP growth
            
            # Apply adjustments to existing growth rates
            for sector in ['logistics_growth', 'biotech_growth', 'manufacturing_growth', 'tech_growth']:
                if sector in enhanced_data:
                    for key in enhanced_data[sector]:
                        if isinstance(enhanced_data[sector][key], (int, float)):
                            enhanced_data[sector][key] *= adjustment_factor
        
        # Calculate market scores for each cluster
        enhanced_data['market_scores'] = {}
        for cluster_type in ['logistics', 'biosciences', 'technology', 'manufacturing']:
            enhanced_data['market_scores'][cluster_type] = self.calculate_market_score(
                cluster_type, enhanced_data
            )
        
        # Generate insights
        enhanced_data['insights'] = self.get_market_insights(enhanced_data)
        
        return enhanced_data
    
    def calculate_market_score(self, cluster_type: str, market_data: Dict) -> float:
        """Calculate market favorability score for a cluster type"""
        
        # Base scores by market conditions
        economic_indicators = market_data.get('economic_indicators', {})
        commodity_prices = market_data.get('commodity_prices', {})
        industry_trends = market_data.get('industry_trends', {}).get(cluster_type, {})
        
        score = 0.5  # Neutral baseline
        
        # Economic conditions impact
        if 'gdp_growth' in economic_indicators:
            gdp = economic_indicators['gdp_growth']['value']
            if gdp > 3.0:
                score += 0.1
            elif gdp < 1.0:
                score -= 0.1
                
        if 'unemployment' in economic_indicators:
            unemployment = economic_indicators['unemployment']['value']
            if unemployment < 4.0:
                score += 0.1
            elif unemployment > 6.0:
                score -= 0.1
                
        # Cluster-specific adjustments
        if cluster_type == 'logistics':
            # Fuel prices impact
            if 'diesel' in commodity_prices:
                diesel = commodity_prices['diesel']['value']
                if diesel < 3.0:
                    score += 0.15
                elif diesel > 4.0:
                    score -= 0.15
                    
            # Freight demand
            if 'truck_tonnage' in industry_trends:
                tonnage = industry_trends['truck_tonnage']['value']
                if tonnage > 110:
                    score += 0.1
                    
        elif cluster_type == 'biosciences':
            # Healthcare employment trends
            if 'healthcare_employment' in industry_trends:
                employment = industry_trends['healthcare_employment']['value']
                if employment > 100:  # Index above 100 indicates growth
                    score += 0.15
                    
        elif cluster_type == 'technology':
            # Tech employment trends
            if 'tech_employment' in industry_trends:
                employment = industry_trends['tech_employment']['value']
                if employment > 100:  # Index above 100
                    score += 0.1
                
        elif cluster_type == 'manufacturing':
            # Manufacturing production
            if 'manufacturing_production' in industry_trends:
                production = industry_trends['manufacturing_production']['value']
                if production > 105:
                    score += 0.15
                elif production < 95:
                    score -= 0.15
                    
        return max(0.0, min(1.0, score))
    
    def get_market_insights(self, market_data: Dict) -> List[str]:
        """Generate actionable insights from market data"""
        insights = []
        
        economic_indicators = market_data.get('economic_indicators', {})
        commodity_prices = market_data.get('commodity_prices', {})
        
        # Economic insights
        if 'gdp_growth' in economic_indicators:
            gdp = economic_indicators['gdp_growth']['value']
            if gdp > 3.0:
                insights.append(f"Strong GDP growth ({gdp:.1f}%) indicates favorable expansion conditions")
            elif gdp < 1.0:
                insights.append(f"Low GDP growth ({gdp:.1f}%) suggests cautious expansion approach")
                
        # Commodity insights
        if 'diesel' in commodity_prices:
            diesel = commodity_prices['diesel']['value']
            if diesel > 4.0:
                insights.append(f"High diesel prices (${diesel:.2f}/gal) may impact logistics costs")
                
        # Interest rate insights
        if 'interest_rate' in economic_indicators:
            rate = economic_indicators['interest_rate']['value']
            if rate > 5.0:
                insights.append(f"High interest rates ({rate:.1f}%) increase financing costs")
            elif rate < 2.0:
                insights.append(f"Low interest rates ({rate:.1f}%) favor capital investments")
                
        return insights
    
    def _get_simulated_economic_data(self) -> Dict:
        """Return simulated economic data for testing"""
        return {
            'gdp_growth': {'value': 2.5, 'date': datetime.now().strftime('%Y-%m-%d')},
            'unemployment': {'value': 3.7, 'date': datetime.now().strftime('%Y-%m-%d')},
            'inflation': {'value': 2.3, 'date': datetime.now().strftime('%Y-%m-%d')},
            'interest_rate': {'value': 5.25, 'date': datetime.now().strftime('%Y-%m-%d')},
            'industrial_production': {'value': 105.2, 'date': datetime.now().strftime('%Y-%m-%d')},
            'business_confidence': {'value': 58.5, 'date': datetime.now().strftime('%Y-%m-%d')},
            'consumer_sentiment': {'value': 68.1, 'date': datetime.now().strftime('%Y-%m-%d')}
        }
    
    def _get_simulated_commodity_data(self) -> Dict:
        """Return simulated commodity data for testing"""
        return {
            'crude_oil': {'value': 75.50, 'date': datetime.now().strftime('%Y-%m-%d'), 'units': 'USD/barrel'},
            'natural_gas': {'value': 2.85, 'date': datetime.now().strftime('%Y-%m-%d'), 'units': 'USD/MMBtu'},
            'diesel': {'value': 3.95, 'date': datetime.now().strftime('%Y-%m-%d'), 'units': 'USD/gallon'},
            'electricity': {'value': 0.105, 'date': datetime.now().strftime('%Y-%m-%d'), 'units': 'USD/kWh'}
        }
    
    def fetch_all_market_data(self) -> Dict:
        """Fetch all market monitoring data"""
        logger.info("Fetching market monitoring data...")
        
        market_data = {
            'timestamp': datetime.now().isoformat(),
            'economic_indicators': {},
            'commodity_prices': {},
            'industry_trends': {},
            'market_scores': {},
            'insights': []
        }
        
        # Fetch data in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.fetch_economic_indicators): 'economic_indicators',
                executor.submit(self.fetch_commodity_prices): 'commodity_prices',
                executor.submit(self.fetch_industry_trends): 'industry_trends'
            }
            
            for future in as_completed(futures):
                data_type = futures[future]
                try:
                    result = future.result()
                    market_data[data_type] = result
                except Exception as e:
                    logger.error(f"Error fetching {data_type}: {e}")
        
        # Calculate market scores for each cluster type
        for cluster_type in ['logistics', 'biosciences', 'technology', 'manufacturing']:
            market_data['market_scores'][cluster_type] = self.calculate_market_score(
                cluster_type, market_data
            )
        
        # Generate insights
        market_data['insights'] = self.get_market_insights(market_data)
        
        logger.info("Market monitoring data fetch complete")
        return market_data