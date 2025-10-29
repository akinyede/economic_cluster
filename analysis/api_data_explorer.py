"""
API Data Explorer - Direct API testing to understand data structures
This script tests all APIs directly without cache to understand the actual data structure.
"""
import requests
import json
import time
import logging
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIDataExplorer:
    def __init__(self):
        # API configurations from config.py
        self.apis = {
            'BLS': {
                'url': 'https://api.bls.gov/publicAPI/v2/timeseries/data/',
                'key': 'ff295e0e87674bedb3898e848e8225f1',
                'description': 'Bureau of Labor Statistics - Employment, wages, industry data'
            },
            'USPTO': {
                'url': 'https://search.patentsview.org/api/v1/patent/',
                'key': 'UrEcFJSy.AJSw9jFKqJUvYEIwSoaR0fU5HI1sN1RL',
                'description': 'Patent and Trademark Office - Innovation metrics'
            },
            'FRED': {
                'url': 'https://api.stlouisfed.org/fred/series/observations',
                'key': '1a764a8681e49e86e164e1d01de7e203',
                'description': 'Federal Reserve Economic Data - Market conditions'
            },
            'EIA': {
                'url': 'https://api.eia.gov/v2/',
                'key': 'rXwPRWA7mmLqRYduQ0k1Mdoa0loH4Pw2Im1Y8wDd',
                'description': 'Energy Information Administration - Energy infrastructure'
            },
            'CENSUS': {
                'url': 'https://api.census.gov/data/',
                'key': '993bd45c6f9eaecc26d7f6a9e074da8ae4ab078f',
                'description': 'Census Bureau - Demographics, business patterns'
            }
        }
        
    def test_bls_api(self) -> Dict:
        """Test BLS API for Kansas City MSA employment data"""
        logger.info("="*50)
        logger.info("Testing BLS API...")
        
        # Kansas City MSA series IDs
        series_ids = [
            'LAUMT292814000000003',  # KC MSA unemployment rate
            'SMS29281400000000001',  # KC MSA total nonfarm employment
            'CUUR0300SA0',           # CPI for Midwest urban
        ]
        
        headers = {'Content-type': 'application/json'}
        data = json.dumps({
            "seriesid": series_ids,
            "startyear": "2023",
            "endyear": "2024",
            "registrationkey": self.apis['BLS']['key']
        })
        
        try:
            response = requests.post(
                self.apis['BLS']['url'], 
                data=data, 
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"BLS API Status: {result['status']}")
                
                if result['status'] == 'REQUEST_SUCCEEDED':
                    logger.info(f"Number of series returned: {len(result.get('Results', {}).get('series', []))}")
                    
                    # Display sample data structure
                    for series in result['Results']['series'][:1]:
                        logger.info(f"\nSeries ID: {series['seriesID']}")
                        logger.info(f"Data points: {len(series.get('data', []))}")
                        if series.get('data'):
                            logger.info(f"Sample data point: {json.dumps(series['data'][0], indent=2)}")
                
                return result
            else:
                logger.error(f"BLS API Error: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"BLS API Exception: {str(e)}")
            return {}
    
    def test_uspto_api(self) -> Dict:
        """Test USPTO Patent API for Kansas City area"""
        logger.info("="*50)
        logger.info("Testing USPTO API...")
        
        # Search for patents in Kansas City area
        params = {
            'q': {"city": "Kansas City"},
            'o': json.dumps({
                "page": 1,
                "size": 5
            }),
            's': json.dumps([
                {"patent_date": "desc"}
            ])
        }
        
        headers = {
            'X-Api-Key': self.apis['USPTO']['key']
        }
        
        try:
            response = requests.get(
                self.apis['USPTO']['url'],
                params=params,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"USPTO API returned {result.get('total_hits', 0)} total patents")
                
                # Display sample patent structure
                if result.get('patents'):
                    logger.info(f"\nSample patent structure:")
                    sample = result['patents'][0]
                    logger.info(f"Patent Number: {sample.get('patent_number')}")
                    logger.info(f"Title: {sample.get('patent_title')}")
                    logger.info(f"Date: {sample.get('patent_date')}")
                    logger.info(f"Assignee: {sample.get('assignee_organization')}")
                    logger.info(f"Available fields: {list(sample.keys())}")
                
                return result
            else:
                logger.error(f"USPTO API Error: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"USPTO API Exception: {str(e)}")
            return {}
    
    def test_fred_api(self) -> Dict:
        """Test FRED API for economic indicators"""
        logger.info("="*50)
        logger.info("Testing FRED API...")
        
        # KC Fed Manufacturing Index
        series_id = 'KCLMCIM'
        
        params = {
            'series_id': series_id,
            'api_key': self.apis['FRED']['key'],
            'file_type': 'json',
            'limit': 5,
            'sort_order': 'desc'
        }
        
        try:
            response = requests.get(
                self.apis['FRED']['url'],
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"FRED API returned {len(result.get('observations', []))} observations")
                
                if result.get('observations'):
                    logger.info(f"\nSample observation:")
                    sample = result['observations'][0]
                    logger.info(f"Date: {sample.get('date')}")
                    logger.info(f"Value: {sample.get('value')}")
                    
                return result
            else:
                logger.error(f"FRED API Error: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"FRED API Exception: {str(e)}")
            return {}
    
    def test_census_api(self) -> Dict:
        """Test Census API for business patterns"""
        logger.info("="*50)
        logger.info("Testing Census API...")
        
        # County Business Patterns for Jackson County, MO (Kansas City)
        year = '2021'
        dataset = f'{year}/cbp'
        
        params = {
            'get': 'NAICS2017,ESTAB,EMP,PAYANN',
            'for': 'county:095',
            'in': 'state:29',  # Missouri
            'NAICS2017': '31-33',  # Manufacturing
            'key': self.apis['CENSUS']['key']
        }
        
        try:
            url = f"{self.apis['CENSUS']['url']}{dataset}"
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Census API returned {len(result)} records")
                
                if len(result) > 1:
                    headers = result[0]
                    data = result[1]
                    logger.info(f"\nData structure:")
                    for i, header in enumerate(headers):
                        logger.info(f"{header}: {data[i]}")
                
                return result
            else:
                logger.error(f"Census API Error: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Census API Exception: {str(e)}")
            return {}
    
    def test_sbir_api(self) -> Dict:
        """Test SBIR API for innovation funding"""
        logger.info("="*50)
        logger.info("Testing SBIR API...")
        
        # Note: SBIR API documentation is limited, using common endpoints
        base_url = "https://api.sbir.gov/v2/award"
        
        params = {
            'state': 'KS',
            'rows': 5
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"SBIR API Status: Success")
                logger.info(f"Sample response: {json.dumps(result, indent=2)[:500]}...")
                return result
            else:
                logger.error(f"SBIR API Error: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"SBIR API Exception: {str(e)}")
            return {}
    
    def analyze_data_structures(self):
        """Analyze all API data structures and create summary"""
        logger.info("\n" + "="*50)
        logger.info("ANALYZING API DATA STRUCTURES")
        logger.info("="*50)
        
        results = {}
        
        # Test each API
        results['BLS'] = self.test_bls_api()
        time.sleep(1)  # Rate limiting
        
        results['USPTO'] = self.test_uspto_api()
        time.sleep(1)
        
        results['FRED'] = self.test_fred_api()
        time.sleep(1)
        
        results['Census'] = self.test_census_api()
        time.sleep(1)
        
        results['SBIR'] = self.test_sbir_api()
        
        # Summarize findings
        self._summarize_findings(results)
        
        return results
    
    def _summarize_findings(self, results: Dict):
        """Summarize API data structure findings"""
        logger.info("\n" + "="*50)
        logger.info("API DATA STRUCTURE SUMMARY")
        logger.info("="*50)
        
        logger.info("\n1. BLS (Bureau of Labor Statistics):")
        logger.info("   - Provides time series data for employment, wages, unemployment")
        logger.info("   - Data structure: series -> data points with year, period, value")
        logger.info("   - Key for clusters: Industry employment trends, wage data")
        
        logger.info("\n2. USPTO (Patent Office):")
        logger.info("   - Provides patent data with assignee, location, dates")
        logger.info("   - Rich metadata: CPC codes, inventors, citations")
        logger.info("   - Key for clusters: Innovation metrics by company/location")
        
        logger.info("\n3. FRED (Federal Reserve):")
        logger.info("   - Economic indicators and time series")
        logger.info("   - Simple structure: date-value pairs")
        logger.info("   - Key for clusters: Economic conditions, market trends")
        
        logger.info("\n4. Census:")
        logger.info("   - County Business Patterns with establishment counts, employment")
        logger.info("   - Detailed NAICS breakdowns available")
        logger.info("   - Key for clusters: Industry concentration, business density")
        
        logger.info("\n5. SBIR:")
        logger.info("   - Small business innovation research awards")
        logger.info("   - Company-level innovation funding data")
        logger.info("   - Key for clusters: R&D activity, emerging technologies")
        
    def extract_cluster_relevant_features(self):
        """Extract features relevant for cluster prediction"""
        logger.info("\n" + "="*50)
        logger.info("RECOMMENDED FEATURES FOR ML MODELS")
        logger.info("="*50)
        
        features = {
            'Business Level Features': [
                'company_name',
                'naics_code',
                'employees',
                'revenue_estimate',
                'year_established',
                'patent_count',
                'patent_quality_score',  # Based on citations
                'sbir_awards',
                'sbir_total_funding',
                'location_county',
                'location_msa'
            ],
            
            'Industry Level Features': [
                'industry_employment_trend',  # From BLS
                'industry_wage_trend',       # From BLS
                'industry_growth_rate',      # From BLS
                'industry_concentration',    # From Census CBP
                'industry_patent_intensity', # Patents per employee
                'industry_investment_trend'  # From FRED/market data
            ],
            
            'Regional Features': [
                'county_business_density',   # From Census
                'county_employment_rate',    # From BLS
                'regional_gdp_growth',       # From BEA/FRED
                'infrastructure_score',      # Composite metric
                'workforce_education_index', # From Census ACS
                'university_research_output' # Patents from universities
            ],
            
            'Cluster Prediction Features': [
                'business_count',
                'total_employees', 
                'total_revenue',
                'avg_business_age',
                'strategic_alignment_score',  # How well businesses fit together
                'innovation_density',         # Patents/SBIR per business
                'market_opportunity_score',   # Based on industry growth
                'competitive_advantage_score' # Based on patents, market position
            ]
        }
        
        for category, feature_list in features.items():
            logger.info(f"\n{category}:")
            for feature in feature_list:
                logger.info(f"  - {feature}")
        
        return features


def main():
    """Main execution"""
    explorer = APIDataExplorer()
    
    # Test all APIs
    results = explorer.analyze_data_structures()
    
    # Extract recommended features
    features = explorer.extract_cluster_relevant_features()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f'api_analysis_{timestamp}.json', 'w') as f:
        json.dump({
            'api_results': {k: str(v)[:1000] for k, v in results.items()},
            'recommended_features': features,
            'timestamp': timestamp
        }, f, indent=2)
    
    logger.info(f"\nAnalysis complete. Results saved to api_analysis_{timestamp}.json")


if __name__ == "__main__":
    main()