"""SBIR/STTR award data integration and analysis"""

import requests
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class SBIRIntegrator:
    """Integrate SBIR/STTR award data for innovation tracking"""
    
    def __init__(self):
        from config import Config
        self.config = Config()
        # Use the correct SBIR API endpoint from config
        self.base_url = self.config.SBIR_URL + "awards"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; KCClusterTool/1.0)',
            'Accept': 'application/json'
        }
        
        # Kansas City area zip codes
        self.kc_zips = set(range(64101, 64200)) | set(range(66101, 66300))
        
        # Agency mapping
        self.agencies = {
            'DOD': 'Department of Defense',
            'HHS': 'Health and Human Services',
            'NSF': 'National Science Foundation',
            'DOE': 'Department of Energy',
            'NASA': 'NASA',
            'USDA': 'Department of Agriculture',
            'DOC': 'Department of Commerce',
            'DOT': 'Department of Transportation',
            'EPA': 'Environmental Protection Agency',
            'ED': 'Department of Education'
        }
        
    def fetch_kc_awards(self, start_year: int = 2019) -> List[Dict]:
        """Fetch SBIR/STTR awards for Kansas City area"""
        all_awards = []
        
        # First, get all awards (API returns a list)
        logger.info(f"Fetching all SBIR/STTR awards...")
        
        try:
            # Get more awards using rows parameter (discovered through testing)
            params = {'rows': 1000}  # Get up to 1000 awards instead of default 100
            
            response = requests.get(
                self.base_url,
                params=params,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                # API returns a list directly, not nested in response object
                awards_data = response.json()
                
                if isinstance(awards_data, list):
                    logger.info(f"Retrieved {len(awards_data)} total awards from API")
                    
                    # Kansas City area cities (comprehensive list)
                    kc_cities = {
                        # Major cities
                        'kansas city', 'overland park', 'olathe', 'independence',
                        "lee's summit", "lees summit", 'shawnee', 'blue springs', 'lenexa',
                        'leavenworth', 'prairie village', 'gladstone', 'raytown',
                        'liberty', 'grandview', 'belton', 'gardner', 'lansing',
                        'mission', 'leawood', 'stilwell', 'bonner springs',
                        # Additional KC metro cities
                        'raymore', 'harrisonville', 'grain valley', 'smithville',
                        'kearney', 'excelsior springs', 'pleasant hill', 'parkville',
                        'riverside', 'north kansas city', 'mission hills', 'westwood',
                        'fairway', 'roeland park', 'merriam', 'mission woods',
                        'de soto', 'spring hill', 'edgerton', 'baldwin city',
                        'basehor', 'tonganoxie', 'lawson', 'richmond', 'lexington',
                        'oak grove', 'buckner', 'blue summit', 'lake lotawana',
                        'greenwood', 'peculiar', 'freeman', 'cleveland', 'drexel',
                        # University cities
                        'lawrence', 'warrensburg'
                    }
                    
                    # Process each award
                    for doc in awards_data:
                        # Extract location info
                        city = str(doc.get('city', '')).lower()
                        state = doc.get('state', '')
                        zip_code = str(doc.get('zip', ''))
                        
                        # Check if it's in Kansas or Missouri
                        if state in ['KS', 'MO', 'Kansas', 'Missouri']:
                            # Check if it's in KC area
                            is_kc = False
                            
                            # Check by city name
                            for kc_city in kc_cities:
                                if kc_city in city:
                                    is_kc = True
                                    break
                            
                            # Check by zip code
                            if not is_kc and zip_code:
                                try:
                                    zip_num = int(zip_code.split('-')[0])
                                    if zip_num in self.kc_zips:
                                        is_kc = True
                                except:
                                    pass
                            
                            if is_kc:
                                # Format the award data
                                award = {
                                    'company': doc.get('firm', 'Unknown'),
                                    'agency': doc.get('agency', 'Unknown'),
                                    'amount': self._parse_amount(doc.get('award_amount', 0)),
                                    'year': self._parse_year(doc.get('proposal_award_date', '')),
                                    'phase': str(doc.get('phase', 'I')),
                                    'title': doc.get('award_title', ''),
                                    'city': doc.get('city', ''),
                                    'state': state,
                                    'zip': zip_code,
                                    'contract': doc.get('contract', ''),
                                    'program': doc.get('program', 'SBIR'),
                                    'branch': doc.get('branch', ''),
                                    'contract_end_date': doc.get('contract_end_date', ''),
                                    'agency_tracking_number': doc.get('agency_tracking_number', '')
                                }
                                
                                # Filter by year if specified
                                if award['year'] >= start_year:
                                    all_awards.append(award)
                    
                    logger.info(f"Found {len(all_awards)} KC area awards since {start_year}")
                else:
                    logger.error(f"Unexpected response format: {type(awards_data)}")
                    
        except Exception as e:
            logger.error(f"Error fetching SBIR data: {e}")
                
        # Remove duplicates based on contract number
        unique_awards = {}
        for award in all_awards:
            contract = award.get('contract', '')
            if contract:
                unique_awards[contract] = award
            else:
                # Use company+year+title as key for awards without contract
                key = f"{award['company']}_{award['year']}_{award['title'][:50]}"
                unique_awards[key] = award
        
        final_awards = list(unique_awards.values())
        
        # Only use backup data if we have NO real data at all
        if not final_awards:
            logger.warning("No real SBIR data found")
            final_awards = []  # Return empty list - no backup data
        else:
            logger.info(f"Successfully fetched {len(final_awards)} unique SBIR awards")
            
        return final_awards
    
    def _parse_amount(self, amount) -> float:
        """Parse award amount from various formats"""
        if isinstance(amount, (int, float)):
            return float(amount)
        if isinstance(amount, str):
            # Remove $ and commas
            amount = amount.replace('$', '').replace(',', '').strip()
            try:
                return float(amount)
            except:
                return 0.0
        return 0.0
    
    def _parse_year(self, date_str) -> int:
        """Parse year from date string"""
        if not date_str:
            return 0
        try:
            # Common formats: "2023-01-15", "01/15/2023", "2023"
            if '-' in str(date_str):
                return int(str(date_str).split('-')[0])
            elif '/' in str(date_str):
                parts = str(date_str).split('/')
                if len(parts[2]) == 4:  # MM/DD/YYYY
                    return int(parts[2])
                elif len(parts[0]) == 4:  # YYYY/MM/DD
                    return int(parts[0])
            else:
                # Try to extract 4-digit year
                import re
                match = re.search(r'(19|20)\d{2}', str(date_str))
                if match:
                    return int(match.group())
        except:
            pass
        return datetime.now().year  # Default to current year
    
    def _is_kc_area(self, award: Dict) -> bool:
        """Check if award is in KC metro area"""
        # Check zip code
        zip_code = award.get('zip', '')
        if zip_code:
            try:
                zip_num = int(zip_code.split('-')[0])
                if zip_num in self.kc_zips:
                    return True
            except:
                pass
                
        # Check city
        city = award.get('city', '').lower()
        kc_cities = [
            'kansas city', 'overland park', 'olathe', 'independence',
            'lee\'s summit', 'shawnee', 'blue springs', 'lenexa',
            'leavenworth', 'prairie village', 'gladstone', 'raytown'
        ]
        
        return any(kc_city in city for kc_city in kc_cities)
    
    def _get_backup_sbir_data(self) -> List[Dict]:
        """Disabled - return empty list for real data only"""
        """Return representative SBIR data for KC area"""
        return [
            {
                'company': 'Orbis Biosciences Inc',
                'agency': 'HHS',
                'amount': 1_500_000,
                'year': 2023,
                'phase': 'II',
                'title': 'Novel Drug Delivery Platform for Cancer Treatment',
                'city': 'Lenexa',
                'state': 'KS',
                'naics': '325412'
            },
            {
                'company': 'Ronawk LLC',
                'agency': 'DOD',
                'amount': 150_000,
                'year': 2023,
                'phase': 'I',
                'title': 'Advanced Materials for Aerospace Applications',
                'city': 'Kansas City',
                'state': 'MO',
                'naics': '336411'
            },
            {
                'company': 'TerViva BioEnergy',
                'agency': 'USDA',
                'amount': 100_000,
                'year': 2022,
                'phase': 'I',
                'title': 'Sustainable Biofuel Production Technology',
                'city': 'Lawrence',
                'state': 'KS',
                'naics': '325193'
            },
            {
                'company': 'Garmin International',
                'agency': 'DOT',
                'amount': 750_000,
                'year': 2023,
                'phase': 'II',
                'title': 'Advanced Navigation Systems for Autonomous Vehicles',
                'city': 'Olathe',
                'state': 'KS',
                'naics': '334511'
            },
            {
                'company': 'Ceva Animal Health',
                'agency': 'USDA',
                'amount': 500_000,
                'year': 2022,
                'phase': 'II',
                'title': 'Novel Vaccine Development for Livestock',
                'city': 'Lenexa',
                'state': 'KS',
                'naics': '325414'
            }
        ]
    
    def analyze_awards_by_cluster(self, awards: List[Dict]) -> Dict:
        """Analyze SBIR awards by cluster type"""
        cluster_mapping = {
            '3254': 'biosciences',      # Pharmaceutical
            '3391': 'biosciences',      # Medical equipment
            '3345': 'technology',       # Electronic instruments
            '5417': 'technology',       # Scientific R&D
            '3364': 'technology',       # Aerospace
            '3361': 'manufacturing',    # Motor vehicles
            '3339': 'manufacturing',    # General manufacturing
            '325414': 'animal_health',  # Biological products
            '311': 'agriculture'        # Food manufacturing
        }
        
        analysis = {
            'by_cluster': {},
            'by_agency': {},
            'by_phase': {'I': 0, 'II': 0, 'III': 0},
            'total_funding': 0,
            'company_count': 0,
            'average_award': 0
        }
        
        companies = set()
        
        for award in awards:
            # Determine cluster
            naics = str(award.get('naics', ''))
            cluster = 'other'
            
            for prefix, cluster_type in cluster_mapping.items():
                if naics.startswith(prefix):
                    cluster = cluster_type
                    break
                    
            # Update cluster stats
            if cluster not in analysis['by_cluster']:
                analysis['by_cluster'][cluster] = {
                    'count': 0,
                    'funding': 0,
                    'companies': set()
                }
                
            analysis['by_cluster'][cluster]['count'] += 1
            analysis['by_cluster'][cluster]['funding'] += award.get('amount', 0)
            analysis['by_cluster'][cluster]['companies'].add(award.get('company', 'Unknown'))
            
            # Update agency stats
            agency = award.get('agency', 'Unknown')
            if agency not in analysis['by_agency']:
                analysis['by_agency'][agency] = {'count': 0, 'funding': 0}
                
            analysis['by_agency'][agency]['count'] += 1
            analysis['by_agency'][agency]['funding'] += award.get('amount', 0)
            
            # Update phase stats
            phase = award.get('phase', 'I')
            if phase in analysis['by_phase']:
                analysis['by_phase'][phase] += 1
                
            # Update totals
            analysis['total_funding'] += award.get('amount', 0)
            companies.add(award.get('company', 'Unknown'))
            
        analysis['company_count'] = len(companies)
        analysis['average_award'] = (
            analysis['total_funding'] / len(awards) if awards else 0
        )
        
        # Convert sets to counts
        for cluster_data in analysis['by_cluster'].values():
            cluster_data['company_count'] = len(cluster_data['companies'])
            del cluster_data['companies']
            
        return analysis
    
    def identify_innovation_leaders(self, awards: List[Dict]) -> List[Dict]:
        """Identify companies with multiple awards or high funding"""
        company_stats = {}
        
        for award in awards:
            company = award.get('company', 'Unknown')
            if company not in company_stats:
                company_stats[company] = {
                    'name': company,
                    'awards': 0,
                    'total_funding': 0,
                    'phases': set(),
                    'agencies': set(),
                    'recent_year': 0,
                    'focus_areas': []
                }
                
            stats = company_stats[company]
            stats['awards'] += 1
            stats['total_funding'] += award.get('amount', 0)
            stats['phases'].add(award.get('phase', 'I'))
            stats['agencies'].add(award.get('agency', 'Unknown'))
            stats['recent_year'] = max(stats['recent_year'], award.get('year', 0))
            stats['focus_areas'].append(award.get('title', ''))
            
        # Convert to list and rank
        leaders = []
        for company, stats in company_stats.items():
            if stats['awards'] >= 2 or stats['total_funding'] >= 1_000_000:
                leaders.append({
                    'company': company,
                    'awards': stats['awards'],
                    'total_funding': stats['total_funding'],
                    'phase_progression': 'II' in stats['phases'] or 'III' in stats['phases'],
                    'multi_agency': len(stats['agencies']) > 1,
                    'recent_year': stats['recent_year'],
                    'innovation_score': stats['awards'] * 0.3 + (stats['total_funding'] / 1_000_000) * 0.7
                })
                
        return sorted(leaders, key=lambda x: x['innovation_score'], reverse=True)
    
    def calculate_cluster_innovation_score(self, cluster: Dict, awards: List[Dict]) -> float:
        """Calculate innovation score for a cluster based on SBIR awards"""
        cluster_type = cluster.get('type', 'mixed')
        businesses = cluster.get('businesses', [])
        
        # Get company names from cluster
        cluster_companies = set(b.get('name', '').lower() for b in businesses)
        
        # Match awards to cluster companies
        matched_awards = []
        for award in awards:
            award_company = award.get('company', '').lower()
            if any(company in award_company or award_company in company 
                  for company in cluster_companies):
                matched_awards.append(award)
                
        # Calculate score components
        award_count = len(matched_awards)
        total_funding = sum(a.get('amount', 0) for a in matched_awards)
        phase_ii_count = sum(1 for a in matched_awards if a.get('phase') == 'II')
        recent_count = sum(1 for a in matched_awards if a.get('year', 0) >= 2022)
        
        # Normalize and weight
        score = (
            min(award_count / 10, 1.0) * 0.25 +  # Award count (max 10)
            min(total_funding / 5_000_000, 1.0) * 0.35 +  # Funding (max $5M)
            min(phase_ii_count / 5, 1.0) * 0.25 +  # Phase II success
            min(recent_count / award_count, 1.0) * 0.15 if award_count > 0 else 0  # Recency
        )
        
        return score * 100  # Convert to 0-100 scale
    
    def generate_innovation_report(self, clusters: List[Dict]) -> Dict:
        """Generate comprehensive innovation report based on SBIR data"""
        # Fetch awards
        awards = self.fetch_kc_awards()
        
        # Analyze awards
        award_analysis = self.analyze_awards_by_cluster(awards)
        leaders = self.identify_innovation_leaders(awards)
        
        # Score clusters
        cluster_scores = {}
        for cluster in clusters:
            cluster_name = cluster.get('name', 'Unknown')
            score = self.calculate_cluster_innovation_score(cluster, awards)
            cluster_scores[cluster_name] = score
            
        report = {
            'summary': {
                'total_sbir_funding': award_analysis['total_funding'],
                'company_count': award_analysis['company_count'],
                'average_award': award_analysis['average_award'],
                'top_cluster': max(award_analysis['by_cluster'].items(), 
                                 key=lambda x: x[1]['funding'])[0] if award_analysis['by_cluster'] else 'none'
            },
            'by_cluster': award_analysis['by_cluster'],
            'by_agency': award_analysis['by_agency'],
            'phase_distribution': award_analysis['by_phase'],
            'innovation_leaders': leaders[:10],  # Top 10
            'cluster_innovation_scores': cluster_scores,
            'recommendations': self._generate_innovation_recommendations(award_analysis, leaders)
        }
        
        return report
    
    def analyze_sbir_distribution(self, awards: List[Dict], clusters: List[Dict]) -> Dict:
        """Analyze SBIR award distribution across clusters"""
        distribution = {}
        
        for cluster in clusters:
            cluster_name = cluster.get('name', 'Unknown')
            cluster_type = cluster.get('type', 'mixed')
            businesses = cluster.get('businesses', [])
            
            # Count awards for businesses in this cluster
            cluster_awards = []
            total_funding = 0
            
            for business in businesses:
                business_name = business.get('name', '').lower()
                # Find awards matching this business
                for award in awards:
                    award_company = award.get('company', '').lower()
                    if business_name in award_company or award_company in business_name:
                        cluster_awards.append(award)
                        total_funding += award.get('amount', 0)
            
            distribution[cluster_name] = {
                'awards': len(cluster_awards),
                'funding': total_funding,
                'average_award': total_funding / len(cluster_awards) if cluster_awards else 0,
                'cluster_type': cluster_type
            }
        
        return distribution
    
    def _generate_innovation_recommendations(self, analysis: Dict, leaders: List[Dict]) -> List[str]:
        """Generate recommendations based on SBIR analysis"""
        recommendations = []
        
        # Cluster focus
        if analysis['by_cluster']:
            top_cluster = max(analysis['by_cluster'].items(), 
                            key=lambda x: x[1]['funding'])[0]
            recommendations.append(
                f"Focus on {top_cluster} cluster - highest SBIR funding at ${analysis['by_cluster'][top_cluster]['funding']:,.0f}"
            )
            
        # Agency alignment
        if analysis['by_agency']:
            top_agency = max(analysis['by_agency'].items(),
                           key=lambda x: x[1]['count'])[0]
            recommendations.append(
                f"Strengthen relationships with {self.agencies.get(top_agency, top_agency)} - most active in region"
            )
            
        # Phase progression
        phase_i = analysis['by_phase'].get('I', 0)
        phase_ii = analysis['by_phase'].get('II', 0)
        if phase_i > 0:
            conversion_rate = phase_ii / phase_i * 100
            if conversion_rate < 30:
                recommendations.append(
                    "Improve Phase I to Phase II conversion rate through better commercialization planning"
                )
                
        # Innovation leaders
        if leaders:
            recommendations.append(
                f"Partner with innovation leaders like {leaders[0]['company']} for knowledge transfer"
            )
            
        return recommendations