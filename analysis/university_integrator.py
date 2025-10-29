"""University research and innovation integration"""

import requests
import pandas as pd
from typing import Dict, List
import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class UniversityIntegrator:
    """Integrate university research, patents, and workforce development"""
    
    def __init__(self):
        self.universities = {
            'ku': {
                'name': 'University of Kansas',
                'location': {'lat': 38.9543, 'lon': -95.2558},
                'counties': ['Douglas', 'Johnson'],  # Main and medical campuses
                'strengths': ['biosciences', 'engineering', 'pharmacy'],
                'research_centers': [
                    'Bioengineering Research Center',
                    'Center for Drug Discovery & Innovation',
                    'Transportation Research Institute'
                ]
            },
            'ksu': {
                'name': 'Kansas State University',
                'location': {'lat': 39.1836, 'lon': -96.5717},
                'counties': ['Riley'],
                'strengths': ['agriculture', 'veterinary', 'engineering'],
                'research_centers': [
                    'Biosecurity Research Institute',
                    'Center for Animal Health',
                    'Advanced Manufacturing Institute'
                ]
            },
            'umkc': {
                'name': 'University of Missouri-Kansas City',
                'location': {'lat': 39.0337, 'lon': -94.5786},
                'counties': ['Jackson'],
                'strengths': ['medicine', 'business', 'computing'],
                'research_centers': [
                    'School of Medicine',
                    'Institute for Entrepreneurship',
                    'UMKC Innovation Center'
                ]
            },
            'mst': {
                'name': 'Missouri S&T',
                'location': {'lat': 37.9537, 'lon': -91.7756},
                'counties': ['Phelps'],
                'strengths': ['engineering', 'materials', 'computing'],
                'research_centers': [
                    'Materials Research Center',
                    'Intelligent Systems Center',
                    'Center for Infrastructure Engineering'
                ]
            }
        }
        
        # Research output API endpoints (where available)
        self.api_endpoints = {
            'nsf_awards': 'https://www.research.gov/awardapi-service/v1/awards.json',
            'nih_reporter': 'https://api.reporter.nih.gov/v2/projects/search'
        }
    
    def get_university_research_output(self, university_key: str, years: int = 5) -> Dict:
        """Get research output metrics for a university"""
        if university_key not in self.universities:
            return {}
            
        uni = self.universities[university_key]
        
        output = {
            'name': uni['name'],
            'research_funding': self._get_research_funding(uni['name'], years),
            'patents': self._get_patent_output(uni['name'], years),
            'spinoffs': self._get_spinoff_companies(uni['name']),
            'graduates': self._get_graduate_production(university_key),
            'industry_partnerships': self._get_industry_partnerships(university_key)
        }
        
        return output
    
    def _get_research_funding(self, uni_name: str, years: int) -> Dict:
        """Get research funding from federal sources"""
        funding = {
            'nsf': 0,
            'nih': 0,
            'other_federal': 0,
            'total': 0
        }
        
        # NSF Awards
        try:
            params = {
                'keyword': uni_name,
                'printFields': 'id,title,fundsObligatedAmt,startDate,endDate',
                'offset': 0
            }
            
            response = requests.get(self.api_endpoints['nsf_awards'], params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                awards = data.get('response', {}).get('award', [])
                for award in awards:
                    funding['nsf'] += float(award.get('fundsObligatedAmt', 0))
        except Exception as e:
            logger.warning(f"Could not fetch NSF data: {e}")
            # Use estimates based on university size
            funding['nsf'] = self._estimate_nsf_funding(uni_name)
        
        # NIH funding would be similar
        funding['nih'] = self._estimate_nih_funding(uni_name)
        funding['other_federal'] = funding['nsf'] * 0.5  # Rough estimate
        funding['total'] = sum(funding.values())
        
        return funding
    
    def _estimate_nsf_funding(self, uni_name: str) -> float:
        """Estimate NSF funding based on university"""
        estimates = {
            'University of Kansas': 45_000_000,
            'Kansas State University': 35_000_000,
            'University of Missouri-Kansas City': 15_000_000,
            'Missouri S&T': 25_000_000
        }
        return estimates.get(uni_name, 10_000_000)
    
    def _estimate_nih_funding(self, uni_name: str) -> float:
        """Estimate NIH funding based on university"""
        estimates = {
            'University of Kansas': 85_000_000,  # KU Medical Center
            'Kansas State University': 15_000_000,
            'University of Missouri-Kansas City': 25_000_000,
            'Missouri S&T': 5_000_000
        }
        return estimates.get(uni_name, 5_000_000)
    
    def _get_patent_output(self, uni_name: str, years: int) -> Dict:
        """Get patent statistics for university"""
        # This would query USPTO but for now use estimates
        patent_estimates = {
            'University of Kansas': {'total': 150, 'licensed': 45, 'commercialized': 12},
            'Kansas State University': {'total': 100, 'licensed': 30, 'commercialized': 8},
            'University of Missouri-Kansas City': {'total': 50, 'licensed': 15, 'commercialized': 4},
            'Missouri S&T': {'total': 80, 'licensed': 25, 'commercialized': 7}
        }
        
        base = patent_estimates.get(uni_name, {'total': 20, 'licensed': 5, 'commercialized': 1})
        return {
            'total_patents': base['total'],
            'licensed_patents': base['licensed'],
            'commercialized': base['commercialized'],
            'licensing_revenue': base['licensed'] * 50_000  # Average per license
        }
    
    def _get_spinoff_companies(self, uni_name: str) -> List[Dict]:
        """Get spinoff companies from university research"""
        # This would be scraped from university sites
        spinoffs = {
            'University of Kansas': [
                {'name': 'Orbis Biosciences', 'sector': 'biosciences', 'employees': 25},
                {'name': 'Savara Inc', 'sector': 'biosciences', 'employees': 50}
            ],
            'Kansas State University': [
                {'name': 'NanoScale Corporation', 'sector': 'materials', 'employees': 35}
            ]
        }
        
        return spinoffs.get(uni_name, [])
    
    def _get_graduate_production(self, university_key: str) -> Dict:
        """Get number of graduates by relevant programs"""
        # Estimates based on program sizes
        graduate_production = {
            'ku': {
                'engineering': 400,
                'computer_science': 250,
                'biosciences': 300,
                'business': 500
            },
            'ksu': {
                'engineering': 600,
                'agriculture': 400,
                'veterinary': 200,
                'business': 300
            },
            'umkc': {
                'medicine': 200,
                'pharmacy': 150,
                'business': 400,
                'computing': 200
            },
            'mst': {
                'engineering': 800,
                'computer_science': 300,
                'materials': 100
            }
        }
        
        return graduate_production.get(university_key, {})
    
    def _get_industry_partnerships(self, university_key: str) -> List[str]:
        """Get known industry partnerships"""
        partnerships = {
            'ku': ['Garmin', 'Black & Veatch', 'Honeywell', 'Cerner'],
            'ksu': ['Cargill', 'Hills Pet Nutrition', 'John Deere'],
            'umkc': ['Cerner', 'H&R Block', 'Burns & McDonnell'],
            'mst': ['Boeing', 'Caterpillar', 'Ameren']
        }
        
        return partnerships.get(university_key, [])
    
    def calculate_university_impact(self, cluster: Dict) -> Dict:
        """Calculate how universities can support a cluster"""
        cluster_type = cluster.get('type', 'mixed')
        businesses = cluster.get('businesses', [])
        
        impact = {
            'talent_pipeline': 0,
            'research_alignment': 0,
            'innovation_potential': 0,
            'partnerships': []
        }
        
        # Find relevant universities
        for uni_key, uni_data in self.universities.items():
            if cluster_type in uni_data['strengths']:
                # Check geographic proximity
                for business in businesses[:10]:  # Sample
                    if business.get('county') in uni_data['counties']:
                        impact['talent_pipeline'] += 100
                        
                # Get specific data
                research = self.get_university_research_output(uni_key)
                graduates = research.get('graduates', {})
                
                # Calculate talent pipeline
                relevant_grads = sum(v for k, v in graduates.items() 
                                   if k in ['engineering', 'computer_science', 'biosciences'])
                impact['talent_pipeline'] += relevant_grads
                
                # Research alignment
                if research['research_funding']['total'] > 50_000_000:
                    impact['research_alignment'] += 10
                    
                # Innovation potential
                try:
                    # Prefer explicit key if available
                    total_pats = research.get('patents', {}).get('total_patents')
                    if total_pats is None:
                        # Backward/alt key fallback
                        total_pats = research.get('patents', {}).get('total', 0)
                    impact['innovation_potential'] += float(total_pats) / 10.0
                except Exception:
                    pass
                
                # Partnerships
                impact['partnerships'].extend(research.get('industry_partnerships', []))
        
        # Calculate overall impact score
        score = 0.0
        
        # Talent pipeline component (0-40 points)
        if impact['talent_pipeline'] > 1000:
            score += 40
        elif impact['talent_pipeline'] > 500:
            score += 30
        elif impact['talent_pipeline'] > 200:
            score += 20
        else:
            score += 10 * (impact['talent_pipeline'] / 200)
            
        # Research alignment (0-30 points)
        score += min(impact['research_alignment'] * 3, 30)
        
        # Innovation potential (0-20 points)
        score += min(impact['innovation_potential'] * 2, 20)
        
        # Partnership strength (0-10 points)
        unique_partnerships = len(set(impact['partnerships']))
        score += min(unique_partnerships * 2, 10)
        
        impact['impact_score'] = min(score, 100)
        impact['partnership_potential'] = unique_partnerships > 5
        
        # Add the expected fields for compatibility
        impact['university_score'] = impact['impact_score']
        impact['talent_supply'] = min(impact['talent_pipeline'] / 10, 100)
        impact['research_alignment'] = min(impact['research_alignment'] * 10, 100)
        impact['overall_score'] = impact['impact_score']
        
        return impact
    
    def recommend_university_partnerships(self, clusters: List[Dict]) -> List[Dict]:
        """Recommend specific university partnerships for clusters"""
        recommendations = []
        
        for cluster in clusters:
            cluster_type = cluster.get('type')
            cluster_name = cluster.get('name')
            
            for uni_key, uni_data in self.universities.items():
                if cluster_type in uni_data['strengths']:
                    score = self._calculate_partnership_score(cluster, uni_data)
                    
                    if score > 0.5:
                        recommendations.append({
                            'cluster': cluster_name,
                            'university': uni_data['name'],
                            'score': score,
                            'strengths': uni_data['strengths'],
                            'research_centers': uni_data['research_centers'],
                            'actions': self._suggest_partnership_actions(cluster_type, uni_data)
                        })
        
        return sorted(recommendations, key=lambda x: x['score'], reverse=True)
    
    def _calculate_partnership_score(self, cluster: Dict, uni_data: Dict) -> float:
        """Calculate partnership compatibility score"""
        score = 0.0
        
        # Type alignment
        if cluster.get('type') in uni_data['strengths']:
            score += 0.4
            
        # Geographic proximity
        cluster_counties = set(b.get('county') for b in cluster.get('businesses', [])[:20])
        if cluster_counties.intersection(set(uni_data['counties'])):
            score += 0.3
            
        # Scale match
        if cluster.get('business_count', 0) > 20:
            score += 0.3
            
        return score
    
    def _suggest_partnership_actions(self, cluster_type: str, uni_data: Dict) -> List[str]:
        """Suggest specific partnership actions"""
        actions = []
        
        base_actions = [
            f"Establish internship pipeline with {uni_data['name']}",
            f"Create joint research projects with {uni_data['research_centers'][0]}",
            "Develop customized training programs",
            "Host student capstone projects"
        ]
        
        if cluster_type == 'biosciences':
            actions.extend([
                "Collaborate on clinical trials",
                "Share laboratory facilities",
                "Joint grant applications to NIH"
            ])
        elif cluster_type == 'technology':
            actions.extend([
                "Sponsor hackathons and coding competitions",
                "Establish innovation challenges",
                "Create startup incubator program"
            ])
        elif cluster_type == 'manufacturing':
            actions.extend([
                "Develop apprenticeship programs",
                "Share advanced manufacturing equipment",
                "Joint workforce training initiatives"
            ])
            
        return actions[:4]
