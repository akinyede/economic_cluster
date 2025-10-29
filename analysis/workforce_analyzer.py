"""Workforce analysis using BLS SOC data"""

import requests
import pandas as pd
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class WorkforceAnalyzer:
    """Analyze workforce needs using Standard Occupational Classification (SOC) codes"""
    
    def __init__(self):
        self.bls_api_key = None  # Optional - works without key but with limits
        self.base_url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
        
        # SOC codes for KC strategic industries
        self.soc_mapping = {
            'logistics': {
                '53-3032': 'Heavy and Tractor-Trailer Truck Drivers',
                '13-1081': 'Logisticians', 
                '53-7062': 'Laborers and Material Movers',
                '11-3071': 'Transportation Managers',
                '53-1047': 'First-Line Supervisors of Transportation'
            },
            'technology': {
                '15-1252': 'Software Developers',
                '15-1211': 'Computer Systems Analysts',
                '15-1212': 'Information Security Analysts',
                '15-1299': 'Computer Occupations, All Other',
                '17-2061': 'Computer Hardware Engineers'
            },
            'biosciences': {
                '19-1021': 'Biochemists and Biophysicists',
                '19-1029': 'Biological Scientists',
                '19-4021': 'Biological Technicians',
                '29-2011': 'Medical Scientists',
                '17-2031': 'Bioengineers'
            },
            'manufacturing': {
                '51-4041': 'Machinists',
                '17-2112': 'Industrial Engineers',
                '51-1011': 'Production Supervisors',
                '51-9061': 'Inspectors and Testers',
                '49-9041': 'Industrial Machinery Mechanics'
            },
            'animal_health': {
                '29-1131': 'Veterinarians',
                '29-2056': 'Veterinary Technologists',
                '19-1011': 'Animal Scientists',
                '31-9096': 'Veterinary Assistants',
                '19-1029': 'Biological Scientists'
            }
        }
        
    def get_occupation_data(self, cluster_type: str, metro_area: str = "28140") -> Dict:
        """
        Get occupation data for a cluster type in KC metro (28140)
        
        Returns:
            Dict with employment, wages, growth projections
        """
        if cluster_type not in self.soc_mapping:
            return {}
            
        occupation_data = {}
        
        # For each occupation in the cluster
        for soc_code, title in self.soc_mapping[cluster_type].items():
            # OEWS series for KC metro area
            series_id = f"OEUM{metro_area}000000{soc_code.replace('-', '')}01"
            
            try:
                # Get employment data
                emp_data = self._fetch_bls_data(series_id)
                
                # Get wage data (series code 03 for hourly mean wage)
                wage_series = f"OEUM{metro_area}000000{soc_code.replace('-', '')}03"
                wage_data = self._fetch_bls_data(wage_series)
                
                occupation_data[soc_code] = {
                    'title': title,
                    'employment': emp_data.get('latest_value', 0),
                    'employment_trend': emp_data.get('trend', 0),
                    'hourly_wage': wage_data.get('latest_value', 0),
                    'annual_wage': wage_data.get('latest_value', 0) * 2080,
                    'wage_trend': wage_data.get('trend', 0)
                }
                
            except Exception as e:
                logger.warning(f"Could not fetch data for {soc_code}: {e}")
                # Use default estimates
                occupation_data[soc_code] = self._get_default_occupation_data(soc_code, title)
                
        return occupation_data
    
    def _fetch_bls_data(self, series_id: str) -> Dict:
        """Fetch data from BLS API"""
        params = {
            'seriesid': [series_id],
            'startyear': '2022',
            'endyear': '2024'
        }
        
        if self.bls_api_key:
            params['registrationkey'] = self.bls_api_key
            
        try:
            response = requests.post(self.base_url, json=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'REQUEST_SUCCEEDED':
                    series = data['Results']['series'][0]['data']
                    latest = float(series[0]['value'])
                    oldest = float(series[-1]['value'])
                    trend = ((latest - oldest) / oldest) * 100 if oldest > 0 else 0
                    
                    return {
                        'latest_value': latest,
                        'trend': trend,
                        'data': series
                    }
        except:
            pass
            
        return {'latest_value': 0, 'trend': 0}
    
    def _get_default_occupation_data(self, soc_code: str, title: str) -> Dict:
        """Get default occupation data based on national averages"""
        # Default data based on occupation type
        defaults = {
            '53-': {'employment': 500, 'wage': 25},  # Transportation
            '13-': {'employment': 200, 'wage': 40},  # Business ops
            '15-': {'employment': 300, 'wage': 50},  # Computer
            '19-': {'employment': 150, 'wage': 35},  # Life sciences
            '29-': {'employment': 100, 'wage': 45},  # Healthcare
            '51-': {'employment': 400, 'wage': 22},  # Production
            '17-': {'employment': 150, 'wage': 45},  # Engineering
        }
        
        prefix = soc_code[:3]
        default = defaults.get(prefix, {'employment': 100, 'wage': 30})
        
        return {
            'title': title,
            'employment': default['employment'],
            'employment_trend': 3.5,  # KC average
            'hourly_wage': default['wage'],
            'annual_wage': default['wage'] * 2080,
            'wage_trend': 2.5
        }
    
    def project_workforce_needs(self, cluster: Dict, years: int = 5) -> Dict:
        """Project workforce needs for a cluster over time"""
        cluster_type = cluster.get('type', 'mixed')
        current_jobs = cluster.get('projected_jobs', 0)
        
        occupation_data = self.get_occupation_data(cluster_type)
        
        projections = {}
        total_need = 0
        
        for soc_code, data in occupation_data.items():
            # Calculate share of employment
            total_employment = sum(d['employment'] for d in occupation_data.values())
            emp_share = data['employment'] / total_employment if total_employment > 0 else 0
            
            # Project need based on cluster growth
            occupation_need = int(current_jobs * emp_share)
            growth_rate = data['employment_trend'] / 100
            future_need = int(occupation_need * (1 + growth_rate) ** years)
            
            projections[soc_code] = {
                'title': data['title'],
                'current_need': occupation_need,
                'future_need': future_need,
                'gap': future_need - occupation_need,
                'annual_wage': data['annual_wage'],
                'total_wage_cost': future_need * data['annual_wage']
            }
            
            total_need += future_need
            
        return {
            'by_occupation': projections,
            'total_positions': total_need,
            'total_wage_cost': sum(p['total_wage_cost'] for p in projections.values()),
            'critical_occupations': self._identify_critical_occupations(projections)
        }
    
    def _identify_critical_occupations(self, projections: Dict) -> List[str]:
        """Identify occupations with largest gaps or highest wages"""
        critical = []
        
        # Sort by gap size
        by_gap = sorted(projections.items(), key=lambda x: x[1]['gap'], reverse=True)
        critical.extend([soc for soc, _ in by_gap[:3]])
        
        # Sort by wage (skills shortage likely)
        by_wage = sorted(projections.items(), key=lambda x: x[1]['annual_wage'], reverse=True)
        critical.extend([soc for soc, _ in by_wage[:2]])
        
        return list(set(critical))
    
    def generate_workforce_report(self, clusters: List[Dict]) -> Dict:
        """Generate comprehensive workforce analysis report"""
        report = {
            'total_jobs_created': sum(c.get('projected_jobs', 0) for c in clusters),
            'by_cluster': {},
            'critical_skills': set(),
            'training_needs': []
        }
        
        for cluster in clusters:
            workforce = self.project_workforce_needs(cluster)
            cluster_name = cluster.get('name', 'Unknown')
            
            report['by_cluster'][cluster_name] = workforce
            report['critical_skills'].update(workforce['critical_occupations'])
            
            # Identify training needs
            for soc, data in workforce['by_occupation'].items():
                if data['gap'] > 50:  # Significant gap
                    report['training_needs'].append({
                        'occupation': data['title'],
                        'cluster': cluster_name,
                        'gap': data['gap'],
                        'priority': 'high' if soc in workforce['critical_occupations'] else 'medium'
                    })
        
        report['critical_skills'] = list(report['critical_skills'])
        return report
    
    def analyze_cluster_workforce(self, cluster: Dict) -> Dict:
        """Analyze workforce requirements for a specific cluster"""
        cluster_type = cluster.get('type', 'mixed')
        businesses = cluster.get('businesses', [])
        
        # Calculate total workforce needs
        total_employees = sum(b.get('employees', 0) for b in businesses)
        
        # Get occupation data for cluster
        occupation_data = self.get_occupation_data(cluster_type)
        
        # Project future needs
        workforce_projection = self.project_workforce_needs(cluster)
        
        # Analyze skill gaps
        skill_gaps = []
        if cluster_type == 'technology':
            skill_gaps = ['AI/ML specialists', 'Cybersecurity experts', 'Cloud architects']
        elif cluster_type == 'biosciences':
            skill_gaps = ['Clinical researchers', 'Biostatisticians', 'Lab technicians']
        elif cluster_type == 'logistics':
            skill_gaps = ['Supply chain analysts', 'Warehouse automation specialists', 'CDL drivers']
        
        # Calculate scores
        skills_availability = 70.0 if total_employees > 100 else 50.0  # Basic scoring
        talent_pipeline = 60.0 if cluster_type in ['technology', 'biosciences'] else 40.0
        wage_competitiveness = 65.0  # Mid-range default
        
        overall_score = (skills_availability * 0.4 + talent_pipeline * 0.4 + wage_competitiveness * 0.2)
        
        return {
            'current_workforce': total_employees,
            'total_workforce': total_employees,  # Add this field
            'occupation_mix': occupation_data,
            'projected_needs': workforce_projection,
            'skill_gaps': skill_gaps,
            'training_recommendations': self._generate_training_recommendations(cluster_type, skill_gaps),
            'skills_availability': skills_availability,
            'talent_pipeline': talent_pipeline,
            'wage_competitiveness': wage_competitiveness,
            'overall_score': overall_score
        }
    
    def _generate_training_recommendations(self, cluster_type: str, skill_gaps: List[str]) -> List[Dict]:
        """Generate specific training program recommendations"""
        recommendations = []
        
        if cluster_type == 'technology':
            recommendations.append({
                'program': 'Software Development Bootcamp',
                'provider': 'KC Tech Council',
                'duration': '12 weeks',
                'cost': '$5,000',
                'target_jobs': 50
            })
        elif cluster_type == 'logistics':
            recommendations.append({
                'program': 'CDL Training Program',
                'provider': 'Metropolitan Community College',
                'duration': '4 weeks',
                'cost': '$3,000',
                'target_jobs': 100
            })
        
        return recommendations