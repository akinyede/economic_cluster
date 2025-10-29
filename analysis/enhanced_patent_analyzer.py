"""Enhanced Patent Analyzer with full PatentSearch API integration"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import re
from collections import Counter

logger = logging.getLogger(__name__)

class EnhancedPatentAnalyzer:
    """Enhanced analyzer using all available PatentSearch API features"""
    
    def __init__(self):
        from config import Config
        self.config = Config()
        self.patent_api_base = self.config.USPTO_API_URL + "patent"
        self.assignee_api_base = self.config.USPTO_API_URL + "assignee"
        self.headers = {
            'User-Agent': 'KCClusterTool/1.0',
            'Content-Type': 'application/json',
            'X-Api-Key': self.config.USPTO_API_KEY
        }
        
        # Patent classification mapping to clusters
        self.cpc_to_cluster = {
            'A61': 'biosciences',      # Medical/veterinary science
            'C07': 'biosciences',      # Organic chemistry
            'C12': 'biosciences',      # Biochemistry
            'G06': 'technology',       # Computing
            'H04': 'technology',       # Electric communication
            'G16': 'technology',       # Healthcare informatics
            'B60': 'logistics',        # Vehicles
            'B65': 'logistics',        # Conveying, packing
            'B61': 'logistics',        # Railways
            'B29': 'manufacturing',    # Plastics processing
            'B23': 'manufacturing',    # Machine tools
            'B25': 'manufacturing',    # Hand tools
            'A01': 'agriculture',      # Agriculture
            'A23': 'agriculture',      # Foods
        }
        
        # Kansas City area assignees
        self.kc_assignees = [
            'garmin', 'cerner', 'sprint', 'honeywell',
            'university of kansas', 'kansas state university',
            'university of missouri', 'black veatch',
            'burns mcdonnell', 'hills pet nutrition'
        ]
        
    def search_patents_enhanced(self, keywords: List[str], 
                              start_date: Optional[datetime] = None,
                              assignee: Optional[str] = None) -> List[Dict]:
        """Enhanced patent search with CPC and citation data"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=5*365)
            
        patents = []
        
        try:
            # Build query
            query = {}
            
            if assignee:
                # Search for assignee in patent text
                query = {"_or": [
                    {"_text_any": {"patent_abstract": assignee}},
                    {"_text_any": {"patent_title": assignee}}
                ]}
            elif keywords:
                # Search by keywords
                keyword_queries = []
                for kw in keywords:
                    keyword_queries.append({"_text_any": {"patent_abstract": kw}})
                    keyword_queries.append({"_text_any": {"patent_title": kw}})
                query = {"_or": keyword_queries}
            else:
                # Default - get recent patents
                query = {"_gte": {"patent_date": "2024-01-01"}}
                
            # Add date filter
            if start_date:
                date_filter = {"_gte": {"patent_date": start_date.strftime("%Y-%m-%d")}}
                query = {"_and": [query, date_filter]}
            
            # Request all available fields including CPC and citations
            request_data = {
                "q": query,
                "f": [
                    "patent_id", "patent_title", "patent_date", "patent_type", 
                    "patent_abstract", "patent_year", "wipo_kind", "patent_processing_days",
                    "cpc_current", "patent_num_times_cited_by_us_patents", 
                    "assignees", "inventors"
                ],
                "s": [{"patent_date": "desc"}],
                "o": {
                    "per_page": 100,
                    "page": 1,
                    "exclude_withdrawn": True
                }
            }
            
            logger.info(f"Searching patents with enhanced fields")
            response = requests.post(
                self.patent_api_base,
                json=request_data,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                total_hits = data.get('total_hits', 0)
                logger.info(f"Found {total_hits} total patents, processing {len(data.get('patents', []))} results")
                
                for patent in data.get('patents', []):
                    # Extract basic fields
                    patent_id = patent.get('patent_id', '')
                    patent_title = patent.get('patent_title', '')
                    patent_date = patent.get('patent_date', '')
                    patent_abstract = patent.get('patent_abstract', '')
                    
                    # Extract CPC codes from cpc_current array
                    cpc_codes = []
                    cpc_data = patent.get('cpc_current', [])
                    for cpc in cpc_data:
                        if 'cpc_group_id' in cpc:
                            cpc_codes.append(cpc['cpc_group_id'])
                        elif 'cpc_subclass_id' in cpc:
                            cpc_codes.append(cpc['cpc_subclass_id'])
                    
                    # Extract citation count
                    citations_received = patent.get('patent_num_times_cited_by_us_patents', 0)
                    
                    # Try to extract assignee from assignees field or text
                    assignee_name = self._extract_assignee(patent, assignee)
                    
                    # Try to extract inventors
                    inventors = self._extract_inventors(patent)
                    
                    patents.append({
                        'patent_number': patent_id,
                        'title': patent_title,
                        'assignee': assignee_name,
                        'inventors': inventors,
                        'filing_date': patent_date,
                        'grant_date': patent_date,
                        'cpc_codes': cpc_codes[:10],  # First 10 CPC codes
                        'abstract': patent_abstract[:500] if patent_abstract else '',
                        'citations_made': 0,  # Not available
                        'citations_received': citations_received,
                        'claims': 0,  # Not available
                        'patent_type': patent.get('patent_type', ''),
                        'wipo_kind': patent.get('wipo_kind', ''),
                        'processing_days': patent.get('patent_processing_days', 0)
                    })
                
                logger.info(f"Processed {len(patents)} patents with enhanced data")
                return patents
            else:
                logger.warning(f"Patent API returned status {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching patents: {e}")
            return []
    
    def search_kc_assignees(self) -> Dict[str, str]:
        """Search for Kansas City area assignees and return mapping"""
        kc_assignee_map = {}
        
        try:
            # Search for Kansas City assignees
            request_data = {
                "q": {"_or": [
                    {"assignee_lastknown_city": "Kansas City"},
                    {"assignee_lastknown_city": "Overland Park"},
                    {"assignee_lastknown_city": "Lenexa"},
                    {"assignee_lastknown_city": "Olathe"}
                ]},
                "f": ["assignee_id", "assignee_organization", 
                      "assignee_lastknown_city", "assignee_lastknown_state"],
                "o": {"per_page": 1000}
            }
            
            response = requests.post(
                self.assignee_api_base,
                json=request_data,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Found {data.get('total_hits', 0)} KC area assignees")
                
                for assignee in data.get('assignees', []):
                    org = assignee.get('assignee_organization', '')
                    assignee_id = assignee.get('assignee_id', '')
                    if org and assignee_id:
                        kc_assignee_map[assignee_id] = {
                            'name': org,
                            'city': assignee.get('assignee_lastknown_city', ''),
                            'state': assignee.get('assignee_lastknown_state', '')
                        }
                        
                # Also search for known KC companies
                for company in self.kc_assignees:
                    request_data = {
                        "q": {"_text_any": {"assignee_organization": company}},
                        "f": ["assignee_id", "assignee_organization"],
                        "o": {"per_page": 10}
                    }
                    
                    response = requests.post(
                        self.assignee_api_base,
                        json=request_data,
                        headers=self.headers,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        for assignee in data.get('assignees', []):
                            assignee_id = assignee.get('assignee_id', '')
                            if assignee_id and assignee_id not in kc_assignee_map:
                                kc_assignee_map[assignee_id] = {
                                    'name': assignee.get('assignee_organization', ''),
                                    'city': 'Kansas City Area',
                                    'state': 'MO/KS'
                                }
                
                logger.info(f"Built KC assignee map with {len(kc_assignee_map)} entries")
                
        except Exception as e:
            logger.error(f"Error searching KC assignees: {e}")
            
        return kc_assignee_map
    
    def _extract_assignee(self, patent: Dict, search_assignee: Optional[str]) -> str:
        """Extract assignee from patent data"""
        # Check assignees field (if populated)
        assignees = patent.get('assignees', [])
        if assignees and isinstance(assignees, list) and len(assignees) > 0:
            # Return first assignee
            if isinstance(assignees[0], dict):
                return assignees[0].get('assignee_organization', 'Unknown')
            elif isinstance(assignees[0], str):
                return assignees[0]
        
        # Fallback: Check if search assignee appears in title/abstract
        if search_assignee:
            title = patent.get('patent_title', '').lower()
            abstract = patent.get('patent_abstract', '').lower()
            search_lower = search_assignee.lower()
            
            if search_lower in title:
                return search_assignee
            elif abstract and search_lower in abstract:
                return f"{search_assignee} (mentioned)"
                
        return 'Unknown'
    
    def _extract_inventors(self, patent: Dict) -> List[str]:
        """Extract inventors from patent data"""
        inventors = patent.get('inventors', [])
        inventor_names = []
        
        if inventors and isinstance(inventors, list):
            for inventor in inventors[:5]:  # First 5 inventors
                if isinstance(inventor, dict):
                    name = inventor.get('inventor_name', '')
                    if not name:
                        first = inventor.get('inventor_name_first', '')
                        last = inventor.get('inventor_name_last', '')
                        if first or last:
                            name = f"{first} {last}".strip()
                    if name:
                        inventor_names.append(name)
                elif isinstance(inventor, str):
                    inventor_names.append(inventor)
                    
        return inventor_names
    
    def analyze_patent_portfolio_enhanced(self, patents: List[Dict]) -> Dict:
        """Enhanced analysis with real CPC and citation data"""
        if not patents:
            return {
                'total_patents': 0,
                'innovation_score': 0,
                'technology_diversity': 0,
                'citation_impact': 0
            }
            
        analysis = {
            'total_patents': len(patents),
            'by_year': {},
            'by_assignee': {},
            'by_technology': {},
            'by_cpc_class': {},
            'citation_metrics': {
                'total_citations_made': 0,
                'total_citations_received': 0,
                'avg_citations_per_patent': 0,
                'highly_cited_patents': 0
            },
            'innovation_metrics': {}
        }
        
        # Analyze by year
        for patent in patents:
            year = patent.get('grant_date', '')[:4]
            if year:
                analysis['by_year'][year] = analysis['by_year'].get(year, 0) + 1
                
        # Analyze by assignee
        assignee_counts = Counter(p.get('assignee', 'Unknown') for p in patents)
        analysis['by_assignee'] = dict(assignee_counts.most_common(10))
        
        # Analyze by CPC codes (real data!)
        cpc_classes = []
        tech_areas = []
        
        for patent in patents:
            cpc_codes = patent.get('cpc_codes', [])
            for code in cpc_codes:
                # Extract class (first 3 chars)
                if len(code) >= 3:
                    cpc_class = code[:3]
                    cpc_classes.append(cpc_class)
                    
                    # Map to technology area
                    if cpc_class in self.cpc_to_cluster:
                        tech_areas.append(self.cpc_to_cluster[cpc_class])
                        
        # Count CPC classes
        cpc_counts = Counter(cpc_classes)
        analysis['by_cpc_class'] = dict(cpc_counts.most_common(10))
        
        # Count technology areas
        tech_counts = Counter(tech_areas)
        analysis['by_technology'] = dict(tech_counts)
        
        # Citation analysis (real data!)
        citations_received = [p.get('citations_received', 0) for p in patents]
        total_received = sum(citations_received)
        highly_cited = sum(1 for c in citations_received if c > 10)
        
        analysis['citation_metrics'] = {
            'total_citations_made': 0,  # Not available
            'total_citations_received': total_received,
            'avg_citations_per_patent': total_received / len(patents) if patents else 0,
            'highly_cited_patents': highly_cited,
            'citation_impact_score': min(total_received / (len(patents) * 5), 1.0) * 100
        }
        
        # Calculate innovation metrics with real data
        analysis['innovation_metrics'] = self._calculate_innovation_metrics_enhanced(
            patents, analysis
        )
        
        return analysis
    
    def _calculate_innovation_metrics_enhanced(self, patents: List[Dict], 
                                             analysis: Dict) -> Dict:
        """Calculate innovation metrics with real CPC diversity"""
        metrics = {}
        
        # Technology diversity using real CPC data
        if analysis['by_cpc_class']:
            total = sum(analysis['by_cpc_class'].values())
            probs = [count/total for count in analysis['by_cpc_class'].values()]
            entropy = -sum(p * np.log(p) for p in probs if p > 0)
            # Normalize by max possible entropy
            max_entropy = np.log(len(analysis['by_cpc_class']))
            metrics['technology_diversity'] = (entropy / max_entropy * 100) if max_entropy > 0 else 0
        else:
            metrics['technology_diversity'] = 0
            
        # Patent velocity
        if analysis['by_year']:
            years = len(analysis['by_year'])
            metrics['patent_velocity'] = len(patents) / years if years > 0 else 0
        else:
            metrics['patent_velocity'] = 0
            
        # Innovation score (composite)
        metrics['innovation_score'] = (
            min(len(patents) / 50, 1.0) * 0.3 +  # Volume
            metrics.get('technology_diversity', 0) / 100 * 0.3 +  # Diversity
            analysis['citation_metrics']['citation_impact_score'] / 100 * 0.4  # Impact
        ) * 100
        
        # Breakthrough potential (using real citation data)
        metrics['breakthrough_ratio'] = (
            analysis['citation_metrics']['highly_cited_patents'] / len(patents) 
            if patents else 0
        )
        
        # CPC concentration (new metric)
        if analysis['by_cpc_class']:
            top_3_classes = list(analysis['by_cpc_class'].values())[:3]
            total_codes = sum(analysis['by_cpc_class'].values())
            metrics['cpc_concentration'] = sum(top_3_classes) / total_codes if total_codes > 0 else 0
        else:
            metrics['cpc_concentration'] = 0
            
        return metrics
    
    def generate_enhanced_patent_report(self, clusters: List[Dict]) -> Dict:
        """Generate comprehensive patent report with real data"""
        # First, get KC assignee mapping
        kc_assignees = self.search_kc_assignees()
        
        report = {
            'total_patents_analyzed': 0,
            'kc_assignees_found': len(kc_assignees),
            'by_cluster': {},
            'top_innovators': [],
            'emerging_technologies': set(),
            'cpc_distribution': {},
            'innovation_rankings': [],
            'strategic_recommendations': []
        }
        
        # Analyze each cluster
        for cluster in clusters:
            cluster_name = cluster.get('name', 'Unknown')
            cluster_type = cluster.get('type', 'mixed')
            
            # Search for patents in this cluster
            keywords = self._get_cluster_keywords(cluster_type)
            patents = self.search_patents_enhanced(keywords)
            
            # Analyze with enhanced metrics
            analysis = self.analyze_patent_portfolio_enhanced(patents)
            
            cluster_innovation = {
                'patent_count': analysis['total_patents'],
                'innovation_score': analysis['innovation_metrics'].get('innovation_score', 0),
                'technology_diversity': analysis['innovation_metrics'].get('technology_diversity', 0),
                'patent_velocity': analysis['innovation_metrics'].get('patent_velocity', 0),
                'citation_impact': analysis['citation_metrics']['citation_impact_score'],
                'top_cpc_classes': list(analysis.get('by_cpc_class', {}).keys())[:5],
                'breakthrough_ratio': analysis['innovation_metrics'].get('breakthrough_ratio', 0),
                'top_innovators': list(analysis['by_assignee'].keys())[:5]
            }
            
            report['by_cluster'][cluster_name] = cluster_innovation
            report['total_patents_analyzed'] += cluster_innovation['patent_count']
            
            # Track CPC distribution
            for cpc, count in analysis.get('by_cpc_class', {}).items():
                report['cpc_distribution'][cpc] = report['cpc_distribution'].get(cpc, 0) + count
            
            # Track rankings
            report['innovation_rankings'].append({
                'cluster': cluster_name,
                'score': cluster_innovation['innovation_score'],
                'patents': cluster_innovation['patent_count'],
                'diversity': cluster_innovation['technology_diversity'],
                'citations': cluster_innovation['citation_impact']
            })
        
        # Sort rankings by innovation score
        report['innovation_rankings'].sort(key=lambda x: x['score'], reverse=True)
        
        # Identify emerging technologies from CPC codes
        emerging_cpc = sorted(report['cpc_distribution'].items(), 
                            key=lambda x: x[1], reverse=True)[:10]
        report['emerging_technologies'] = [f"{cpc[0]} ({cpc[1]} patents)" 
                                         for cpc in emerging_cpc]
        
        # Generate strategic recommendations
        report['strategic_recommendations'] = self._generate_enhanced_recommendations(report)
        
        return report
    
    def _generate_enhanced_recommendations(self, report: Dict) -> List[str]:
        """Generate recommendations based on real patent data"""
        recommendations = []
        
        # Based on CPC distribution
        if report['cpc_distribution']:
            top_cpc = list(report['cpc_distribution'].keys())[0]
            if top_cpc in self.cpc_to_cluster:
                cluster_type = self.cpc_to_cluster[top_cpc]
                recommendations.append(
                    f"Focus on {cluster_type} cluster - highest patent activity in {top_cpc}"
                )
        
        # Based on KC assignees
        if report['kc_assignees_found'] > 0:
            recommendations.append(
                f"Leverage {report['kc_assignees_found']} KC-area patent assignees for partnerships"
            )
        
        # Based on innovation rankings
        if report['innovation_rankings']:
            leader = report['innovation_rankings'][0]
            if leader['diversity'] < 50:
                recommendations.append(
                    f"Increase technology diversity in {leader['cluster']} cluster"
                )
                
        # Based on citation impact
        low_citation_clusters = [r for r in report['innovation_rankings'] 
                               if r['citations'] < 20]
        if low_citation_clusters:
            recommendations.append(
                "Focus on breakthrough innovations to increase citation impact"
            )
            
        return recommendations
    
    def _get_cluster_keywords(self, cluster_type: str) -> List[str]:
        """Get search keywords for cluster type"""
        keywords = {
            'technology': ['artificial intelligence', 'machine learning', 
                          'software', 'data analytics', 'cybersecurity'],
            'biosciences': ['drug delivery', 'biotechnology', 'pharmaceutical',
                           'medical device', 'diagnostic'],
            'logistics': ['supply chain', 'warehouse automation', 'transportation',
                         'tracking', 'logistics optimization'],
            'manufacturing': ['automation', 'process control', 'quality control',
                            'additive manufacturing', 'robotics'],
            'animal_health': ['veterinary', 'animal diagnostic', 'pet nutrition',
                            'livestock', 'animal pharmaceutical']
        }
        
        return keywords.get(cluster_type, ['innovation'])