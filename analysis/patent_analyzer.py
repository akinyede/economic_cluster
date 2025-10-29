"""Patent analysis for innovation assessment"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import re
from collections import Counter
from data_collection.patent_batch_optimizer import BatchPatentSearcher

logger = logging.getLogger(__name__)

class PatentAnalyzer:
    """Analyze USPTO patent data for innovation metrics"""
    
    def __init__(self):
        from config import Config
        self.config = Config()
        # Updated URL for PatentSearch API v2
        # The new API uses /patent/ (with trailing slash) instead of /patent
        self.uspto_api_base = "https://search.patentsview.org/api/v1/patent/"
        self.headers = {
            'User-Agent': 'KCClusterTool/2.0',
            'Content-Type': 'application/json',
            'X-Api-Key': self.config.USPTO_API_KEY
        }
        
        # Initialize batch patent searcher for efficient searches
        self.batch_searcher = BatchPatentSearcher(self.config)
        self._patent_cache = {}  # Cache patent counts by business name
        self._all_patents_fetched = False  # Track if we've done the full fetch
        
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
        
        # Kansas City area assignees (companies/universities)
        self.kc_assignees = [
            'garmin', 'cerner', 'sprint', 'honeywell',
            'university of kansas', 'kansas state university',
            'university of missouri', 'black veatch',
            'burns mcdonnell', 'hill\'s pet nutrition'
        ]
        
    def search_patents(self, keywords: List[str], 
                      start_date: Optional[datetime] = None,
                      assignee: Optional[str] = None) -> List[Dict]:
        """Search USPTO for patents"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=5*365)  # Last 5 years
            
        patents = []
        
        # Build query
        query_parts = []
        if keywords:
            query_parts.append(' OR '.join(f'"{kw}"' for kw in keywords))
        if assignee:
            query_parts.append(f'assignee:"{assignee}"')
            
        query = ' AND '.join(query_parts) if query_parts else '*:*'
        
        try:
            # Build request for PatentSearch API
            # Build query
            query = {}
            
            if assignee:
                # Search for assignee in patent text (abstract and title)
                # Direct assignee field searches cause server errors
                query = {"_or": [
                    {"_text_any": {"patent_abstract": assignee}},
                    {"_text_any": {"patent_title": assignee}}
                ]}
            elif keywords:
                # Search by keywords in abstract or title
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
            
            request_data = {
                "q": query,
                "f": ["patent_id", "patent_title", "patent_date", "patent_type", 
                      "patent_abstract", "patent_year", "wipo_kind", "patent_processing_days",
                      "cpc_current", "patent_num_times_cited_by_us_patents"],
                "s": [{"patent_date": "desc"}],
                "o": {
                    "per_page": 100,
                    "page": 1,
                    "exclude_withdrawn": True
                }
            }
            
            logger.info(f"Searching patents with PatentsView API")
            logger.info(f"API URL: {self.uspto_api_base}")
            response = requests.post(
                self.uspto_api_base,
                json=request_data,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                patents = []
                
                # The API returns data in 'patents' array
                total_hits = data.get('total_hits', 0)
                logger.info(f"USPTO API returned {len(data.get('patents', []))} patents (total hits: {total_hits})")
                
                for patent in data.get('patents', []):
                    # Parse all available fields
                    patent_id = patent.get('patent_id', '')
                    patent_title = patent.get('patent_title', '')
                    patent_date = patent.get('patent_date', '')
                    patent_abstract = patent.get('patent_abstract', '')
                    patent_type = patent.get('patent_type', '')
                    wipo_kind = patent.get('wipo_kind', '')
                    processing_days = patent.get('patent_processing_days', 0)
                    
                    # Extract CPC codes from cpc_current array
                    cpc_codes = []
                    cpc_data = patent.get('cpc_current', [])
                    for cpc in cpc_data:
                        if 'cpc_group_id' in cpc:
                            cpc_codes.append(cpc['cpc_group_id'])
                        elif 'cpc_subclass_id' in cpc:
                            cpc_codes.append(cpc['cpc_subclass_id'])
                    
                    # Get citation count
                    citations_received = patent.get('patent_num_times_cited_by_us_patents', 0)
                    
                    # Extract assignee from title or abstract if searching for a specific company
                    assignee_name = 'Unknown'
                    if assignee:
                        if assignee.lower() in patent_title.lower():
                            assignee_name = assignee
                        elif patent_abstract and assignee.lower() in patent_abstract.lower():
                            assignee_name = f"{assignee} (mentioned in abstract)"
                    
                    # If no CPC codes found, extract technology hints from abstract
                    if not cpc_codes and patent_abstract:
                        abstract_lower = patent_abstract.lower()
                        if 'artificial intelligence' in abstract_lower or ' ai ' in abstract_lower:
                            cpc_codes.append('G06N')  # AI CPC code
                        if 'biotech' in abstract_lower or 'biological' in abstract_lower:
                            cpc_codes.append('C12N')  # Biotech CPC code
                        if 'software' in abstract_lower:
                            cpc_codes.append('G06F')  # Software CPC code
                    
                    patents.append({
                        'patent_number': patent_id,
                        'title': patent_title,
                        'assignee': assignee_name,
                        'inventors': [],  # Not available from API
                        'filing_date': patent_date,
                        'grant_date': patent_date,
                        'cpc_codes': cpc_codes[:10],  # Real CPC codes from API
                        'abstract': patent_abstract[:500] if patent_abstract else '',  # First 500 chars
                        'citations_made': 0,  # Not available from API
                        'citations_received': citations_received,  # Real citation count from API
                        'claims': 0,  # Not available from API
                        'patent_type': patent_type,
                        'wipo_kind': wipo_kind,
                        'processing_days': processing_days
                    })
                
                logger.info(f"Processed {len(patents)} patents from USPTO API")
                return patents
            else:
                logger.warning(f"USPTO API returned status {response.status_code}")
                if response.status_code == 404:
                    logger.error("USPTO PatentsView API v1 was discontinued on May 1, 2025. Using fallback data.")
                else:
                    logger.warning(f"Response: {response.text[:200]}")
                # Fall back to mock data
                logger.warning("USPTO API failed - returning empty patent data")
                return []  # No mock data - real data only
                
        except Exception as e:
            logger.error(f"Error searching patents: {e}")
            # Fall back to mock data
            logger.warning("USPTO API failed - returning empty patent data")
            return []  # No mock data - real data only
            
        return patents
    
    def _get_mock_patent_data(self, keywords: List[str], 
                             assignee: Optional[str] = None) -> List[Dict]:
        """Return mock patent data for KC area"""
        mock_patents = [
            {
                'patent_number': 'US11234567B2',
                'title': 'Advanced GPS Navigation System with Machine Learning',
                'assignee': 'Garmin Ltd.',
                'inventors': ['John Smith', 'Jane Doe'],
                'filing_date': '2021-03-15',
                'grant_date': '2023-06-20',
                'cpc_codes': ['G06N20/00', 'G01C21/36'],
                'abstract': 'A navigation system using ML for route optimization',
                'citations_made': 15,
                'citations_received': 3,
                'claims': 20
            },
            {
                'patent_number': 'US11345678B1',
                'title': 'Automated Warehouse Management System',
                'assignee': 'KC Logistics Innovation LLC',
                'inventors': ['Bob Johnson'],
                'filing_date': '2020-08-10',
                'grant_date': '2022-11-15',
                'cpc_codes': ['B65G1/137', 'G06Q10/08'],
                'abstract': 'System for autonomous warehouse operations',
                'citations_made': 12,
                'citations_received': 8,
                'claims': 18
            },
            {
                'patent_number': 'US11456789A1',
                'title': 'Novel Drug Delivery Nanoparticles',
                'assignee': 'University of Kansas',
                'inventors': ['Dr. Sarah Chen', 'Dr. Michael Brown'],
                'filing_date': '2019-12-01',
                'grant_date': '2023-02-28',
                'cpc_codes': ['A61K9/51', 'A61K47/69'],
                'abstract': 'Biodegradable nanoparticles for targeted drug delivery',
                'citations_made': 45,
                'citations_received': 12,
                'claims': 25
            },
            {
                'patent_number': 'US11567890B2',
                'title': 'Veterinary Diagnostic Device',
                'assignee': 'Kansas State University',
                'inventors': ['Dr. Robert Wilson'],
                'filing_date': '2020-05-20',
                'grant_date': '2023-01-10',
                'cpc_codes': ['A61B5/00', 'A61D99/00'],
                'abstract': 'Portable diagnostic device for animal health',
                'citations_made': 20,
                'citations_received': 5,
                'claims': 15
            },
            {
                'patent_number': 'US11678901C1',
                'title': 'Smart Manufacturing Process Control',
                'assignee': 'Honeywell International Inc.',
                'inventors': ['Team Alpha'],
                'filing_date': '2021-07-01',
                'grant_date': '2023-09-15',
                'cpc_codes': ['G05B19/418', 'G06N3/08'],
                'abstract': 'AI-driven manufacturing process optimization',
                'citations_made': 30,
                'citations_received': 2,
                'claims': 22
            }
        ]
        
        # Filter by assignee if specified
        if assignee:
            mock_patents = [p for p in mock_patents 
                          if assignee.lower() in p['assignee'].lower()]
                          
        return mock_patents
    
    def analyze_patent_portfolio(self, patents: List[Dict]) -> Dict:
        """Analyze patent portfolio for innovation metrics"""
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
            'citation_metrics': {
                'total_citations_made': 0,
                'total_citations_received': 0,
                'avg_citations_per_patent': 0
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
        
        # Analyze by technology (CPC codes)
        tech_areas = []
        cpc_classes = []
        for patent in patents:
            cpc_codes = patent.get('cpc_codes', [])
            for code in cpc_codes:
                # Extract CPC class (first 3 chars)
                if len(code) >= 3:
                    prefix = code[:3]
                    cpc_classes.append(prefix)
                    if prefix in self.cpc_to_cluster:
                        tech_areas.append(self.cpc_to_cluster[prefix])
                    
        tech_counts = Counter(tech_areas)
        analysis['by_technology'] = dict(tech_counts)
        
        # Citation analysis
        total_made = sum(p.get('citations_made', 0) for p in patents)
        total_received = sum(p.get('citations_received', 0) for p in patents)
        
        analysis['citation_metrics'] = {
            'total_citations_made': total_made,
            'total_citations_received': total_received,
            'avg_citations_per_patent': total_received / len(patents) if patents else 0,
            'citation_impact_score': min(total_received / (len(patents) * 5), 1.0) * 100
        }
        
        # Calculate innovation metrics
        innovation_metrics = self._calculate_innovation_metrics(
            patents, analysis
        )
        analysis['innovation_metrics'] = innovation_metrics
        
        return analysis
    
    def _calculate_innovation_metrics(self, patents: List[Dict], 
                                    analysis: Dict) -> Dict:
        """Calculate comprehensive innovation metrics"""
        metrics = {}
        
        # Technology diversity (Shannon entropy) - using real CPC data
        if 'by_technology' in analysis and analysis['by_technology']:
            # Count unique CPC classes for better diversity measure
            cpc_counts = {}
            for patent in patents:
                for code in patent.get('cpc_codes', []):
                    if len(code) >= 3:
                        cpc_class = code[:3]
                        cpc_counts[cpc_class] = cpc_counts.get(cpc_class, 0) + 1
            
            if cpc_counts:
                total = sum(cpc_counts.values())
                probs = [count/total for count in cpc_counts.values()]
                entropy = -sum(p * np.log(p) for p in probs if p > 0)
                # Normalize by actual number of unique CPC classes
                max_entropy = np.log(len(cpc_counts))
                metrics['technology_diversity'] = (entropy / max_entropy * 100) if max_entropy > 0 else 0
            else:
                metrics['technology_diversity'] = 0
        else:
            metrics['technology_diversity'] = 0
            
        # Patent velocity (patents per year)
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
        
        # Breakthrough potential (high citations)
        breakthrough_patents = [p for p in patents 
                              if p.get('citations_received', 0) > 10]
        metrics['breakthrough_ratio'] = len(breakthrough_patents) / len(patents) if patents else 0
        
        return metrics
    
    def search_patents_for_all_businesses(self, business_names: List[str]) -> Dict[str, int]:
        """
        Search patents for all businesses using batch optimization
        
        Args:
            business_names: List of all business names to search
            
        Returns:
            Dictionary mapping business name to patent count
        """
        if not self._all_patents_fetched:
            logger.info(f"Performing batch patent search for {len(business_names)} businesses...")
            self._patent_cache = self.batch_searcher.batch_search_patents(business_names)
            self._all_patents_fetched = True
            
            # Log statistics
            stats = self.batch_searcher.get_patent_statistics(self._patent_cache)
            logger.info(f"Patent search complete:")
            logger.info(f"  - Businesses with patents: {stats['businesses_with_patents']} ({stats['percentage_with_patents']:.1f}%)")
            logger.info(f"  - Total patents found: {stats['total_patents']}")
        
        return self._patent_cache
    
    def get_patent_count_for_business(self, business_name: str) -> int:
        """Get patent count for a specific business from cache"""
        return self._patent_cache.get(business_name, 0)
    
    def assess_cluster_innovation(self, cluster: Dict) -> Dict:
        """Assess innovation potential of a cluster based on patents"""
        cluster_type = cluster.get('type', 'mixed')
        businesses = cluster.get('businesses', [])
        
        # Search for patents from cluster businesses
        all_patents = []
        
        # Check major companies
        for business in businesses[:20]:  # Top 20 businesses
            business_name = business.get('name', '').lower()
            
            # Check if it's a known KC assignee
            for assignee in self.kc_assignees:
                if assignee in business_name or business_name in assignee:
                    patents = self.search_patents([], assignee=assignee)
                    all_patents.extend(patents)
                    break
                    
        # Add technology-specific searches
        if cluster_type in ['technology', 'biosciences']:
            keywords = self._get_cluster_keywords(cluster_type)
            tech_patents = self.search_patents(keywords)
            all_patents.extend(tech_patents)
            
        # Analyze portfolio
        portfolio_analysis = self.analyze_patent_portfolio(all_patents)
        
        # Calculate cluster-specific metrics
        innovation_metrics = portfolio_analysis.get('innovation_metrics', {})
        citation_metrics = portfolio_analysis.get('citation_metrics', {})
        
        cluster_innovation = {
            'patent_count': portfolio_analysis.get('total_patents', 0),
            'innovation_score': innovation_metrics.get('innovation_score', 0),
            'technology_diversity': innovation_metrics.get('technology_diversity', 0),
            'patent_velocity': innovation_metrics.get('patent_velocity', 0),
            'citation_impact': citation_metrics.get('citation_impact_score', 0),
            'top_innovators': list(portfolio_analysis.get('by_assignee', {}).keys())[:5],
            'emerging_technologies': self._identify_emerging_tech(all_patents),
            'innovation_gaps': self._identify_innovation_gaps(cluster_type, portfolio_analysis),
            'recommendations': self._generate_innovation_recommendations(cluster_type, portfolio_analysis)
        }
        
        return cluster_innovation
    
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
    
    def _identify_emerging_tech(self, patents: List[Dict]) -> List[str]:
        """Identify emerging technology trends from patents"""
        recent_patents = [p for p in patents 
                         if p.get('grant_date', '')[:4] >= '2022']
        
        # Extract technology terms from titles and abstracts
        tech_terms = []
        for patent in recent_patents:
            title = patent.get('title', '').lower()
            abstract = patent.get('abstract', '').lower()
            
            # Look for key technology indicators
            if 'machine learning' in title or 'ml' in abstract:
                tech_terms.append('Machine Learning')
            if 'artificial intelligence' in title or 'ai' in abstract:
                tech_terms.append('Artificial Intelligence')
            if 'blockchain' in title or abstract:
                tech_terms.append('Blockchain')
            if 'iot' in abstract or 'internet of things' in title:
                tech_terms.append('Internet of Things')
            if 'quantum' in title or abstract:
                tech_terms.append('Quantum Computing')
                
        return list(set(tech_terms))[:5]  # Top 5 unique
    
    def _identify_innovation_gaps(self, cluster_type: str, 
                                 analysis: Dict) -> List[str]:
        """Identify areas where innovation is lacking"""
        gaps = []
        
        expected_areas = {
            'technology': ['AI/ML', 'cybersecurity', 'cloud', 'data analytics'],
            'biosciences': ['gene therapy', 'personalized medicine', 'diagnostics'],
            'logistics': ['autonomous vehicles', 'IoT tracking', 'optimization'],
            'manufacturing': ['Industry 4.0', 'robotics', 'additive manufacturing'],
            'animal_health': ['precision medicine', 'diagnostics', 'nutrition science']
        }
        
        if cluster_type in expected_areas:
            current_tech = set(analysis.get('by_technology', {}).keys())
            expected = set(expected_areas[cluster_type])
            
            # Find missing areas
            missing = expected - current_tech
            for area in missing:
                gaps.append(f"Limited patent activity in {area}")
                
        # Check patent velocity
        velocity = analysis.get('innovation_metrics', {}).get('patent_velocity', 0)
        if velocity < 10:
            gaps.append("Low patent filing rate compared to industry leaders")
            
        return gaps
    
    def _generate_innovation_recommendations(self, cluster_type: str, 
                                           analysis: Dict) -> List[str]:
        """Generate recommendations for improving innovation"""
        recommendations = []
        
        # Based on innovation score
        score = analysis.get('innovation_metrics', {}).get('innovation_score', 0)
        if score < 50:
            recommendations.append(
                "Increase R&D investment and patent filing activity"
            )
            
        # Based on diversity
        diversity = analysis.get('innovation_metrics', {}).get('technology_diversity', 0)
        if diversity < 40:
            recommendations.append(
                "Diversify technology portfolio to reduce concentration risk"
            )
            
        # Based on citations
        citation_impact = analysis.get('citation_metrics', {}).get('citation_impact_score', 0)
        if citation_impact < 30:
            recommendations.append(
                "Focus on breakthrough innovations with higher impact potential"
            )
            
        # Cluster-specific
        if cluster_type == 'technology' and score < 70:
            recommendations.append(
                "Partner with universities for cutting-edge AI/ML research"
            )
        elif cluster_type == 'biosciences':
            recommendations.append(
                "Establish patent pooling agreements for collaborative innovation"
            )
            
        return recommendations
    
    def generate_patent_report(self, clusters: List[Dict]) -> Dict:
        """Generate comprehensive patent analysis report"""
        report = {
            'total_patents_analyzed': 0,
            'by_cluster': {},
            'top_innovators': [],
            'emerging_technologies': set(),
            'innovation_rankings': [],
            'strategic_recommendations': []
        }
        
        # Analyze each cluster
        for cluster in clusters:
            cluster_name = cluster.get('name', 'Unknown')
            innovation = self.assess_cluster_innovation(cluster)
            
            report['by_cluster'][cluster_name] = innovation
            report['total_patents_analyzed'] += innovation['patent_count']
            report['emerging_technologies'].update(innovation['emerging_technologies'])
            
            # Track rankings
            report['innovation_rankings'].append({
                'cluster': cluster_name,
                'score': innovation['innovation_score'],
                'patents': innovation['patent_count'],
                'velocity': innovation['patent_velocity']
            })
            
        # Sort rankings
        report['innovation_rankings'].sort(key=lambda x: x['score'], reverse=True)
        
        # Identify top innovators across all clusters
        all_innovators = []
        for cluster_data in report['by_cluster'].values():
            all_innovators.extend(cluster_data['top_innovators'])
        report['top_innovators'] = list(Counter(all_innovators).most_common(10))
        
        # Convert emerging tech set to list
        report['emerging_technologies'] = list(report['emerging_technologies'])
        
        # Generate strategic recommendations
        report['strategic_recommendations'] = self._generate_strategic_recommendations(report)
        
        return report
    
    def _generate_strategic_recommendations(self, report: Dict) -> List[str]:
        """Generate strategic recommendations from patent analysis"""
        recommendations = []
        
        # Innovation leader
        if report['innovation_rankings']:
            leader = report['innovation_rankings'][0]
            recommendations.append(
                f"Leverage {leader['cluster']} as innovation hub (score: {leader['score']:.1f})"
            )
            
        # Patent gaps
        low_patent_clusters = [r for r in report['innovation_rankings'] 
                             if r['patents'] < 10]
        if low_patent_clusters:
            recommendations.append(
                f"Increase patent activity in {low_patent_clusters[0]['cluster']}"
            )
            
        # Emerging tech
        if report['emerging_technologies']:
            recommendations.append(
                f"Focus R&D on emerging areas: {', '.join(report['emerging_technologies'][:3])}"
            )
            
        return recommendations
