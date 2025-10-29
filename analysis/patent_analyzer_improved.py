"""Improved patent analyzer that searches ALL businesses efficiently"""

import logging
import pandas as pd
from typing import Dict, List, Optional
from data_collection.patent_batch_optimizer import BatchPatentSearcher
from datetime import datetime
import os
import json

logger = logging.getLogger(__name__)

class ImprovedPatentAnalyzer:
    """Patent analyzer that actually checks ALL businesses, not just top 20"""
    
    def __init__(self, config):
        self.config = config
        self.batch_searcher = BatchPatentSearcher(config)
        self.patent_data_file = 'data/kc_patent_mapping.json'
        self.patent_mapping = None
        self._progress_cb = None

    def set_progress_callback(self, cb):
        """Set a progress callback that receives (percent:int, message:str)."""
        self._progress_cb = cb
        try:
            if hasattr(self.batch_searcher, 'set_progress_callback'):
                self.batch_searcher.set_progress_callback(cb)
        except Exception:
            pass
        
    def analyze_all_businesses(self, force_refresh: bool = False, business_names: List[str] = None, progress_callback=None, org_only: bool = False) -> Dict[str, int]:
        """
        Analyze patents for businesses
        
        Args:
            force_refresh: If True, fetch fresh data even if cache exists
            business_names: Optional list of specific business names to search. 
                          If None, searches all businesses in CSV
            
        Returns:
            Mapping of business name to patent count
        """
        # Wire progress callback if provided
        if progress_callback is not None:
            self.set_progress_callback(progress_callback)

        # Check if we have recent patent data and not doing subset search
        if not force_refresh and business_names is None and self._has_recent_patent_data():
            logger.info("Loading existing patent data...")
            return self._load_patent_data()
        
        # Load business names if not provided
        if business_names is None:
            logger.info("Loading all KC businesses...")
            df = pd.read_csv('data/kc_businesses_all.csv')
            business_names = df['name'].unique().tolist()
        else:
            logger.info(f"Searching patents for {len(business_names):,} specified businesses...")
        
        logger.info(f"Starting patent search for {len(business_names):,} businesses...")
        if self._progress_cb:
            try:
                self._progress_cb(10, f"Preparing KC-area patent fetch for {len(business_names):,} businesses")
            except Exception:
                pass
        
        # Use batch searcher to efficiently find patents
        if business_names is not None and org_only:
            # Quick-mode fast path: only fetch patents for these orgs
            patent_mapping = self.batch_searcher.batch_search_patents_for_orgs(business_names)
        else:
            patent_mapping = self.batch_searcher.batch_search_patents(business_names)
        
        # Save results
        self._save_patent_data(patent_mapping)
        if self._progress_cb:
            try:
                self._progress_cb(95, "Patent analysis complete; finalizing indexes")
            except Exception:
                pass
        
        # Log statistics
        stats = self.batch_searcher.get_patent_statistics(patent_mapping)
        self._log_statistics(stats)
        
        return patent_mapping
    
    def get_cluster_patent_analysis(self, cluster: Dict, patent_mapping: Dict[str, int]) -> Dict:
        """
        Analyze patents for a specific cluster using pre-fetched data
        
        Args:
            cluster: Cluster dictionary with businesses
            patent_mapping: Pre-fetched patent counts for all businesses
            
        Returns:
            Detailed patent analysis for the cluster
        """
        businesses = cluster.get('businesses', [])
        cluster_type = cluster.get('type', 'mixed')
        
        # Calculate patent metrics for ALL businesses in cluster
        patent_holders = []
        total_patents = 0
        
        for business in businesses:  # ALL businesses, not just top 20!
            name = business.get('name', '')
            patent_count = patent_mapping.get(name, 0)
            
            if patent_count > 0:
                patent_holders.append({
                    'name': name,
                    'patents': patent_count,
                    'naics': business.get('naics_code', ''),
                    'employees': business.get('employees', 0),
                    'revenue': business.get('revenue', 0)
                })
                total_patents += patent_count
        
        # Sort by patent count
        patent_holders.sort(key=lambda x: x['patents'], reverse=True)
        
        # Calculate metrics
        businesses_with_patents = len(patent_holders)
        patent_percentage = (businesses_with_patents / len(businesses) * 100) if businesses else 0
        
        # Innovation concentration (Gini coefficient)
        innovation_concentration = self._calculate_innovation_concentration(
            [h['patents'] for h in patent_holders]
        )
        
        return {
            'total_businesses': len(businesses),
            'businesses_with_patents': businesses_with_patents,
            'patent_percentage': patent_percentage,
            'total_patents': total_patents,
            'avg_patents_per_holder': total_patents / businesses_with_patents if businesses_with_patents else 0,
            'top_10_innovators': patent_holders[:10],
            'innovation_concentration': innovation_concentration,
            'cluster_type': cluster_type,
            'innovation_score': self._calculate_innovation_score(
                patent_percentage, total_patents, innovation_concentration
            )
        }
    
    def _calculate_innovation_score(self, patent_percentage: float, 
                                  total_patents: int, 
                                  concentration: float) -> float:
        """Calculate composite innovation score"""
        # Higher percentage of businesses with patents = better
        percentage_score = min(patent_percentage / 10, 1.0) * 40  # Max 40 points
        
        # More total patents = better (logarithmic scale)
        import math
        patent_score = min(math.log10(total_patents + 1) / 3, 1.0) * 40  # Max 40 points
        
        # Lower concentration = better (more distributed innovation)
        distribution_score = (1 - concentration) * 20  # Max 20 points
        
        return percentage_score + patent_score + distribution_score
    
    def _calculate_innovation_concentration(self, patent_counts: List[int]) -> float:
        """Calculate Gini coefficient for innovation concentration"""
        if not patent_counts:
            return 0.0
            
        sorted_counts = sorted(patent_counts)
        n = len(sorted_counts)
        cumsum = 0
        
        for i, count in enumerate(sorted_counts):
            cumsum += (n - i) * count
            
        return (n + 1 - 2 * cumsum / sum(sorted_counts)) / n if sum(sorted_counts) > 0 else 0
    
    def _has_recent_patent_data(self) -> bool:
        """Check if we have patent data from the last 30 days"""
        if not os.path.exists(self.patent_data_file):
            return False
            
        # Check file age
        file_age_days = (datetime.now().timestamp() - 
                        os.path.getmtime(self.patent_data_file)) / 86400
        
        return file_age_days < 30
    
    def _load_patent_data(self) -> Dict[str, int]:
        """Load saved patent data"""
        with open(self.patent_data_file, 'r') as f:
            return json.load(f)
    
    def _save_patent_data(self, patent_mapping: Dict[str, int]):
        """Save patent data with metadata"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'total_businesses': len(patent_mapping),
            'businesses_with_patents': sum(1 for count in patent_mapping.values() if count > 0),
            'patent_mapping': patent_mapping
        }
        
        with open(self.patent_data_file, 'w') as f:
            json.dump(data, f)
            
        logger.info(f"Saved patent data to {self.patent_data_file}")
    
    def _log_statistics(self, stats: Dict):
        """Log detailed statistics"""
        logger.info("=== PATENT ANALYSIS COMPLETE ===")
        logger.info(f"Total businesses analyzed: {stats['total_businesses']:,}")
        logger.info(f"Businesses with patents: {stats['businesses_with_patents']:,} ({stats['percentage_with_patents']:.1f}%)")
        logger.info(f"Total patents found: {stats['total_patents']:,}")
        logger.info(f"Average patents per holder: {stats['average_patents_per_holder']:.1f}")
        
        logger.info("\nTop 10 Patent Holders:")
        for i, (name, count) in enumerate(stats['top_10_patent_holders'], 1):
            logger.info(f"  {i}. {name}: {count} patents")

# Usage example:
def main():
    from config import Config
    
    config = Config()
    analyzer = ImprovedPatentAnalyzer(config)
    
    # First, analyze all businesses (this takes ~1-2 hours)
    patent_mapping = analyzer.analyze_all_businesses()
    
    # Then use the mapping for cluster analysis (instant)
    # cluster = {...}  # Your cluster data
    # cluster_analysis = analyzer.get_cluster_patent_analysis(cluster, patent_mapping)
    
if __name__ == '__main__':
    main()
