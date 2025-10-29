"""Inter-Cluster Synergy Analysis based on Economic Development Best Practices"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
import logging

logger = logging.getLogger(__name__)

class InterClusterSynergyAnalyzer:
    """Analyzes synergies between clusters based on multiple economic factors"""
    
    def __init__(self):
        # Define supply chain relationships based on NAICS codes
        self.supply_chain_map = {
            # Manufacturing supply chains
            '336': ['332', '333', '326'],  # Auto manufacturing needs metal, machinery, plastics
            '332': ['331', '212'],          # Metal fabrication needs primary metals, mining
            '333': ['332', '335'],          # Machinery needs metal fabrication, electrical
            '311': ['111', '112', '115'],   # Food manufacturing needs agriculture
            '312': ['111'],                 # Beverage manufacturing needs agriculture
            '325': ['324', '211', '212'],   # Chemical manufacturing needs petroleum, mining
            '326': ['325', '324'],          # Plastics needs chemicals, petroleum
            
            # Service supply chains
            '541': ['511', '518'],          # Professional services needs software, data
            '511': ['334', '518'],          # Software needs computer hardware, data
            '493': ['484', '488'],          # Warehousing serves trucking, rail
            '488': ['482', '483'],          # Transportation support serves rail, water
            
            # Biotech/pharma chains
            '3254': ['325', '541'],         # Pharma needs chemicals, R&D services
            '5417': ['3254', '621'],        # Scientific R&D serves pharma, healthcare
        }
        
        # Adjacent Kansas City counties
        self.adjacent_counties = {
            'Jackson': ['Clay', 'Cass', 'Lafayette'],
            'Clay': ['Jackson', 'Platte', 'Ray'],
            'Platte': ['Clay', 'Buchanan'],
            'Johnson': ['Wyandotte', 'Miami', 'Douglas'],
            'Wyandotte': ['Johnson', 'Leavenworth'],
            'Cass': ['Jackson', 'Johnson', 'Bates'],
            'Lafayette': ['Jackson', 'Ray', 'Saline']
        }
    
    def calculate_comprehensive_synergy(self, cluster_i: Dict, cluster_j: Dict) -> Dict:
        """
        Calculate synergy between two clusters based on best practices
        
        Returns dict with:
        - total_score: 0-100 overall synergy
        - components: breakdown of each factor
        - recommendations: actionable insights
        """
        if cluster_i.get('name') == cluster_j.get('name'):
            return {
                'total_score': 100,
                'components': {'self': 100},
                'recommendations': []
            }
        
        components = {}
        
        # 1. Critical Mass Synergy (0-30 points)
        # Based on Porter's cluster theory - complete ecosystems create more value
        cm_score = self._calculate_critical_mass_synergy(cluster_i, cluster_j)
        components['critical_mass'] = cm_score
        
        # 2. Supply Chain Synergy (0-25 points)
        # Based on input-output economic models
        sc_score = self._calculate_supply_chain_synergy(cluster_i, cluster_j)
        components['supply_chain'] = sc_score
        
        # 3. Geographic Proximity (0-20 points)
        # Based on knowledge spillover research (Jaffe et al.)
        geo_score = self._calculate_geographic_synergy(cluster_i, cluster_j)
        components['geographic'] = geo_score
        
        # 4. Workforce Complementarity (0-15 points)
        # Based on labor market pooling benefits (Marshall)
        wf_score = self._calculate_workforce_synergy(cluster_i, cluster_j)
        components['workforce'] = wf_score
        
        # 5. Innovation Synergy (0-10 points)
        # Based on innovation ecosystem research
        inn_score = self._calculate_innovation_synergy(cluster_i, cluster_j)
        components['innovation'] = inn_score
        
        # Calculate total
        total_score = sum(components.values())
        
        # Apply competition discount for same-type clusters
        if cluster_i.get('type') == cluster_j.get('type'):
            total_score *= 0.8  # 20% reduction for competition
            components['competition_adjustment'] = -0.2
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            cluster_i, cluster_j, components
        )
        
        return {
            'total_score': min(total_score, 100),
            'components': components,
            'recommendations': recommendations
        }
    
    def _calculate_critical_mass_synergy(self, cluster_i: Dict, cluster_j: Dict) -> float:
        """
        Calculate synergy based on ecosystem completeness
        High Critical Mass clusters can better support each other
        """
        cm_i = cluster_i.get('critical_mass', 50)
        cm_j = cluster_j.get('critical_mass', 50)
        
        # Base score on average Critical Mass
        avg_cm = (cm_i + cm_j) / 2
        base_score = (avg_cm / 100) * 20  # Max 20 points
        
        # Bonus for both having high Critical Mass
        if cm_i > 70 and cm_j > 70:
            base_score += 10  # Strong ecosystems collaborate better
        elif cm_i > 60 and cm_j > 60:
            base_score += 5
        
        return min(base_score, 30)
    
    def _calculate_supply_chain_synergy(self, cluster_i: Dict, cluster_j: Dict) -> float:
        """
        Calculate synergy based on supply chain relationships
        Uses input-output analysis principles
        """
        businesses_i = cluster_i.get('businesses', [])
        businesses_j = cluster_j.get('businesses', [])
        
        if not businesses_i or not businesses_j:
            return 0
        
        # Get NAICS codes
        naics_i = set()
        naics_j = set()
        
        for b in businesses_i:
            naics = str(b.get('naics_code', ''))[:4]  # Use 4-digit for precision
            if naics:
                naics_i.add(naics)
        
        for b in businesses_j:
            naics = str(b.get('naics_code', ''))[:4]
            if naics:
                naics_j.add(naics)
        
        # Check for direct supply relationships
        synergy_score = 0
        relationships_found = []
        
        for supplier, customers in self.supply_chain_map.items():
            # Check if cluster i supplies to cluster j
            if any(supplier.startswith(n[:3]) for n in naics_i):
                for customer in customers:
                    if any(customer.startswith(n[:3]) for n in naics_j):
                        synergy_score += 8
                        relationships_found.append(f"{supplier}→{customer}")
            
            # Check reverse relationship
            if any(supplier.startswith(n[:3]) for n in naics_j):
                for customer in customers:
                    if any(customer.startswith(n[:3]) for n in naics_i):
                        synergy_score += 8
                        relationships_found.append(f"{supplier}→{customer}")
        
        # Cap at 25 points
        return min(synergy_score, 25)
    
    def _calculate_geographic_synergy(self, cluster_i: Dict, cluster_j: Dict) -> float:
        """
        Calculate synergy based on geographic proximity
        Based on knowledge spillover and labor market sharing research
        """
        businesses_i = cluster_i.get('businesses', [])
        businesses_j = cluster_j.get('businesses', [])
        
        if not businesses_i or not businesses_j:
            return 0
        
        # Get counties
        counties_i = set(b.get('county', '') for b in businesses_i if b.get('county'))
        counties_j = set(b.get('county', '') for b in businesses_j if b.get('county'))
        
        # Same county = highest synergy
        if counties_i.intersection(counties_j):
            overlap_pct = len(counties_i.intersection(counties_j)) / len(counties_i.union(counties_j))
            return 20 * overlap_pct  # Max 20 points for same county
        
        # Adjacent counties = moderate synergy
        adjacent_score = 0
        for county_i in counties_i:
            adjacent = self.adjacent_counties.get(county_i, [])
            if any(county_j in adjacent for county_j in counties_j):
                adjacent_score = 10  # Adjacent counties get 10 points
                break
        
        return adjacent_score
    
    def _calculate_workforce_synergy(self, cluster_i: Dict, cluster_j: Dict) -> float:
        """
        Calculate synergy based on workforce complementarity
        Different skill requirements can share labor market
        """
        # Get workforce characteristics
        stem_i = cluster_i.get('metrics', {}).get('stem_concentration', 50)
        stem_j = cluster_j.get('metrics', {}).get('stem_concentration', 50)
        
        avg_size_i = cluster_i.get('metrics', {}).get('avg_employees', 50)
        avg_size_j = cluster_j.get('metrics', {}).get('avg_employees', 50)
        
        synergy_score = 0
        
        # Complementary STEM levels (one high-tech, one traditional)
        stem_diff = abs(stem_i - stem_j)
        if 30 <= stem_diff <= 50:
            synergy_score += 8  # Good complementarity
        elif stem_diff > 50:
            synergy_score += 5  # Some complementarity
        
        # Similar business sizes can share workforce
        size_ratio = max(avg_size_i, avg_size_j) / max(min(avg_size_i, avg_size_j), 1)
        if size_ratio < 2:  # Similar sizes
            synergy_score += 7
        elif size_ratio < 3:
            synergy_score += 4
        
        return min(synergy_score, 15)
    
    def _calculate_innovation_synergy(self, cluster_i: Dict, cluster_j: Dict) -> float:
        """
        Calculate synergy based on innovation potential
        High-innovation clusters can cross-pollinate ideas
        """
        innovation_i = cluster_i.get('metrics', {}).get('innovation_score', 50)
        innovation_j = cluster_j.get('metrics', {}).get('innovation_score', 50)
        
        avg_innovation = (innovation_i + innovation_j) / 2
        
        # Both highly innovative
        if innovation_i > 70 and innovation_j > 70:
            return 10  # Maximum innovation synergy
        elif avg_innovation > 60:
            return 7
        elif avg_innovation > 40:
            return 4
        else:
            return 2
    
    def _generate_recommendations(self, cluster_i: Dict, cluster_j: Dict, 
                                 components: Dict) -> List[str]:
        """Generate actionable recommendations based on synergy analysis"""
        recommendations = []
        total_score = sum(v for v in components.values() if isinstance(v, (int, float)))
        
        if total_score >= 70:
            recommendations.append("Strong synergy potential - prioritize joint initiatives")
        elif total_score >= 40:
            recommendations.append("Moderate synergy - explore collaboration opportunities")
        else:
            recommendations.append("Limited synergy - focus on other partnerships")
        
        # Specific recommendations based on components
        if components.get('supply_chain', 0) > 15:
            recommendations.append("Establish formal supplier-customer relationships")
        
        if components.get('geographic', 0) > 10:
            recommendations.append("Leverage geographic proximity for shared infrastructure")
        
        if components.get('workforce', 0) > 10:
            recommendations.append("Create joint workforce development programs")
        
        if components.get('innovation', 0) > 7:
            recommendations.append("Facilitate R&D partnerships and knowledge exchange")
        
        if components.get('critical_mass', 0) > 20:
            recommendations.append("Both clusters have strong ecosystems - maximize network effects")
        
        return recommendations
    
    def create_synergy_matrix(self, clusters: List[Dict]) -> np.ndarray:
        """Create full synergy matrix for all clusters"""
        n = len(clusters)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                synergy = self.calculate_comprehensive_synergy(clusters[i], clusters[j])
                matrix[i][j] = synergy['total_score']
        
        return matrix