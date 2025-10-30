"""Improved cluster visualization with clearer representation"""

import folium
from folium import plugins
import numpy as np
from typing import Dict, List, Tuple
import logging
from shapely.geometry import Point, Polygon, MultiPoint
from shapely.ops import unary_union
import json

logger = logging.getLogger(__name__)

class ImprovedClusterVisualizer:
    """Enhanced cluster visualization with multiple representation options"""
    
    def __init__(self):
        self.kc_center = (39.0997, -94.5786)
        
        # Enhanced color scheme with better contrast
        self.cluster_colors = {
            "logistics": "#1E88E5",         # Bright Blue
            "biosciences": "#43A047",       # Green
            "technology": "#8E24AA",        # Purple
            "manufacturing": "#FB8C00",     # Orange
            "animal_health": "#E53935",     # Red
            "finance": "#00ACC1",           # Cyan
            "healthcare": "#D81B60",        # Pink
            "mixed": "#757575",             # Gray
        }
        
        # Visual styles
        self.styles = {
            'cluster_boundary': {
                'weight': 3,
                'opacity': 0.8,
                'fillOpacity': 0.2,
                'dashArray': '10, 5'
            },
            'cluster_core': {
                'weight': 4,
                'opacity': 1.0,
                'fillOpacity': 0.4
            },
            'business_marker': {
                'radius': 8,
                'weight': 2,
                'opacity': 0.8,
                'fillOpacity': 0.6
            }
        }
    
    def create_enhanced_cluster_map(self, results: Dict, visualization_mode: str = 'boundaries') -> str:
        """Create map with selected visualization mode"""
        
        # Initialize map
        m = folium.Map(
            location=self.kc_center,
            zoom_start=10,
            tiles='OpenStreetMap',
            prefer_canvas=True
        )
        
        clusters = results.get('steps', {}).get('cluster_optimization', {}).get('clusters', [])
        
        if visualization_mode == 'boundaries':
            self._add_cluster_boundaries(m, clusters)
        elif visualization_mode == 'density':
            self._add_density_visualization(m, clusters)
        elif visualization_mode == 'network':
            self._add_network_visualization(m, clusters)
        elif visualization_mode == 'impact':
            self._add_impact_visualization(m, clusters)
        else:
            self._add_hybrid_visualization(m, clusters)
        
        # Add controls and legend
        self._add_enhanced_controls(m, clusters)
        self._add_interactive_legend(m, clusters)
        
        return m._repr_html_()
    
    def _add_cluster_boundaries(self, m: folium.Map, clusters: List[Dict]):
        """Add clear cluster boundaries using convex hulls"""
        
        for i, cluster in enumerate(clusters):
            cluster_name = cluster.get('name', f'Cluster {i+1}')
            cluster_type = cluster.get('type', 'mixed')
            color = self.cluster_colors.get(cluster_type, '#757575')
            
            businesses = cluster.get('businesses', [])
            if not businesses:
                continue
            
            # Group businesses by proximity to create sub-clusters
            sub_clusters = self._create_subclusters(businesses)
            
            for j, sub_cluster in enumerate(sub_clusters):
                if len(sub_cluster) >= 3:
                    # Create boundary polygon
                    boundary = self._create_smooth_boundary(sub_cluster)
                    
                    if boundary:
                        # Add boundary polygon
                        folium.Polygon(
                            locations=boundary,
                            color=color,
                            fill=True,
                            fillColor=color,
                            **self.styles['cluster_boundary'],
                            popup=self._create_cluster_popup(cluster, sub_cluster),
                            tooltip=f"{cluster_name} - Area {j+1}"
                        ).add_to(m)
                        
                        # Add cluster center marker
                        center = self._calculate_cluster_center(sub_cluster)
                        self._add_cluster_center_marker(m, center, cluster, len(sub_cluster))
    
    def _add_density_visualization(self, m: folium.Map, clusters: List[Dict]):
        """Add kernel density estimation visualization"""
        
        # Collect all business locations by cluster
        for cluster in clusters:
            cluster_type = cluster.get('type', 'mixed')
            color = self.cluster_colors.get(cluster_type, '#757575')
            
            businesses = cluster.get('businesses', [])
            heat_data = []
            
            for business in businesses:
                if 'lat' in business and 'lon' in business:
                    # Weight by business score and size
                    weight = (business.get('composite_score', 50) / 100) * \
                            (1 + np.log10(max(business.get('employees', 1), 1)))
                    heat_data.append([business['lat'], business['lon'], weight])
            
            if heat_data:
                # Add gradient heatmap for this cluster
                gradient = {
                    0.0: '#FFFFFF',
                    0.5: color + '80',  # Semi-transparent
                    1.0: color
                }
                
                plugins.HeatMap(
                    heat_data,
                    radius=20,
                    blur=15,
                    gradient=gradient,
                    overlay=True,
                    control=True,
                    name=cluster.get('name', 'Cluster')
                ).add_to(m)
    
    def _add_impact_visualization(self, m: folium.Map, clusters: List[Dict]):
        """Add 3D-style impact visualization"""
        
        for cluster in clusters:
            cluster_type = cluster.get('type', 'mixed')
            color = self.cluster_colors.get(cluster_type, '#757575')
            
            # Calculate cluster metrics
            gdp_impact = cluster.get('projected_gdp_impact', 0)
            job_impact = cluster.get('projected_jobs', 0)
            businesses = cluster.get('businesses', [])
            
            # Group by geographic area
            area_groups = self._group_by_area(businesses)
            
            for area, area_businesses in area_groups.items():
                center = self._calculate_center(area_businesses)
                
                # Create layered circles for 3D effect
                for layer in range(3):
                    radius = 3000 + (gdp_impact / 1e8) * 500 - (layer * 500)
                    opacity = 0.6 - (layer * 0.2)
                    
                    folium.Circle(
                        location=center,
                        radius=radius,
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=opacity,
                        weight=2,
                        popup=self._create_impact_popup(cluster, area_businesses, gdp_impact, job_impact)
                    ).add_to(m)
                
                # Add metric marker at center
                self._add_metric_marker(m, center, cluster, len(area_businesses))
    
    def _add_hybrid_visualization(self, m: folium.Map, clusters: List[Dict]):
        """Add hybrid visualization combining multiple techniques"""
        
        # Create feature groups for different visualization layers
        boundary_layer = folium.FeatureGroup(name="Cluster Boundaries", show=True)
        density_layer = folium.FeatureGroup(name="Density Heatmap", show=False)
        business_layer = folium.FeatureGroup(name="Business Locations", show=True)
        
        for i, cluster in enumerate(clusters):
            cluster_name = cluster.get('name', f'Cluster {i+1}')
            cluster_type = cluster.get('type', 'mixed')
            color = self.cluster_colors.get(cluster_type, '#757575')
            businesses = cluster.get('businesses', [])
            
            if not businesses:
                continue
            
            # 1. Add subtle boundary
            sub_clusters = self._create_subclusters(businesses)
            for sub_cluster in sub_clusters:
                if len(sub_cluster) >= 3:
                    boundary = self._create_smooth_boundary(sub_cluster)
                    if boundary:
                        folium.Polygon(
                            locations=boundary,
                            color=color,
                            weight=2,
                            opacity=0.6,
                            fill=True,
                            fillColor=color,
                            fillOpacity=0.15,
                            dashArray='5, 5'
                        ).add_to(boundary_layer)
            
            # 2. Add density overlay
            heat_data = [[b['lat'], b['lon'], 1] for b in businesses 
                        if 'lat' in b and 'lon' in b]
            if heat_data:
                # Create mini heatmap for cluster
                folium.plugins.HeatMap(
                    heat_data,
                    radius=15,
                    blur=10,
                    max_zoom=13,
                    gradient={0.4: color + '00', 0.8: color + '80', 1: color}
                ).add_to(density_layer)
            
            # 3. Add smart business markers
            self._add_smart_business_markers(business_layer, businesses, cluster, color)
        
        # Add layers to map
        boundary_layer.add_to(m)
        density_layer.add_to(m)
        business_layer.add_to(m)
        
        # Add layer control
        folium.LayerControl(position='topright').add_to(m)
    
    def _create_subclusters(self, businesses: List[Dict], max_distance: float = 0.05) -> List[List[Dict]]:
        """Group businesses into geographic sub-clusters"""
        from sklearn.cluster import DBSCAN
        
        # Extract coordinates
        coords = []
        valid_businesses = []
        for b in businesses:
            if 'lat' in b and 'lon' in b:
                coords.append([b['lat'], b['lon']])
                valid_businesses.append(b)
        
        if len(coords) < 3:
            return [valid_businesses]
        
        # Use DBSCAN for geographic clustering
        coords_array = np.array(coords)
        clustering = DBSCAN(eps=max_distance, min_samples=3).fit(coords_array)
        
        # Group businesses by cluster label
        sub_clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label == -1:  # Noise points
                continue
            if label not in sub_clusters:
                sub_clusters[label] = []
            sub_clusters[label].append(valid_businesses[idx])
        
        return list(sub_clusters.values())
    
    def _create_smooth_boundary(self, businesses: List[Dict], buffer: float = 0.02) -> List[Tuple[float, float]]:
        """Create smooth boundary around businesses using buffered convex hull"""
        try:
            points = []
            for b in businesses:
                if 'lat' in b and 'lon' in b:
                    points.append(Point(b['lon'], b['lat']))
            
            if len(points) < 3:
                return None
            
            # Create multipoint and get convex hull
            multipoint = MultiPoint(points)
            hull = multipoint.convex_hull
            
            # Buffer for smoother boundary
            buffered = hull.buffer(buffer)
            
            # Extract coordinates
            if hasattr(buffered, 'exterior'):
                coords = list(buffered.exterior.coords)
                # Convert back to lat/lon format
                return [(lat, lon) for lon, lat in coords]
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating boundary: {e}")
            return None
    
    def _add_cluster_center_marker(self, m: folium.Map, center: Tuple[float, float], 
                                 cluster: Dict, business_count: int):
        """Add informative marker at cluster center"""
        
        cluster_type = cluster.get('type', 'mixed')
        color = self.cluster_colors.get(cluster_type, '#757575')
        
        # Create custom icon with metrics
        icon_html = f"""
        <div style='position: relative;'>
            <div style='background: {color}; border-radius: 50%; width: 50px; height: 50px; 
                        display: flex; align-items: center; justify-content: center;
                        border: 3px solid white; box-shadow: 0 2px 6px rgba(0,0,0,0.3);'>
                <span style='color: white; font-weight: bold; font-size: 16px;'>
                    {business_count}
                </span>
            </div>
            <div style='position: absolute; top: -8px; right: -8px; background: #FFD700; 
                        border-radius: 50%; width: 20px; height: 20px; 
                        display: flex; align-items: center; justify-content: center;
                        border: 2px solid white; font-size: 10px; font-weight: bold;'>
                #{cluster.get('rank', 0)}
            </div>
        </div>
        """
        
        folium.Marker(
            location=center,
            icon=folium.DivIcon(html=icon_html, icon_size=(50, 50), icon_anchor=(25, 25)),
            popup=self._create_detailed_popup(cluster),
            tooltip=f"{cluster.get('name')} - {business_count} businesses"
        ).add_to(m)
    
    def _add_enhanced_controls(self, m: folium.Map, clusters: List[Dict]):
        """Add enhanced map controls"""
        
        # Add cluster filter controls
        filter_html = """
        <div style='position: fixed; top: 80px; right: 10px; z-index: 1000; 
                    background: white; padding: 10px; border-radius: 5px; 
                    box-shadow: 0 2px 6px rgba(0,0,0,0.3); width: 200px;'>
            <h5 style='margin: 0 0 10px 0;'>Filter Clusters</h5>
            <div id='cluster-filters'>
                <!-- Filters will be added by JavaScript -->
            </div>
            <hr style='margin: 10px 0;'>
            <button onclick='applyFilters()' style='width: 100%; padding: 5px; 
                    background: #2196F3; color: white; border: none; 
                    border-radius: 3px; cursor: pointer;'>
                Apply Filters
            </button>
        </div>
        
        <script>
        // Add filter checkboxes for each cluster type
        const clusterTypes = %s;
        const filterDiv = document.getElementById('cluster-filters');
        
        clusterTypes.forEach(type => {
            const label = document.createElement('label');
            label.style.display = 'block';
            label.style.marginBottom = '5px';
            label.innerHTML = `
                <input type='checkbox' checked value='${type}' 
                       style='margin-right: 5px;'>
                ${type.replace('_', ' ').charAt(0).toUpperCase() + type.slice(1)}
            `;
            filterDiv.appendChild(label);
        });
        
        function applyFilters() {
            // Implementation would filter map layers
            console.log('Applying filters...');
        }
        </script>
        """ % json.dumps(list(set(c.get('type', 'mixed') for c in clusters)))
        
        m.get_root().html.add_child(folium.Element(filter_html))
    
    def _add_interactive_legend(self, m: folium.Map, clusters: List[Dict]):
        """Add interactive legend with cluster information"""
        
        legend_html = """
        <div style='position: fixed; bottom: 30px; left: 10px; z-index: 1000; 
                    background: white; padding: 15px; border-radius: 5px; 
                    box-shadow: 0 2px 6px rgba(0,0,0,0.3); max-width: 300px;'>
            <h5 style='margin: 0 0 10px 0;'>Cluster Analysis Results</h5>
            
            <div style='font-size: 12px;'>
                <b>Total Clusters:</b> %d<br>
                <b>Total Businesses:</b> %d<br>
                <b>Projected Impact:</b><br>
                • GDP: $%s<br>
                • Jobs: %s<br>
            </div>
            
            <hr style='margin: 10px 0;'>
            
            <div style='font-size: 11px;'>
                <b>Visualization Guide:</b><br>
                • Circle size = Economic impact<br>
                • Color intensity = Business density<br>
                • Number = Business count<br>
                • Gold badge = Cluster rank<br>
            </div>
            
            <hr style='margin: 10px 0;'>
            
            <div style='font-size: 11px;'>
                <b>Cluster Types:</b><br>
                %s
            </div>
        </div>
        """ % (
            len(clusters),
            sum(c.get('business_count', 0) for c in clusters),
            self._format_number(sum(c.get('projected_gdp_impact', 0) for c in clusters)),
            self._format_number(sum(c.get('projected_jobs', 0) for c in clusters)),
            self._create_legend_items(clusters)
        )
        
        m.get_root().html.add_child(folium.Element(legend_html))
    
    def _format_number(self, num: float) -> str:
        """Format large numbers for display"""
        if num >= 1e9:
            return f"{num/1e9:.1f}B"
        elif num >= 1e6:
            return f"{num/1e6:.1f}M"
        elif num >= 1e3:
            return f"{num/1e3:.1f}K"
        else:
            return f"{num:.0f}"
    
    def _create_legend_items(self, clusters: List[Dict]) -> str:
        """Create legend items for cluster types"""
        cluster_types = {}
        for c in clusters:
            ctype = c.get('type', 'mixed')
            if ctype not in cluster_types:
                cluster_types[ctype] = 0
            cluster_types[ctype] += 1
        
        items = []
        for ctype, count in cluster_types.items():
            color = self.cluster_colors.get(ctype, '#757575')
            items.append(
                f"<span style='color: {color}; font-size: 16px;'>●</span> "
                f"{ctype.replace('_', ' ').title()} ({count})"
            )
        
        return "<br>".join(items)
    
    def _create_detailed_popup(self, cluster: Dict) -> str:
        """Create detailed popup content for cluster"""
        return f"""
        <div style='width: 350px; font-family: Arial, sans-serif;'>
            <h4 style='margin: 0 0 10px 0; color: #333;'>{cluster.get('name', 'Cluster')}</h4>
            
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px;'>
                <div style='background: #f5f5f5; padding: 8px; border-radius: 4px;'>
                    <div style='font-size: 11px; color: #666;'>Businesses</div>
                    <div style='font-size: 18px; font-weight: bold; color: #333;'>
                        {cluster.get('business_count', 0)}
                    </div>
                </div>
                <div style='background: #f5f5f5; padding: 8px; border-radius: 4px;'>
                    <div style='font-size: 11px; color: #666;'>Employees</div>
                    <div style='font-size: 18px; font-weight: bold; color: #333;'>
                        {cluster.get('total_employees', 0):,}
                    </div>
                </div>
            </div>
            
            <div style='background: #e3f2fd; padding: 10px; border-radius: 4px; margin-bottom: 10px;'>
                <h5 style='margin: 0 0 5px 0; color: #1976d2;'>Economic Impact</h5>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 5px;'>
                    <div>
                        <span style='font-size: 11px; color: #666;'>GDP Impact:</span><br>
                        <span style='font-weight: bold;'>${cluster.get('projected_gdp_impact', 0):,.0f}</span>
                    </div>
                    <div>
                        <span style='font-size: 11px; color: #666;'>Job Creation:</span><br>
                        <span style='font-weight: bold;'>{cluster.get('projected_jobs', 0):,}</span>
                    </div>
                </div>
            </div>
            
            <div style='font-size: 12px; color: #666;'>
                <b>Cluster Score:</b> {cluster.get('cluster_score', 0):.1f}/100<br>
                <b>Type:</b> {cluster.get('type', 'mixed').replace('_', ' ').title()}<br>
                <b>Rank:</b> #{cluster.get('rank', 'N/A')} of total clusters
            </div>
        </div>
        """
    
    def _calculate_cluster_center(self, businesses: List[Dict]) -> Tuple[float, float]:
        """Calculate the geographic center of a cluster"""
        lats = [b['lat'] for b in businesses if 'lat' in b]
        lons = [b['lon'] for b in businesses if 'lon' in b]
        
        if lats and lons:
            return (sum(lats) / len(lats), sum(lons) / len(lons))
        return self.kc_center
    
    def _group_by_area(self, businesses: List[Dict]) -> Dict[str, List[Dict]]:
        """Group businesses by geographic area"""
        areas = {}
        for b in businesses:
            county = b.get('county', 'Unknown')
            if county not in areas:
                areas[county] = []
            areas[county].append(b)
        return areas
    
    def _calculate_center(self, businesses: List[Dict]) -> Tuple[float, float]:
        """Calculate center point of businesses"""
        return self._calculate_cluster_center(businesses)
    
    def _create_impact_popup(self, cluster: Dict, businesses: List[Dict], 
                           gdp_impact: float, job_impact: int) -> str:
        """Create popup for impact visualization"""
        return self._create_detailed_popup(cluster)
    
    def _add_metric_marker(self, m: folium.Map, location: Tuple[float, float], 
                         cluster: Dict, business_count: int):
        """Add marker with key metrics"""
        self._add_cluster_center_marker(m, location, cluster, business_count)
    
    def _add_smart_business_markers(self, layer: folium.FeatureGroup, 
                                  businesses: List[Dict], cluster: Dict, color: str):
        """Add intelligent business markers that avoid overlap"""
        
        # Sort businesses by importance
        sorted_businesses = sorted(
            businesses,
            key=lambda b: b.get('composite_score', 0) * b.get('employees', 1),
            reverse=True
        )
        
        # Add markers for top businesses only
        for i, business in enumerate(sorted_businesses[:20]):
            if 'lat' in business and 'lon' in business:
                # Determine marker size and style based on ranking
                if i < 3:
                    icon = 'star'
                    marker_color = 'gold'
                    size = 12
                elif i < 10:
                    icon = 'certificate'
                    marker_color = color
                    size = 10
                else:
                    icon = 'circle'
                    marker_color = color
                    size = 8
                
                folium.CircleMarker(
                    location=[business['lat'], business['lon']],
                    radius=size,
                    popup=self._create_business_popup(business, cluster),
                    tooltip=f"{business.get('name', 'Business')} - Score: {business.get('composite_score', 0):.0f}",
                    color=marker_color,
                    fill=True,
                    fillColor=marker_color,
                    fillOpacity=0.7,
                    weight=2
                ).add_to(layer)
    
    def _create_business_popup(self, business: Dict, cluster: Dict) -> str:
        """Create popup for individual business"""
        return f"""
        <div style='width: 250px;'>
            <h5 style='margin: 0 0 5px 0;'>{business.get('name', 'Business')}</h5>
            <div style='font-size: 12px;'>
                <b>Cluster:</b> {cluster.get('name', 'Unknown')}<br>
                <b>Industry:</b> {business.get('naics_code', 'N/A')}<br>
                <b>Employees:</b> {business.get('employees', 0)}<br>
                <b>Score:</b> {business.get('composite_score', 0):.1f}/100<br>
                <b>Revenue Est:</b> ${business.get('revenue_estimate', 0):,.0f}
            </div>
        </div>
        """
