"""Improved map generator with clearer cluster visualization"""
import folium
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

def create_improved_cluster_layers(m: folium.Map, results: Dict):
    """Create individual toggleable layers for each discovered cluster"""
    
    clusters = results.get('steps', {}).get('cluster_optimization', {}).get('clusters', [])
    cluster_layers = []
    
    # Color map for cluster types
    cluster_colors = {
        "logistics": "blue",
        "biosciences": "green", 
        "technology": "purple",
        "manufacturing": "orange",
        "animal_health": "red",
        "finance": "darkblue",
        "healthcare": "pink",
        "agriculture": "beige",
        "retail": "lightgray",
        "professional_services": "cadetblue",
        "mixed": "gray"
    }
    
    logger.info(f"Creating individual layers for {len(clusters)} discovered clusters")
    
    # Create a parent group for all clusters
    all_clusters_group = folium.FeatureGroup(
        name=f"🎯 All Discovered Clusters ({len(clusters)})", 
        show=True
    )
    
    # Create individual layer for each cluster
    for i, cluster in enumerate(clusters, 1):
        cluster_name = cluster.get('name', f'Cluster {i}')
        cluster_type = cluster.get('type', 'mixed')
        color = cluster_colors.get(cluster_type, 'gray')
        
        # Create a feature group for this specific cluster
        cluster_group = folium.FeatureGroup(
            name=f"📍 {i}. {cluster_name}",
            show=True
        )
        
        # Add cluster visualization (circles, markers, etc.)
        businesses = cluster.get('businesses', [])
        gdp_impact = cluster.get('projected_gdp_impact', 0)
        job_count = cluster.get('projected_jobs', 0)
        
        # Group businesses by county
        county_groups = {}
        for business in businesses:
            county = business.get('county', 'Unknown')
            state = business.get('state', 'MO')
            county_key = f"{county} County, {state}"
            
            if county_key not in county_groups:
                county_groups[county_key] = []
            county_groups[county_key].append(business)
        
        # Add visualization for each county in this cluster
        for county_key, county_businesses in county_groups.items():
            # Create popup with cluster details
            popup_html = f"""
            <div style='width: 300px'>
                <h4>{cluster_name}</h4>
                <b>Type:</b> {cluster_type.replace('_', ' ').title()}<br>
                <b>Location:</b> {county_key}<br>
                <b>Businesses:</b> {len(county_businesses)}<br>
                <hr>
                <b>Economic Impact:</b><br>
                • GDP: ${gdp_impact:,.0f}<br>
                • Jobs: {job_count:,}<br>
                <hr>
                <b>Top Businesses:</b><br>
                <ul style='margin: 5px 0; padding-left: 20px;'>
            """
            
            # Add top 3 businesses
            for biz in county_businesses[:3]:
                popup_html += f"<li>{biz.get('name', 'Unknown')} ({biz.get('employees', 0)} employees)</li>"
            
            popup_html += "</ul></div>"
            
            # Add circle for this part of the cluster
            # You'll need actual coordinates - this is placeholder
            lat, lon = 39.0997, -94.5786  # KC center as placeholder
            
            folium.Circle(
                location=[lat, lon],
                radius=5000 + (gdp_impact / 1e9) * 1000,  # Size by impact
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{cluster_name} - {county_key}",
                color=color,
                fill=True,
                fillOpacity=0.4,
                weight=3
            ).add_to(cluster_group)
            
            # Add marker for cluster center
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=cluster_name,
                icon=folium.Icon(
                    color=color if color != 'gray' else 'lightgray',
                    icon='info-sign',
                    prefix='glyphicon'
                )
            ).add_to(cluster_group)
        
        # Add this cluster's layer to the map
        cluster_group.add_to(m)
        cluster_layers.append(cluster_group)
        
        # Also add to the all clusters group
        cluster_group.add_to(all_clusters_group)
    
    # Add the all clusters group
    all_clusters_group.add_to(m)
    
    # Add supplementary layers
    # Infrastructure layer
    infra_group = folium.FeatureGroup(
        name="🏗️ Infrastructure Assets", 
        show=False
    )
    # Add infrastructure markers here...
    infra_group.add_to(m)
    
    # County boundaries
    county_group = folium.FeatureGroup(
        name="🗺️ County Boundaries",
        show=False  
    )
    # Add county visualization here...
    county_group.add_to(m)
    
    logger.info(f"Created {len(cluster_layers)} individual cluster layers")
    
    return cluster_layers