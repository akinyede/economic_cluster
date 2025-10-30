# -*- coding: utf-8 -*-
"""Enhanced geographic visualization module for KC Cluster Prediction Tool with improved boundaries and interactive features"""
# Apply Folium CDN fixes before importing
from utils.folium_fix import patch_folium_cdns

import folium
from folium import plugins
import json
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import math
import random
import hashlib
from utils.kc_county_boundaries import KC_COUNTY_BOUNDARIES, COUNTY_CLASSIFICATION
from config import Config as AppConfig
import os
import time
import requests
from pathlib import Path

# Import new enhancement modules
logger = logging.getLogger(__name__)

try:
    from utils.geographic_validator import GeographicValidator
    from utils.enhanced_color_scheme import EnhancedColorScheme
    from utils.interactive_map_features import InteractiveMapFeatures
    from utils.spatial_accuracy_enhancer import SpatialAccuracyEnhancer
    ENHANCEMENT_MODULES_AVAILABLE = True
    logger.info("Enhanced visualization modules loaded successfully")
except ImportError as e:
    logger.warning(f"Enhanced visualization modules not available: {e}")
    ENHANCEMENT_MODULES_AVAILABLE = False

# Import improved visualization capabilities
try:
    from shapely.geometry import Point, Polygon, MultiPoint, shape
    from shapely.ops import unary_union
    from sklearn.cluster import DBSCAN
    SHAPELY_AVAILABLE = True
except ImportError:
    logger.warning("Shapely or sklearn not available - using fallback visualization")
    SHAPELY_AVAILABLE = False

class MapGenerator:
    """Generate interactive maps for cluster visualization"""
    
    def __init__(self):
        # Kansas City MSA center coordinates
        self.kc_center = (39.0997, -94.5786)
        
        # Visualization mode
        self.visualization_mode = 'improved' if SHAPELY_AVAILABLE else 'legacy'
        
        # Initialize enhancement modules if available
        if ENHANCEMENT_MODULES_AVAILABLE:
            self.geographic_validator = GeographicValidator()
            self.enhanced_color_scheme = EnhancedColorScheme()
            self.interactive_features = InteractiveMapFeatures()
            self.spatial_enhancer = SpatialAccuracyEnhancer()
            logger.info("Enhanced visualization modules initialized")
        else:
            self.geographic_validator = None
            self.enhanced_color_scheme = None
            self.interactive_features = None
            self.spatial_enhancer = None
            logger.warning("Using legacy visualization - enhancement modules not available")
        
        # Enhanced color scheme for better contrast (fallback if enhancement modules not available)
        self.enhanced_cluster_colors = {
            "logistics": "#1E88E5",         # Bright Blue
            "biosciences": "#43A047",       # Green
            "technology": "#8E24AA",        # Purple
            "manufacturing": "#FB8C00",     # Orange
            "animal_health": "#E53935",     # Red
            "finance": "#00ACC1",           # Cyan
            "healthcare": "#D81B60",        # Pink
            "mixed": "#757575",             # Gray
            # Combined types
            "supply_chain": "#0D47A1",      # Dark Blue
            "biotech": "#1B5E20",           # Dark Green
            "advanced_manufacturing": "#B71C1C",  # Dark Red
        }
        
        # County coordinates (approximate centers)
        self.county_coords = {
            # Kansas counties
            "Johnson County, KS": (38.8814, -94.8191),
            "Leavenworth County, KS": (39.2695, -95.0132),
            "Linn County, KS": (38.2806, -94.8440),
            "Miami County, KS": (38.6373, -94.8797),
            "Wyandotte County, KS": (39.1178, -94.7479),
            # Missouri counties
            "Bates County, MO": (38.2609, -94.3411),
            "Caldwell County, MO": (39.6586, -93.9916),
            "Cass County, MO": (38.6467, -94.3480),
            "Clay County, MO": (39.3072, -94.4191),
            "Clinton County, MO": (39.6586, -94.3968),
            "Jackson County, MO": (39.0119, -94.3633),
            "Lafayette County, MO": (39.0492, -93.7755),
            "Platte County, MO": (39.3697, -94.7633),
            "Ray County, MO": (39.3583, -94.0663)
        }
        
        # County boundaries (simplified - in production would use actual GeoJSON)
        self.county_colors = {
            "urban": "#FF6B6B",     # Red for urban
            "suburban": "#4ECDC4",  # Teal for suburban  
            "rural": "#95E1D3"      # Light green for rural
        }
        
        # Infrastructure icons
        self.infrastructure_icons = {
            "rail": "train",
            "highway": "road",
            "airport": "plane",
            "university": "graduation-cap",
            "logistics_park": "warehouse",
            "hospital": "hospital",
            "school": "school",
            "transit": "bus",
            "power": "bolt"
        }

        # Cache dir for OSM overlays
        try:
            base_dir = Path(__file__).resolve().parents[1]
        except Exception:
            base_dir = Path('.')
        self._osm_cache_dir = base_dir / 'analysis_output' / 'osm_cache'
        self._kcmo_cache_dir = base_dir / 'analysis_output' / 'kcmo_cache'
        try:
            self._osm_cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            self._kcmo_cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Socrata token
        try:
            self._kcmo_headers = {}
            token = AppConfig().KCMO_APP_TOKEN
            if token:
                self._kcmo_headers['X-App-Token'] = token
        except Exception:
            self._kcmo_headers = {}
    
    def normalize_county_name(self, county: str, state: str) -> str:
        """Normalize county name to consistent format
        
        Args:
            county: County name (may or may not include 'County')
            state: State abbreviation
            
        Returns:
            Normalized county name in format "County Name County, STATE"
        """
        if not county:
            return ""
        # Clean inputs
        county_raw = str(county).strip()
        state_in = (str(state).strip().upper() if state else '').upper()

        if not county_raw or county_raw.lower() in ['none', 'null', '']:
            return ""

        # If county string already includes a state (e.g., "Jackson County, MO"), split it
        county_core = county_raw
        state_from_county = ''
        if ',' in county_core:
            parts = [p.strip() for p in county_core.split(',') if p.strip()]
            if parts:
                county_core = parts[0]
                if len(parts) > 1:
                    state_from_county = parts[1].upper()

        # Prefer explicit state argument, else use the parsed one, fallback to MO
        state_norm = state_in or state_from_county or 'MO'

        # Normalize whitespace
        county_core = ' '.join(county_core.split())

        # Remove accidental duplication like "Jackson County County"
        if county_core.endswith('County County'):
            county_core = county_core[:-len(' County')]

        # Ensure single "County" suffix
        if not county_core.endswith('County'):
            county_core = county_core.rstrip(',')
            county_core = f"{county_core} County"

        if county_core == 'County' or county_core == ' County':
            logger.debug(f"Invalid county name after normalization: '{county_raw}'")
            return ""

        return f"{county_core}, {state_norm}"

    
    def validate_coordinates(self, lat: float, lon: float) -> bool:
        """Validate if coordinates are within Kansas City metro area
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            True if coordinates are valid and within KC area
        """
        try:
            # Convert to float if string
            lat = float(lat)
            lon = float(lon)
            
            # Check if coordinates are valid numbers
            if math.isnan(lat) or math.isnan(lon):
                return False
                
            # Kansas City metro area approximate bounds
            # North: 39.5 (includes Platte County)
            # South: 38.2 (includes Linn County)  
            # East: -94.0 (includes Ray County)
            # West: -95.2 (includes Leavenworth County)
            
            if not (38.2 <= lat <= 39.7 and -95.2 <= lon <= -94.0):
                # Too chatty as WARNING during boundary generation; reduce to debug
                logger.debug(f"Coordinates outside KC metro area: {lat}, {lon}")
                return False
                
            return True
            
        except (ValueError, TypeError):
            logger.warning(f"Invalid coordinate values: lat={lat}, lon={lon}")
            return False

    def _get_business_coordinates(self, business: Dict) -> Optional[Tuple[float, float]]:
        """Return usable coordinates for a business, falling back to county centroids."""
        lat = business.get('lat') or business.get('latitude')
        lon = business.get('lon') or business.get('longitude')

        # Use enhanced geographic validator if available
        if self.geographic_validator and lat is not None and lon is not None:
            try:
                lat_f = float(lat)
                lon_f = float(lon)
                
                # Use enhanced validation with correction
                validation = self.geographic_validator.validate_coordinates(lat_f, lon_f)
                if validation['valid']:
                    business['latitude'] = lat_f
                    business['longitude'] = lon_f
                    return lat_f, lon_f
                elif validation['corrected']:
                    # Apply corrected coordinates
                    lat_f, lon_f = validation['corrected']
                    business['latitude'] = lat_f
                    business['longitude'] = lon_f
                    logger.debug(f"Applied coordinate correction for {business.get('name')}: {lat}, {lon} -> {lat_f}, {lon_f}")
                    return lat_f, lon_f
                else:
                    logger.debug(f"Invalid coordinates for {business.get('name')}: {lat}, {lon} - {validation['issues']}")
            except (ValueError, TypeError):
                logger.debug(f"Could not parse coordinates for business {business.get('name')}: {lat}, {lon}")
        elif lat is not None and lon is not None:
            # Fallback to legacy validation
            try:
                lat_f = float(lat)
                lon_f = float(lon)
                if self.validate_coordinates(lat_f, lon_f):
                    business['latitude'] = lat_f
                    business['longitude'] = lon_f
                    return lat_f, lon_f
            except (ValueError, TypeError):
                logger.debug(f"Could not parse coordinates for business {business.get('name')}: {lat}, {lon}")

        # County fallback
        county = business.get('county')
        if county:
            state = business.get('state', 'MO')
            county_key = self.normalize_county_name(county, state)
            if county_key and county_key in self.county_coords:
                lat_f, lon_f = self.county_coords[county_key]

                # If we have polygon data, sample a deterministic point inside it
                if SHAPELY_AVAILABLE and county_key in KC_COUNTY_BOUNDARIES:
                    polygon = shape(KC_COUNTY_BOUNDARIES[county_key])
                    seed_val = f"{business.get('name','')}-{business.get('address','')}-{county_key}"
                    seed = int(hashlib.sha256(seed_val.encode('utf-8')).hexdigest(), 16) % (2 ** 32)
                    rng = random.Random(seed)
                    minx, miny, maxx, maxy = polygon.bounds

                    for _ in range(50):
                        trial_lon = rng.uniform(minx, maxx)
                        trial_lat = rng.uniform(miny, maxy)
                        if self.validate_coordinates(trial_lat, trial_lon) and polygon.contains(Point(trial_lon, trial_lat)):
                            lat_f, lon_f = trial_lat, trial_lon
                            break
                else:
                    # Apply a small deterministic jitter around the centroid
                    seed_val = f"{business.get('name','')}-{business.get('address','')}-{county_key}"
                    seed = int(hashlib.sha256(seed_val.encode('utf-8')).hexdigest(), 16) % (2 ** 32)
                    rng = random.Random(seed)
                    lat_f += (rng.random() - 0.5) * 0.02
                    lon_f += (rng.random() - 0.5) * 0.02 / max(math.cos(math.radians(lat_f)), 0.1)

                business['latitude'] = lat_f
                business['longitude'] = lon_f
                return lat_f, lon_f
            else:
                logger.debug(f"No county centroid found for {county} ({state})")

        return None
    
    def create_cluster_map(self, results: Dict) -> str:
        """Create an interactive map showing clusters and businesses"""
        try:
            logger.info("Creating cluster map")
            
            # Initialize map with explicit dimensions
            m = folium.Map(
                location=self.kc_center,
                zoom_start=10,  # original focus
                tiles='OpenStreetMap',
                width='100%',
                height='100%',
                min_zoom=8,
                max_zoom=15,
                prefer_canvas=True
            )
            # Add fullscreen control for better viewing
            try:
                plugins.Fullscreen(position='topright', title='Full Screen', title_cancel='Exit Full Screen').add_to(m)
            except Exception:
                pass
            
            # Interactive features disabled for cleaner map - use Folium's built-in layer controls instead
            # if self.interactive_features:
            #     self.interactive_features.add_interactive_features(m, results.get('clusters', []))
            
            # Add layers in specific order (ensure county boundaries can be toggled visibly)
            # 1. Add cluster visualizations with unified boundaries first
            #    Also add an additional consolidated layer that merges overlapping/duplicate clusters
            self._add_improved_cluster_visualization(m, results)
            try:
                self._add_consolidated_clusters_layer(m, results)
            except Exception as e:
                logger.debug(f"Consolidated clusters layer skipped: {e}")
            
            # 3. Add infrastructure markers (optional) - only if infrastructure data exists
            if self._has_infrastructure_data(results):
                infra_group = folium.FeatureGroup(name="ðŸš‚ Infrastructure", show=False)
                self._add_infrastructure_markers_to_group(infra_group, results)
                infra_group.add_to(m)

            # 3b. Add rich OSM-based infrastructure overlays (toggleable)
            try:
                self._add_osm_infrastructure_layers(m)
            except Exception as e:
                logger.debug(f"OSM infrastructure overlays skipped: {e}")

            # 3c. Add KCMO authoritative layers (Hospitals/Clinics, Schools), if available
            try:
                self._add_kcmo_open_data_layers(m)
            except Exception as e:
                logger.debug(f"KCMO open data layers skipped: {e}")
            
            # 4. Add geopolitical risk overlay if available (optional)
            market_data = results.get('market_data', {})
            if 'geopolitical_risks' in market_data:
                risk_group = folium.FeatureGroup(name="âš ï¸ Risk Areas", show=False)
                self._add_geopolitical_risk_to_group(risk_group, results)
                risk_group.add_to(m)
            
            # 2. Add county boundaries last so they sit above other polygons (toggling is visible)
            self._add_subtle_county_boundaries(m)

            # Add improved legend
            self._add_improved_legend(m, results)
            
            # Add layer control for feature groups only (no base maps)
            # This allows toggling visibility of different data layers
            layer_control = folium.LayerControl(
                collapsed=True,
                position='topleft',
                autoZIndex=False
            )
            
            # Only add the layer control if there are actual feature groups to control
            # The base map switching is disabled to prevent conflicts
            layer_control.add_to(m)

            # Return HTML
            return m._repr_html_()
            
        except Exception as e:
            logger.error(f"Error creating cluster map: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return a simple error map
            error_map = folium.Map(location=self.kc_center, zoom_start=9)
            folium.Marker(
                self.kc_center,
                popup=f"Error loading map: {str(e)}",
                icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa')
            ).add_to(error_map)
            return error_map._repr_html_()
    
    def _add_subtle_county_boundaries(self, m: folium.Map):
        """Add subtle county boundaries as background context"""
        # Create a single group for all counties, hidden by default
        counties_group = folium.FeatureGroup(name="County Boundaries", overlay=True, control=True, show=False)
        
        urban_count = 0
        suburban_count = 0
        rural_count = 0
        
        # Add county polygons
        for county_name, geojson in KC_COUNTY_BOUNDARIES.items():
            
            # Create GeoJSON feature
            feature = {
                "type": "Feature",
                "properties": {"name": county_name},
                "geometry": geojson
            }
            
            # Add county polygon with subtle styling
            folium.GeoJson(
                feature,
                name=county_name,
                style_function=lambda x: {
                    'fillColor': 'transparent',
                    'color': '#666666',   # Darker gray for visibility
                    'weight': 2,          # Thicker border
                    'dashArray': '4,4',   # Subtle dashed line
                    'fillOpacity': 0,
                    'opacity': 0.9
                },
                highlight_function=None,  # No highlight to reduce clutter
                tooltip=folium.Tooltip(county_name, sticky=True)
            ).add_to(counties_group)
        
        # Add single counties group to map
        counties_group.add_to(m)
        
        logger.info(f"Added subtle county boundaries")
    
    def _add_improved_cluster_visualization(self, m: folium.Map, results: Dict):
        """Add improved cluster visualization with unified boundaries"""
        # Log the structure to debug
        logger.info(f"Results keys: {list(results.keys())}")
        if 'steps' in results:
            logger.info(f"Steps keys: {list(results.get('steps', {}).keys())}")
            if 'cluster_optimization' in results.get('steps', {}):
                logger.info(f"Cluster optimization keys: {list(results['steps']['cluster_optimization'].keys())}")
        
        # Try multiple possible paths for clusters
        clusters = results.get('steps', {}).get('cluster_optimization', {}).get('clusters', [])
        if not clusters:
            # Try direct clusters key
            clusters = results.get('clusters', [])
        if not clusters:
            # Try in economic_impact
            clusters = results.get('economic_impact', {}).get('clusters', [])
        
        logger.info(f"Creating improved visualization for {len(clusters)} clusters")
        
        if len(clusters) == 0:
            logger.warning("No clusters found in results - map will only show base layers")
            # Add a message to the map
            message_html = '''
            <div style='position: fixed; 
                        top: 60px; left: 50%; transform: translateX(-50%);
                        padding: 15px 30px;
                        background-color: #fff3cd; border: 2px solid #ffeeba;
                        border-radius: 5px; z-index: 9999;
                        font-family: Arial, sans-serif;'>
                <strong>âš ï¸ No Cluster Data Available</strong><br>
                <span style='font-size: 14px;'>Please run an analysis first to see clusters on the map.</span>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(message_html))
        
        # Create cleaner layer groups with better defaults
        cluster_boundaries_group = folium.FeatureGroup(name="Cluster Regions", show=True)
        cluster_centers_group = folium.FeatureGroup(name="Cluster Centers", show=True)
        business_markers_group = folium.FeatureGroup(name="Key Businesses", show=False)  # Hidden by default
        all_businesses_group = folium.FeatureGroup(name="All Businesses", show=False)
        
        # Create type-specific layers but keep them hidden by default to reduce clutter
        type_layers: Dict[str, folium.FeatureGroup] = {}
        type_key_layers: Dict[str, folium.FeatureGroup] = {}
        def _get_type_layer(cluster_type: str) -> folium.FeatureGroup:
            key = (cluster_type or 'mixed')
            if key not in type_layers:
                display = key.replace('_', ' ').title()
                type_layers[key] = folium.FeatureGroup(name=f"Type: {display}", show=False)
            return type_layers[key]
        def _get_type_key_layer(cluster_type: str) -> folium.FeatureGroup:
            key = (cluster_type or 'mixed')
            if key not in type_key_layers:
                display = key.replace('_', ' ').title()
                type_key_layers[key] = folium.FeatureGroup(name=f"Type: {display} (Key Businesses)", show=False)
            return type_key_layers[key]
        
        # Process each cluster
        for i, cluster in enumerate(clusters):
            cluster_name = cluster.get('name', f'Cluster {i+1}')
            cluster_type = cluster.get('type', 'mixed')
            color = self.enhanced_cluster_colors.get(cluster_type, self.enhanced_cluster_colors.get('mixed', '#757575'))
            businesses = cluster.get('businesses', [])
            
            if not businesses:
                continue
            
            # 1. Add unified cluster boundary (overall + type layer)
            self._add_unified_cluster_boundary(cluster_boundaries_group, cluster, i, color)
            type_layer = _get_type_layer(cluster_type)
            self._add_unified_cluster_boundary(type_layer, cluster, i, color)
            
            # 2. Add cluster center marker (overall + type layer)
            self._add_cluster_center_marker(cluster_centers_group, cluster, i, color)
            self._add_cluster_center_marker(type_layer, cluster, i, color)
            
            # 3. Add top businesses (filtered) overall + per type
            self._add_top_businesses_only(business_markers_group, cluster, color)
            type_key_layer = _get_type_key_layer(cluster_type)
            self._add_top_businesses_only(type_key_layer, cluster, color)
            self._add_all_businesses(all_businesses_group, cluster, color)
        
        # Optional Heatmap of all businesses - weight by growth potential (business score × cluster ROI)
        try:
            # Create a clustered view of businesses instead of heatmap for better performance
            business_cluster = plugins.MarkerCluster(name="All Businesses (Clustered)", show=False)
            
            business_count = 0
            for c in clusters:
                for b in c.get('businesses', [])[:200]:  # Reduced cap per cluster
                    if business_count >= 1000:  # Total cap across all clusters
                        break
                        
                    coord = self._get_business_coordinates(b)
                    if coord:
                        # Create simple popup
                        popup_html = f"""
                        <div style='width: 180px; font-family: Arial, sans-serif;'>
                            <h6 style='margin: 0 0 3px 0;'>{b.get('name', 'Business')}</h6>
                            <div style='font-size: 10px; color: #666;'>
                                <b>Cluster:</b> {c.get('name', 'Unknown')}<br>
                                <b>Score:</b> {b.get('composite_score', 0):.1f}
                            </div>
                        </div>
                        """
                        
                        folium.Marker(
                            location=[coord[0], coord[1]],
                            popup=folium.Popup(popup_html, max_width=200),
                            tooltip=f"{b.get('name', 'Business')} - {b.get('composite_score', 0):.0f}",
                            icon=folium.Icon(color='gray', icon='circle', icon_size='tiny')
                        ).add_to(business_cluster)
                        business_count += 1
                        
                if business_count >= 1000:
                    break
            
            # Add the clustered business markers to map
            business_cluster.add_to(m)

            # Also add a lightweight growth heatmap with fewer points for performance
            # Weight definition: normalized business composite_score (0..1) × adjusted cluster ROI (0.5..1.5)
            heat_points = []
            for c in clusters:
                # Determine cluster ROI (as fraction). Fallback to ml_predictions if needed.
                roi_frac = 0.0
                try:
                    roi_frac = float(c.get('roi', 0.0))
                    if roi_frac == 0.0 and 'ml_predictions' in c:
                        mp = c.get('ml_predictions') or {}
                        if 'expected_roi' in mp:
                            roi_frac = float(mp.get('expected_roi') or 0.0)
                        elif 'roi_percentage' in mp:
                            roi_frac = float(mp.get('roi_percentage') or 0.0) / 100.0
                except Exception:
                    roi_frac = 0.0
                # Clamp to reasonable band to avoid zeroing
                roi_adjust = max(0.5, min(1.5, 0.5 + roi_frac))

                for b in c.get('businesses', [])[:100]:  # Further reduced for heatmap
                    coord = self._get_business_coordinates(b)
                    if not coord:
                        continue
                    score = 0.0
                    try:
                        raw_score = float(b.get('composite_score', b.get('score', 0.0)) or 0.0)
                        # Composite score commonly 0..100; normalize
                        score = max(0.0, min(1.0, raw_score / 100.0))
                    except Exception:
                        score = 0.0
                    weight = max(0.05, min(1.0, score * roi_adjust))
                    heat_points.append([coord[0], coord[1], weight])
            if heat_points:
                heat = plugins.HeatMap(heat_points, name="Growth Potential (Heatmap)", blur=25, radius=20, min_opacity=0.05, show=False)
                heat.add_to(m)
        except Exception as e:
            logger.debug(f"Could not create business clusters/heatmap: {e}")
            pass

        # Add groups to map
        cluster_boundaries_group.add_to(m)
        cluster_centers_group.add_to(m)
        business_markers_group.add_to(m)
        all_businesses_group.add_to(m)
        # Add type-specific layers
        for tl in type_layers.values():
            tl.add_to(m)
        for tl in type_key_layers.values():
            tl.add_to(m)
        
        logger.info("Improved cluster visualization added")
    
    def _add_unified_cluster_boundary(self, layer: folium.FeatureGroup, cluster: Dict, index: int, color: str):
        """Add cleaner boundaries with reduced visual weight."""
        businesses = cluster.get('businesses', [])
        logger.info(f"Adding boundary for cluster {index+1} with {len(businesses)} businesses")

        if not businesses:
            self._add_cluster_by_center(layer, cluster, index, color)
            return

        # Use enhanced color scheme if available
        if self.enhanced_color_scheme:
            cluster_type = cluster.get('type', 'mixed')
            color = self.enhanced_color_scheme.get_color_for_cluster(cluster_type)

        # Skip spatial accuracy enhancer for performance reasons
        # The enhancer is being called too many times and slowing down map generation
        # Use standard boundary creation instead
        pass

        # Split into geographic sub-clusters to reduce over-coverage
        # Increase subcluster distance to group nearby businesses more broadly
        subclusters = self._create_subclusters(businesses, max_distance=0.06)
        any_drawn = False
        for sub in subclusters:
            if len(sub) < 5:  # Increased minimum from 3 to 5
                continue
            # Expand buffer to enlarge visual footprint of cluster polygons
            boundary = self._create_cluster_boundaries(sub, buffer=0.01, clip_to_kc=True)
            if not boundary:
                continue
            any_drawn = True
            # Confidence-aware styling
            conf = float(cluster.get('confidence_score', 0.6) or 0.6)
            conf = max(0.0, min(1.0, conf))
            # Opacity scales with confidence
            fill_opacity = 0.1 + 0.5 * conf
            # Border weight/opacity tuned by confidence
            border_weight = 1 + int(2 * conf)
            border_opacity = 0.6 + 0.3 * conf

            folium.Polygon(
                locations=boundary,
                color=color,
                weight=border_weight,
                opacity=border_opacity,
                fill=True,
                fill_color=color,
                fillOpacity=fill_opacity,
                popup=self._create_detailed_cluster_popup(cluster, sub),
                tooltip=f"{cluster.get('name', f'Cluster {index+1}')} - {len(sub)} businesses"
            ).add_to(layer)

        if not any_drawn:
            logger.info("No subcluster boundaries created, using center circle fallback")
            self._add_cluster_by_center(layer, cluster, index, color)
    
    def _add_cluster_by_center(self, layer: folium.FeatureGroup, cluster: Dict, index: int, color: str):
        """Add cluster as a circle at its geographic center"""
        businesses = cluster.get('businesses', [])
        
        # Calculate center from businesses or use county centers
        center = self._calculate_cluster_center(cluster)
        if not center:
            # Use first business's county as fallback
            for b in businesses:
                county = b.get('county', 'Unknown')
                state = b.get('state', 'MO')
                
                # Use normalized county name
                county_key = self.normalize_county_name(county, state)
                    
                if county_key and county_key in self.county_coords:
                    center = self.county_coords[county_key]
                    break
        
        if center:
            logger.info(f"Adding circle at center {center}")
            # Size based on impact and business count
            gdp_impact = cluster.get('projected_gdp_impact', 0)
            # Scale radius more generously; cap higher for visibility
            radius = 7000 + (len(businesses) * 150) + (gdp_impact / 1e6 * 15)
            radius = min(radius, 30000)
            
            folium.Circle(
                location=center,
                radius=radius,
                color=color,
                weight=3,
                opacity=0.8,
                fill=True,
                fill_color=color,
                fillOpacity=0.15,
                popup=self._create_detailed_cluster_popup(cluster),
                tooltip=f"{cluster.get('name', f'Cluster {index+1}')} - {len(businesses)} businesses"
            ).add_to(layer)
        else:
            logger.warning(f"No center found for cluster {index+1}")
    
    def _calculate_cluster_center(self, cluster: Dict) -> Optional[Tuple[float, float]]:
        """Calculate the geographic center of a cluster"""
        businesses = cluster.get('businesses', [])
        coords = []
        
        for business in businesses:
            coord = self._get_business_coordinates(business)
            if coord:
                coords.append(list(coord))
        
        if coords:
            coords_array = np.array(coords)
            return tuple(np.mean(coords_array, axis=0))
        
        return None
    
    def _add_cluster_center_marker(self, layer: folium.FeatureGroup, cluster: Dict, index: int, color: str):
        """Add a cleaner, smaller marker for the cluster center"""
        center = self._calculate_cluster_center(cluster)
        if not center:
            return
        
        gdp_impact = cluster.get('projected_gdp_impact', 0)
        businesses = cluster.get('businesses', [])
        
        # Create smaller, cleaner custom icon
        icon_html = f"""
        <div style="
            background-color: {color};
            color: white;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 14px;
            border: 2px solid white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        ">
            {index+1}
        </div>
        """
        
        folium.Marker(
            location=center,
            icon=folium.DivIcon(html=icon_html, icon_size=(30, 30), icon_anchor=(15, 15)),
            popup=self._create_detailed_cluster_popup(cluster),
            tooltip=f"{cluster.get('name')} | ${gdp_impact/1e6:.1f}M GDP | {len(businesses)} businesses"
        ).add_to(layer)
    
    def _add_top_businesses_only(self, layer: folium.FeatureGroup, cluster: Dict, color: str):
        """Add markers only for the most important businesses - reduced for cleaner map"""
        businesses = cluster.get('businesses', [])
        
        # Track locations to detect stacking
        location_counts = {}
        
        # Score and sort businesses
        scored_businesses = []
        for business in businesses:
            coord = self._get_business_coordinates(business)
            if not coord:
                continue

            lat, lon = coord
            loc_key = f"{lat:.4f},{lon:.4f}"
            location_counts[loc_key] = location_counts.get(loc_key, 0) + 1

            score = business.get('composite_score', business.get('score', 0))
            employees = business.get('employees', 1)
            importance = score * np.log10(max(employees, 1))

            scored_businesses.append({
                'business': business,
                'importance': importance,
                'lat': lat,
                'lon': lon,
                'loc_key': loc_key
            })
        
        # Sort by importance and take only top 5 per cluster (reduced from 10)
        scored_businesses.sort(key=lambda x: x['importance'], reverse=True)
        
        # Track which businesses we've placed at each location for jittering
        location_indices = {}
        
        for i, item in enumerate(scored_businesses[:5]):  # Reduced from 10 to 5
            business = item['business']
            loc_key = item['loc_key']
            
            # Apply jittering if multiple businesses at same location
            lat, lon = item['lat'], item['lon']
            if location_counts[loc_key] > 1:
                # Get index of this business at this location
                if loc_key not in location_indices:
                    location_indices[loc_key] = 0
                else:
                    location_indices[loc_key] += 1
                
                # Apply circular jitter pattern
                idx = location_indices[loc_key]
                total = location_counts[loc_key]
                angle = (2 * math.pi * idx) / total
                radius = 0.003  # Increased offset to 300m for better separation
                
                lat += radius * math.cos(angle)
                lon += radius * math.sin(angle)
            
            # Simplified styling - only top 3 get special treatment
            if i < 3:
                marker_size = 8  # Reduced from 10
                marker_opacity = 0.8  # Reduced from 0.9
            else:
                marker_size = 5  # Reduced from 6
                marker_opacity = 0.6  # Reduced from 0.7
            
            folium.CircleMarker(
                location=[lat, lon],  # Use jittered coordinates
                radius=marker_size,
                popup=self._create_business_popup(business, cluster),
                tooltip=f"{business.get('name', 'Business')} - Score: {business.get('composite_score', 0):.0f}",
                color=color,
                fill=True,
                fill_color=color,
                fillOpacity=marker_opacity,
                weight=1  # Reduced from 2
            ).add_to(layer)
    
    
    def _add_all_businesses(self, layer: folium.FeatureGroup, cluster: Dict, color: str):
        """Add business points using MarkerCluster to reduce visual clutter."""
        businesses = cluster.get('businesses', [])
        if not businesses:
            return
            
        # Create a marker cluster for this cluster's businesses
        marker_cluster = plugins.MarkerCluster(name=f"Businesses - {cluster.get('name', 'Cluster')}")
        
        # Collect valid coordinates
        valid_businesses = []
        for business in businesses:
            coord = self._get_business_coordinates(business)
            if coord:
                valid_businesses.append((business, coord))
        
        if not valid_businesses:
            return
            
        # Sample businesses to reduce markers (show every 4th business)
        sampled_businesses = valid_businesses[::4]
        
        # Add markers to cluster
        for business, (lat, lon) in sampled_businesses[:500]:  # Cap at 500 markers per cluster
            # Create a simple popup for each business
            popup_html = f"""
            <div style='width: 200px; font-family: Arial, sans-serif;'>
                <h6 style='margin: 0 0 5px 0;'>{business.get('name', 'Business')}</h6>
                <div style='font-size: 11px; color: #666;'>
                    <b>Employees:</b> {business.get('employees', 0)}<br>
                    <b>Score:</b> {business.get('composite_score', 0):.1f}<br>
                    <b>Revenue:</b> ${business.get('revenue_estimate', 0):,.0f}
                </div>
            </div>
            """
            
            # Add a simple marker to the cluster
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{business.get('name', 'Business')} - Score: {business.get('composite_score', 0):.0f}",
                icon=folium.Icon(color='lightblue', icon='building', prefix='fa', icon_size='small')
            ).add_to(marker_cluster)
        
        # Add the marker cluster to the layer
        marker_cluster.add_to(layer)

    def _has_infrastructure_data(self, results: Dict) -> bool:
        """Check if infrastructure data exists in results"""
        # Check if there's any infrastructure data in the results
        if 'infrastructure' in results:
            return bool(results['infrastructure'])
        # For now, return False to hide empty infrastructure layer
        return False
    
    def _add_infrastructure_markers_to_group(self, layer: folium.FeatureGroup, results: Dict):
        """Add infrastructure markers to a specific layer group"""
        # Delegate to existing method but add to specific layer
        self._add_infrastructure_markers(layer, results)
    
    def _add_geopolitical_risk_to_group(self, layer: folium.FeatureGroup, results: Dict):
        """Add geopolitical risk overlay to a specific layer group"""
        # This would need to be implemented based on your risk data structure
        pass
    
    def _add_improved_legend(self, m: folium.Map, results: Dict):
        """Add a collapsible mini-legend that doesn't obstruct the map"""
        clusters = results.get('steps', {}).get('cluster_optimization', {}).get('clusters', [])
        
        # Get unique cluster types
        cluster_types = {}
        for cluster in clusters:
            ctype = cluster.get('type', 'mixed')
            if ctype not in cluster_types:
                cluster_types[ctype] = 0
            cluster_types[ctype] += 1
        
        # Create a collapsible mini-legend to reduce screen space
        legend_html = '''
        <div id="miniLegend" style="
            position: fixed;
            bottom: 10px;
            right: 10px;
            z-index: 1000;
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 5px;
            font-family: Arial, sans-serif;
            font-size: 11px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
            max-width: 200px;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; cursor: pointer;" onclick="toggleLegendContent()">
                <span style="font-weight: bold; color: #333;">📊 Clusters</span>
                <span id="legendToggleIcon">▼</span>
            </div>
            <div id="legendContent" style="display: none; margin-top: 5px;">
        '''
        
        # Add cluster types - only show top 4 to save space
        sorted_types = sorted(cluster_types.items(), key=lambda x: x[1], reverse=True)[:4]
        for ctype, count in sorted_types:
            color = self.enhanced_cluster_colors.get(ctype, self.enhanced_cluster_colors.get('mixed', '#757575'))
            legend_html += f'''
            <div style='margin: 2px 0; display: flex; align-items: center;'>
                <div style='width: 12px; height: 12px; background-color: {color};
                            border: 1px solid white; border-radius: 50%;
                            margin-right: 5px;'></div>
                <span style='font-size: 10px; color: #555;'>{ctype.replace("_", " ").title()} ({count})</span>
            </div>
            '''
        
        # Add compact summary stats
        total_businesses = sum(len(c.get('businesses', [])) for c in clusters)
        total_gdp = sum(c.get('projected_gdp_impact', 0) for c in clusters)
        
        legend_html += f'''
                <hr style='margin: 5px 0; border: none; border-top: 1px solid #ddd;'>
                <div style='font-size: 9px; color: #666; text-align: center;'>
                    <div><strong>{len(clusters)}</strong> clusters</div>
                    <div><strong>{total_businesses:,}</strong> businesses</div>
                    <div><strong>${total_gdp/1e9:.1f}B</strong> GDP</div>
                </div>
            </div>
        </div>
        
        <script>
        function toggleLegendContent() {{
            var content = document.getElementById('legendContent');
            var icon = document.getElementById('legendToggleIcon');
            if (content.style.display === 'none') {{
                content.style.display = 'block';
                icon.textContent = '▲';
            }} else {{
                content.style.display = 'none';
                icon.textContent = '▼';
            }}
        }}
        </script>
        '''
        
        m.get_root().html.add_child(folium.Element(legend_html))

    # ---------------- Consolidation logic (display-only) ----------------
    def _business_key(self, b: Dict) -> str:
        name = str(b.get('name', '')).upper().strip()
        addr = str(b.get('address', '')).upper().strip()
        zip5 = str(b.get('zip', '')).strip()[:5]
        return f"{name}|{addr}|{zip5}".strip('|')

    def _cluster_signature(self, cluster: Dict) -> Dict:
        """Return helpers for comparing clusters: business keys set and centroid."""
        keys = set()
        coords = []
        for b in cluster.get('businesses', []) or []:
            try:
                keys.add(self._business_key(b))
                coord = self._get_business_coordinates(b)
                if coord:
                    coords.append(coord)
            except Exception:
                continue
        # centroid
        lat = sum(c[0] for c in coords)/len(coords) if coords else self.kc_center[0]
        lon = sum(c[1] for c in coords)/len(coords) if coords else self.kc_center[1]
        return {'keys': keys, 'centroid': (lat, lon)}

    def _km(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        R = 6371.0
        from math import radians, sin, cos, asin, sqrt
        lat1, lon1 = a; lat2, lon2 = b
        dlat = radians(lat2-lat1); dlon = radians(lon2-lon1)
        p1 = radians(lat1); p2 = radians(lat2)
        h = sin(dlat/2)**2 + cos(p1)*cos(p2)*sin(dlon/2)**2
        return 2*R*asin(sqrt(h))

    def _consolidate_clusters(self, clusters: List[Dict]) -> List[Dict]:
        """Group highly overlapping clusters (same type) and merge for display.

        Criteria to merge A,B:
        - Same cluster 'type' AND
        - Jaccard(business_keys) >= 0.5 OR
          centroid distance < 8 km with Jaccard >= 0.3
        """
        if not clusters:
            return []
        sigs = [self._cluster_signature(c) for c in clusters]
        n = len(clusters)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a,b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            for j in range(i+1, n):
                if (clusters[i].get('type') or 'mixed') != (clusters[j].get('type') or 'mixed'):
                    continue
                A, B = sigs[i], sigs[j]
                if not A['keys'] or not B['keys']:
                    continue
                inter = len(A['keys'] & B['keys'])
                union_size = len(A['keys'] | B['keys']) or 1
                jacc = inter / union_size
                dist = self._km(A['centroid'], B['centroid'])
                if jacc >= 0.5 or (dist < 8 and jacc >= 0.3):
                    union(i, j)

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            r = find(i)
            groups.setdefault(r, []).append(i)

        consolidated: List[Dict] = []
        for root, idxs in groups.items():
            if len(idxs) == 1:
                consolidated.append(clusters[idxs[0]])
                continue
            # Merge
            merged = {}
            members = [clusters[k] for k in idxs]
            merged['type'] = members[0].get('type', 'mixed')
            merged['name'] = ' + '.join({m.get('name', f'Cluster {i+1}') for m in members})
            # Businesses: union by key
            seen = set()
            merged_businesses = []
            for m in members:
                for b in m.get('businesses', []) or []:
                    k = self._business_key(b)
                    if not k or k in seen:
                        continue
                    seen.add(k)
                    merged_businesses.append(b)
            merged['businesses'] = merged_businesses
            # Metrics: recompute light aggregates
            try:
                merged['business_count'] = len(merged_businesses)
                # Use max of strategic/market where present to avoid diluting
                merged['strategic_score'] = max(m.get('strategic_score', 0) for m in members)
                # Combine ML predictions conservatively: take max GDP/jobs/ROI
                preds = [m.get('ml_predictions', {}) for m in members]
                merged['ml_predictions'] = {
                    'gdp_impact': max((p.get('gdp_impact') or 0) for p in preds),
                    'job_creation': max((p.get('job_creation') or 0) for p in preds),
                    'roi_percentage': max((p.get('roi_percentage') or 0) for p in preds),
                    'confidence': min((p.get('confidence') or 0.0) for p in preds) if preds else 0.0
                }
            except Exception:
                pass
            consolidated.append(merged)

        return consolidated

    def _add_consolidated_clusters_layer(self, m: folium.Map, results: Dict):
        """Add a layer with merged cluster polygons for clearer regional view."""
        clusters = results.get('steps', {}).get('cluster_optimization', {}).get('clusters', [])
        if not clusters:
            return
        merged = self._consolidate_clusters(clusters)
        if len(merged) <= len(clusters):
            layer = folium.FeatureGroup(name=f"Consolidated Regions ({len(merged)})", show=False)
            # Draw polygons for merged clusters
            for i, c in enumerate(merged):
                color = self.enhanced_cluster_colors.get(c.get('type', 'mixed'), '#757575')
                boundary = self._create_cluster_boundaries(c.get('businesses', []), buffer=0.004, clip_to_kc=True)
                if not boundary:
                    # fallback to center circle
                    center = self.kc_center
                    try:
                        # try compute approximate center from businesses
                        coords = [self._get_business_coordinates(b) for b in c.get('businesses', [])]
                        coords = [xy for xy in coords if xy]
                        if coords:
                            center = (sum(x for x,_ in coords)/len(coords), sum(y for _,y in coords)/len(coords))
                    except Exception:
                        pass
                    folium.Circle(location=center, radius=9000, color=color, fill=True, fill_color=color, fillOpacity=0.2, weight=2,
                                  tooltip=f"{c.get('name','Cluster')} (Consolidated)").add_to(layer)
                else:
                    folium.Polygon(locations=boundary, color=color, weight=2, opacity=0.8, fill=True,
                                   fill_color=color, fillOpacity=0.25,
                                   tooltip=f"{c.get('name','Cluster')} (Consolidated)").add_to(layer)
            layer.add_to(m)

    def _kc_bounds(self):
        """Compute KC metro bounding box as (south, west, north, east)."""
        try:
            kc = self._kc_union_polygon()
            if kc is not None:
                minx, miny, maxx, maxy = kc.bounds  # lon/lat
                return (miny, minx, maxy, maxx)
        except Exception:
            pass
        # Fallback approximate bounds
        return (38.2, -95.2, 39.7, -94.0)

    def _overpass_fetch(self, cache_key: str, query: str, ttl_hours: int = 24) -> Optional[dict]:
        """Fetch Overpass API result with simple file cache."""
        cache_path = self._osm_cache_dir / f"{cache_key}.json"
        try:
            if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < ttl_hours * 3600:
                return json.loads(cache_path.read_text(encoding='utf-8'))
        except Exception:
            pass

        urls = [
            "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"
        ]
        last_err = None
        for u in urls:
            try:
                r = requests.post(u, data={"data": query}, timeout=30)
                r.raise_for_status()
                data = r.json()
                try:
                    cache_path.write_text(json.dumps(data), encoding='utf-8')
                except Exception:
                    pass
                return data
            except Exception as e:
                last_err = e
                continue
        if last_err:
            logger.debug(f"Overpass fetch failed for {cache_key}: {last_err}")
        return None

    def _socrata_fetch(self, resource_id: str, params: Dict[str, str], cache_key: str, ttl_hours: int = 24) -> Optional[list]:
        """Fetch a Socrata dataset (records list) with file cache.
        resource_id: like 'abcd-1234'
        params: SoQL params (e.g., $select, $limit)
        """
        if not resource_id:
            return None
        cache_path = self._kcmo_cache_dir / f"{cache_key}.json"
        try:
            if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < ttl_hours * 3600:
                return json.loads(cache_path.read_text(encoding='utf-8'))
        except Exception:
            pass
        url = f"https://data.kcmo.org/resource/{resource_id}.json"
        # Default limit to 20k
        p = {'$limit': '20000'}
        p.update(params or {})
        last_err = None
        try:
            r = requests.get(url, params=p, headers=self._kcmo_headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict) and 'data' in data:
                # Some endpoints might return wrapped structure; unwrap
                data = data['data']
            try:
                cache_path.write_text(json.dumps(data), encoding='utf-8')
            except Exception:
                pass
            return data if isinstance(data, list) else None
        except Exception as e:
            last_err = e
            logger.debug(f"Socrata fetch failed for {resource_id}: {e}")
        return None

    def _parse_socrata_point(self, rec: dict) -> Optional[Tuple[float, float]]:
        """Parse lat/lon from typical Socrata geospatial fields."""
        if not isinstance(rec, dict):
            return None
        # Direct lat/lon keys
        for la, lo in ((rec.get('latitude'), rec.get('longitude')),
                       (rec.get('lat'), rec.get('lon')),
                       (rec.get('y'), rec.get('x'))):
            try:
                if la is not None and lo is not None:
                    return float(la), float(lo)
            except Exception:
                pass
        # Object geocoded field variants
        for key in ('location', 'location_1', 'geocoded_column', 'geo_location', 'point', 'the_geom'):
            v = rec.get(key)
            if isinstance(v, dict):
                # Likely {'latitude': '...', 'longitude': '...'} or {'coordinates': [lon, lat]}
                if 'latitude' in v and 'longitude' in v:
                    try:
                        return float(v['latitude']), float(v['longitude'])
                    except Exception:
                        pass
                coords = v.get('coordinates') or v.get('coordinates_')
                if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                    try:
                        # Socrata coords typically [lon, lat]
                        return float(coords[1]), float(coords[0])
                    except Exception:
                        pass
        # WKT string variants
        for key in ('location', 'location_1', 'the_geom', 'wkt'):
            v = rec.get(key)
            if isinstance(v, str) and 'POINT' in v.upper():
                m = re.search(r"POINT\s*\(([-0-9\.]+)\s+([-0-9\.]+)\)", v)
                if m:
                    try:
                        lon = float(m.group(1)); lat = float(m.group(2))
                        return lat, lon
                    except Exception:
                        pass
        return None

    def _add_kcmo_open_data_layers(self, m: folium.Map):
        """Add KCMO authoritative layers (Hospitals/Clinics, Schools) if datasets are configured or found."""
        # Dataset IDs can be provided via env variables
        hospital_ids = [
            os.getenv('KCMO_DATASET_HOSPITALS') or '',
            # Add known IDs here if available in the future
        ]
        school_ids = [
            os.getenv('KCMO_DATASET_SCHOOLS') or '',
            # Add known IDs here if available in the future
        ]

        # Hospitals/Clinics
        for dsid in hospital_ids:
            dsid = (dsid or '').strip()
            if not dsid:
                continue
            recs = self._socrata_fetch(dsid, params={}, cache_key=f"kcmo_{dsid}")
            if not recs:
                continue
            fg = folium.FeatureGroup(name='KCMO Hospitals/Clinics', show=False)
            count = 0
            for rec in recs:
                pt = self._parse_socrata_point(rec)
                if not pt:
                    continue
                lat, lon = pt
                name = rec.get('name') or rec.get('facility_name') or rec.get('provider_name') or 'Hospital/Clinic'
                addr = rec.get('address') or rec.get('street_address') or rec.get('location_address') or ''
                tooltip = name
                popup = f"<b>{name}</b><br/>{addr}"
                try:
                    folium.Marker(
                        location=[lat, lon],
                        tooltip=tooltip,
                        popup=popup,
                        icon=folium.Icon(color='red', icon='hospital', prefix='fa')
                    ).add_to(fg)
                    count += 1
                except Exception:
                    pass
            if count:
                fg.add_to(m)
                break  # first working dataset is enough

        # Schools
        for dsid in school_ids:
            dsid = (dsid or '').strip()
            if not dsid:
                continue
            recs = self._socrata_fetch(dsid, params={}, cache_key=f"kcmo_{dsid}")
            if not recs:
                continue
            fg = folium.FeatureGroup(name='KCMO Schools', show=False)
            count = 0
            for rec in recs:
                pt = self._parse_socrata_point(rec)
                if not pt:
                    continue
                lat, lon = pt
                name = rec.get('name') or rec.get('school_name') or rec.get('facility_name') or 'School'
                addr = rec.get('address') or rec.get('street_address') or ''
                tooltip = name
                popup = f"<b>{name}</b><br/>{addr}"
                try:
                    folium.Marker(
                        location=[lat, lon],
                        tooltip=tooltip,
                        popup=popup,
                        icon=folium.Icon(color='green', icon='school', prefix='fa')
                    ).add_to(fg)
                    count += 1
                except Exception:
                    pass
            if count:
                fg.add_to(m)
                break

    def _add_osm_infrastructure_layers(self, m: folium.Map):
        """Add multiple OSM-derived infrastructure overlays with toggles."""
        south, west, north, east = self._kc_bounds()
        bbox = f"({south},{west},{north},{east})"

        # Define OSM queries for each layer
        layers = [
            {
                'name': 'Roads: Major Highways',
                'key': 'highways',
                'color': '#f39c12',
                'weight': 3,
                'query': f"[out:json][timeout:25];(way['highway'~'motorway|trunk|primary']{bbox};);out geom;"
            },
            {
                'name': 'Rail Network',
                'key': 'rail',
                'color': '#2c3e50',
                'weight': 2,
                'dashArray': '6,6',
                'query': f"[out:json][timeout:25];(way['railway'~'rail|light_rail|tram']{bbox};);out geom;"
            },
            {
                'name': 'Transit Hubs',
                'key': 'transit',
                'marker_icon': 'bus',
                'query': f"[out:json][timeout:25];(nwr['amenity'='bus_station']{bbox};nwr['public_transport'='station']{bbox};);out center;"
            },
            {
                'name': 'Airports',
                'key': 'airports',
                'marker_icon': 'plane',
                'query': f"[out:json][timeout:25];(nwr['aeroway'='aerodrome']{bbox};);out center;"
            },
            {
                'name': 'Universities',
                'key': 'universities',
                'marker_icon': 'graduation-cap',
                'query': f"[out:json][timeout:25];(nwr['amenity'='university']{bbox};);out center;"
            },
            {
                'name': 'Hospitals',
                'key': 'hospitals',
                'marker_icon': 'hospital',
                'query': f"[out:json][timeout:25];(nwr['amenity'='hospital']{bbox};);out center;"
            },
            {
                'name': 'Schools/Colleges',
                'key': 'schools',
                'marker_icon': 'school',
                'query': f"[out:json][timeout:25];(nwr['amenity'='school']{bbox};nwr['amenity'='college']{bbox};);out center;"
            },
            {
                'name': 'Logistics & Intermodal',
                'key': 'logistics',
                'marker_icon': 'warehouse',
                'query': f"[out:json][timeout:25];(nwr['railway'='yard']{bbox};nwr['building'='warehouse']{bbox};nwr['landuse'='industrial']{bbox};);out center;"
            },
            {
                'name': 'Power Grid (Substations/Plants)',
                'key': 'power',
                'marker_icon': 'bolt',
                'query': f"[out:json][timeout:25];(nwr['power'~'substation|plant']{bbox};);out center;"
            }
        ]

        # Draw line layers (ways with geometry)
        for lyr in layers:
            if 'color' in lyr:
                data = self._overpass_fetch(lyr['key'], lyr['query'])
                if not data or 'elements' not in data:
                    continue
                fg = folium.FeatureGroup(name=lyr['name'], show=False)
                for el in data['elements']:
                    if el.get('type') == 'way' and 'geometry' in el:
                        coords = [(pt['lat'], pt['lon']) for pt in el['geometry']]
                        try:
                            folium.PolyLine(
                                locations=coords,
                                color=lyr['color'],
                                weight=lyr.get('weight', 2),
                                dash_array=lyr.get('dashArray')
                            ).add_to(fg)
                        except Exception:
                            pass
                fg.add_to(m)

        # Draw point layers (nwr center outputs)
        for lyr in layers:
            if 'marker_icon' in lyr:
                data = self._overpass_fetch(lyr['key'], lyr['query'])
                if not data or 'elements' not in data:
                    continue
                fg = folium.FeatureGroup(name=lyr['name'], show=False)
                for el in data['elements']:
                    lat = None
                    lon = None
                    if 'lat' in el and 'lon' in el:
                        lat, lon = el['lat'], el['lon']
                    elif 'center' in el and isinstance(el['center'], dict):
                        lat = el['center'].get('lat')
                        lon = el['center'].get('lon')
                    if lat is None or lon is None:
                        continue
                    name = el.get('tags', {}).get('name', lyr['name'])
                    try:
                        folium.Marker(
                            location=[lat, lon],
                            tooltip=name,
                            icon=folium.Icon(color='darkblue', icon=lyr['marker_icon'], prefix='fa')
                        ).add_to(fg)
                    except Exception:
                        pass
                fg.add_to(m)

        # Broadband (proxy) layer: shade counties by urban/suburban/rural classification
        try:
            broadband_group = folium.FeatureGroup(name='Broadband Availability (Proxy)', show=False)
            for county_name, geojson in KC_COUNTY_BOUNDARIES.items():
                ctype = COUNTY_CLASSIFICATION.get(county_name, 'suburban')
                # Proxy opacity: urban 0.6, suburban 0.4, rural 0.25
                opacity = 0.6 if ctype == 'urban' else (0.4 if ctype == 'suburban' else 0.25)
                color = '#1abc9c' if ctype == 'urban' else ('#16a085' if ctype == 'suburban' else '#0e6655')
                feature = {
                    'type': 'Feature',
                    'properties': {'name': county_name, 'classification': ctype},
                    'geometry': geojson
                }
                folium.GeoJson(
                    feature,
                    name=f"Broadband: {county_name}",
                    style_function=lambda x, color=color, opacity=opacity: {
                        'fillColor': color,
                        'color': color,
                        'weight': 1,
                        'fillOpacity': opacity,
                        'opacity': 0.5
                    },
                    tooltip=folium.Tooltip(f"{county_name} ({ctype.title()}) - Broadband proxy")
                ).add_to(broadband_group)
            broadband_group.add_to(m)
        except Exception as e:
            logger.debug(f"Broadband proxy layer skipped: {e}")
    
    def _create_cluster_boundaries(self, businesses: List[Dict], buffer: float = 0.01, clip_to_kc: bool = False) -> Optional[List[Tuple[float, float]]]:
        """Create smooth boundary around businesses using enhanced spatial accuracy if available."""
        if not SHAPELY_AVAILABLE or len(businesses) < 3:
            return None
            
        # Use spatial accuracy enhancer if available
        if self.spatial_enhancer:
            try:
                # Create a temporary cluster structure for enhancer
                temp_cluster = {
                    'businesses': businesses,
                    'type': 'mixed'  # Default type for boundary generation
                }
                
                # Enhance boundaries using spatial accuracy enhancer
                enhanced_clusters = self.spatial_enhancer.enhance_cluster_boundaries([temp_cluster])
                if enhanced_clusters:
                    enhanced_cluster = enhanced_clusters[0]
                    boundaries = enhanced_cluster.get('boundaries', [])
                    
                    if boundaries:
                        # Use highest confidence boundary
                        best_boundary = max(boundaries, key=lambda b: b.get('confidence', 0))
                        geometry = best_boundary.get('geometry')
                        
                        if geometry and hasattr(geometry, 'exterior'):
                            coords = list(geometry.exterior.coords)
                            return [(lat, lon) for lon, lat in coords]
            except Exception as e:
                logger.debug(f"Spatial enhancer failed, falling back to standard method: {e}")
        
        # Try concave hull via union-of-buffers before convex hull
        try:
            points = []
            for business in businesses:
                coord = self._get_business_coordinates(business)
                if coord:
                    lat, lon = coord
                    # Shapely uses (x, y) = (lon, lat)
                    points.append(Point(lon, lat))
            if len(points) < 3:
                return None
            # Radius ~ 3x the smoothing buffer to bridge close gaps
            concave = self._concave_hull_from_points(points, radius_deg=max(buffer * 3.0, 0.004))
            if concave is not None:
                geom = concave
                # Optionally clip to KC counties union to prevent spillover
                if clip_to_kc:
                    try:
                        kc_union = self._kc_union_polygon()
                        if kc_union is not None:
                            # Slightly expand county union to avoid visual clipping at edges
                            try:
                                geom = geom.intersection(kc_union)
                            except Exception:
                                pass
                    except Exception:
                        pass
                # Handle MultiPolygon by choosing the largest area polygon
                try:
                    if hasattr(geom, 'geoms'):
                        geom = max(list(geom.geoms), key=lambda g: getattr(g, 'area', 0))
                except Exception:
                    pass
                if hasattr(geom, 'exterior'):
                    coords = list(geom.exterior.coords)
                    return [(lat, lon) for lon, lat in coords]
        except Exception as e:
            logger.debug(f"Concave hull failed, falling back to convex hull: {e}")
        
        # Fallback to original convex hull method
        try:
            points = []
            for business in businesses:
                coord = self._get_business_coordinates(business)
                if coord:
                    lat, lon = coord
                    points.append(Point(lon, lat))
            
            if len(points) < 3:
                return None
            
            # Create multipoint and get convex hull
            multipoint = MultiPoint(points)
            hull = multipoint.convex_hull
            
            # Buffer for smoother boundary
            buffered = hull.buffer(buffer)

            # Optionally clip to KC counties union to prevent spillover
            if clip_to_kc:
                try:
                    kc_union = self._kc_union_polygon()
                    if kc_union is not None:
                        try:
                            buffered = buffered.intersection(kc_union)
                        except Exception:
                            pass
                except Exception as _:
                    pass
            
            # Extract coordinates; handle MultiPolygon by choosing the largest area polygon
            geom = buffered
            try:
                if hasattr(geom, 'geoms'):
                    # MultiPolygon: pick the largest by area
                    geom = max(list(geom.geoms), key=lambda g: getattr(g, 'area', 0))
            except Exception:
                pass

            if hasattr(geom, 'exterior'):
                coords = list(geom.exterior.coords)
                return [(lat, lon) for lon, lat in coords]
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating boundary: {e}")
            return None

    def _kc_union_polygon(self):
        """Return a cached shapely polygon of the KC county union for clipping."""
        if not SHAPELY_AVAILABLE:
            return None
        if not hasattr(self, "_kc_union_cache"):
            try:
                polys = [shape(geo) for geo in KC_COUNTY_BOUNDARIES.values()]
                # Slightly expand the union to avoid hard clipping at county edges
                self._kc_union_cache = unary_union(polys).buffer(0.02)
            except Exception:
                self._kc_union_cache = None
        return self._kc_union_cache

    def _concave_hull_from_points(self, points, radius_deg: float):
        """
        Build a concave hull by unioning small buffers around points.
        points: list of shapely Point (lon, lat)
        radius_deg: buffer radius in degrees (~0.005 ≈ ~0.5–0.6 km in KC)
        Returns shapely geometry or None.
        Note: Shapely expects (x, y) = (lon, lat). Leaflet/Folium uses (lat, lon).
        """
        try:
            if not points:
                return None
            # Union of small discs around each point to produce a concave silhouette
            unioned = unary_union([pt.buffer(radius_deg) for pt in points])
            # Smooth and clean minor artifacts
            # Lighter simplification to preserve organic outlines
            geom = unioned.buffer(0).simplify(radius_deg * 0.15, preserve_topology=True)
            return geom
        except Exception:
            return None

    
    def _create_subclusters(self, businesses: List[Dict], max_distance: float = 0.05) -> List[List[Dict]]:
        """Group businesses into geographic sub-clusters using DBSCAN"""
        if not SHAPELY_AVAILABLE or len(businesses) < 3:
            return [businesses]
            
        try:
            # Extract coordinates
            coords = []
            valid_businesses = []
            for business in businesses:
                coord = self._get_business_coordinates(business)
                if coord:
                    lat, lon = coord
                    coords.append([lat, lon])
                    valid_businesses.append(business)
            
            if len(coords) < 3:
                return [valid_businesses]
            
            coords_array = np.array(coords, dtype=float)
            # Scale longitude by cos(latitude) to approximate meters (guard empty/NaN)
            if coords_array.size == 0 or not np.isfinite(coords_array[:, 0]).any():
                mean_lat = self.kc_center[0]
            else:
                mean_lat = float(np.nanmean(coords_array[:, 0]))
            lon_scale = math.cos(math.radians(mean_lat)) if np.isfinite(mean_lat) else 0.0
            lon_scale = max(0.1, float(lon_scale))
            scaled = coords_array.copy()
            scaled[:,1] = scaled[:,1] * lon_scale
            # Use DBSCAN for geographic clustering in scaled degrees
            clustering = DBSCAN(eps=max_distance, min_samples=3).fit(scaled)

            
            # Group businesses by cluster label
            sub_clusters = {}
            for idx, label in enumerate(clustering.labels_):
                if label == -1:  # Noise points - add to nearest cluster
                    continue
                if label not in sub_clusters:
                    sub_clusters[label] = []
                sub_clusters[label].append(valid_businesses[idx])
            
            # Include noise points in nearest cluster
            for idx, label in enumerate(clustering.labels_):
                if label == -1 and len(sub_clusters) > 0:
                    # Find nearest cluster
                    min_dist = float('inf')
                    nearest_cluster = 0
                    point = coords_array[idx]
                    
                    for cluster_label, cluster_businesses in sub_clusters.items():
                        cluster_coords = []
                        for cb in cluster_businesses:
                            coord = self._get_business_coordinates(cb)
                            if coord:
                                cluster_coords.append(coord)
                        if not cluster_coords:
                            continue
                        cluster_coords = np.array(cluster_coords, dtype=float)
                        # scale lon by cos(mean_lat) for distance consistency (guard empty/NaN)
                        if cluster_coords.size == 0 or not np.isfinite(cluster_coords[:, 0]).any():
                            mean_lat2 = self.kc_center[0]
                        else:
                            mean_lat2 = float(np.nanmean(cluster_coords[:, 0]))
                        scale2 = math.cos(math.radians(mean_lat2)) if np.isfinite(mean_lat2) else 0.0
                        scale2 = max(0.1, float(scale2))
                        scaled_cluster = cluster_coords.copy()
                        scaled_cluster[:,1] *= scale2
                        scaled_point = point.copy()
                        scaled_point[1] *= scale2
                        dist = np.min(np.linalg.norm(scaled_cluster - scaled_point, axis=1))
                        if dist < min_dist:
                            min_dist = dist
                            nearest_cluster = cluster_label
                    
                    sub_clusters[nearest_cluster].append(valid_businesses[idx])
            
            return list(sub_clusters.values()) if sub_clusters else [valid_businesses]
            
        except Exception as e:
            logger.error(f"Error creating subclusters: {e}")
            return [businesses]
    
    def _add_cluster_visualization_improved(self, m: folium.Map, cluster: Dict, rank: int, color: str):
        """Add improved cluster visualization with boundaries and smart markers"""
        businesses = cluster.get('businesses', [])
        if not businesses:
            return
            
        # Create sub-clusters for better visualization
        sub_clusters = self._create_subclusters(businesses)
        
        cluster_name = cluster.get('name', f'Cluster {rank}')
        cluster_layer = folium.FeatureGroup(
            name=f"ðŸ“ {rank}. {cluster_name}",
            show=True
        )
        
        # Add boundary for each sub-cluster
        for i, sub_cluster in enumerate(sub_clusters):
            if len(sub_cluster) >= 3:
                boundary = self._create_cluster_boundaries(sub_cluster, clip_to_kc=True)
                if boundary:
                    # Add smooth boundary polygon
                    folium.Polygon(
                        locations=boundary,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fillOpacity=0.2,
                        weight=3,
                        dashArray='5, 5',
                        popup=self._create_detailed_cluster_popup(cluster, sub_cluster),
                        tooltip=f"{cluster_name} - Area {i+1}"
                    ).add_to(cluster_layer)
            
            # Add cluster center marker with metrics
            center = self._calculate_cluster_center(sub_cluster)
            if center:
                self._add_enhanced_center_marker(cluster_layer, center, cluster, len(sub_cluster), color, rank)
        
        # Add top business markers (limit to reduce clutter)
        self._add_smart_business_markers(cluster_layer, businesses, cluster, color)
        
        cluster_layer.add_to(m)
        return cluster_layer
    
    
    def _add_enhanced_center_marker(self, layer: folium.FeatureGroup, center: Tuple[float, float],
                                  cluster: Dict, business_count: int, color: str, rank: int):
        """Add enhanced center marker with metrics"""
        gdp_impact = cluster.get('projected_gdp_impact', 0) or cluster.get('metrics', {}).get('projected_gdp_impact', 0)
        
        # Create custom HTML marker
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
                #{rank}
            </div>
        </div>
        """
        
        folium.Marker(
            location=center,
            icon=folium.DivIcon(html=icon_html, icon_size=(50, 50), icon_anchor=(25, 25)),
            popup=self._create_detailed_cluster_popup(cluster),
            tooltip=f"{cluster.get('name')} - {business_count} businesses | ${gdp_impact/1e6:.1f}M GDP impact"
        ).add_to(layer)
    
    def _add_smart_business_markers(self, layer: folium.FeatureGroup, businesses: List[Dict], 
                                  cluster: Dict, color: str):
        """Add markers for top businesses only to reduce clutter"""
        # Sort businesses by score and size
        scored_businesses = []
        for business in businesses:
            score = business.get('composite_score', business.get('score', business.get('total_score', 0)))
            employees = business.get('employees', 0)
            coord = self._get_business_coordinates(business)
            if not coord:
                continue
            lat, lon = coord

            scored_businesses.append({
                'business': business,
                'importance': score * (1 + np.log10(max(employees, 1))),
                'lat': lat,
                'lon': lon
            })
        
        # Sort by importance and take top 15
        scored_businesses.sort(key=lambda x: x['importance'], reverse=True)
        
        for i, item in enumerate(scored_businesses[:15]):
            business = item['business']
            
            # Determine marker style based on ranking
            if i < 3:
                marker_color = 'gold'
                icon_color = 'black'
                size = 12
            elif i < 8:
                marker_color = color
                icon_color = 'white'
                size = 10
            else:
                marker_color = color
                icon_color = 'white'
                size = 8
            
            # Create circle marker
            folium.CircleMarker(
                location=[item['lat'], item['lon']],
                radius=size,
                popup=self._create_business_popup(business, cluster),
                tooltip=f"{business.get('name', 'Business')} - Score: {business.get('composite_score', 0):.0f}",
                color=marker_color,
                fill=True,
                fill_color=marker_color,
                fillOpacity=0.7,
                weight=2
            ).add_to(layer)
    
    def _create_detailed_cluster_popup(self, cluster: Dict, businesses: List[Dict] = None) -> str:
        """Create detailed popup content"""
        if businesses is None:
            businesses = cluster.get('businesses', [])
            
        gdp_impact = cluster.get('projected_gdp_impact', 0) or cluster.get('metrics', {}).get('projected_gdp_impact', 0)
        job_impact = cluster.get('projected_jobs', 0) or cluster.get('metrics', {}).get('projected_jobs', 0)
        
        return f"""
        <div style='width: 350px; font-family: Arial, sans-serif;'>
            <h4 style='margin: 0 0 10px 0; color: #333;'>{cluster.get('name', 'Cluster')}</h4>
            
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px;'>
                <div style='background: #f5f5f5; padding: 8px; border-radius: 4px;'>
                    <div style='font-size: 11px; color: #666;'>Businesses</div>
                    <div style='font-size: 18px; font-weight: bold; color: #333;'>
                        {len(businesses)}
                    </div>
                </div>
                <div style='background: #f5f5f5; padding: 8px; border-radius: 4px;'>
                    <div style='font-size: 11px; color: #666;'>Employees</div>
                    <div style='font-size: 18px; font-weight: bold; color: #333;'>
                        {sum(b.get('employees', 0) for b in businesses):,}
                    </div>
                </div>
            </div>
            
            <div style='background: #e3f2fd; padding: 10px; border-radius: 4px; margin-bottom: 10px;'>
                <h5 style='margin: 0 0 5px 0; color: #1976d2;'>Economic Impact</h5>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 5px;'>
                    <div>
                        <span style='font-size: 11px; color: #666;'>GDP Impact:</span><br>
                        <span style='font-weight: bold;'>${gdp_impact:,.0f}</span>
                    </div>
                    <div>
                        <span style='font-size: 11px; color: #666;'>Job Creation:</span><br>
                        <span style='font-weight: bold;'>{job_impact:,}</span>
                    </div>
                </div>
            </div>
            
            <div style='font-size: 12px; color: #666;'>
                <b>Cluster Type:</b> {cluster.get('type', 'mixed').replace('_', ' ').title()}<br>
                <b>Score:</b> {cluster.get('cluster_score', cluster.get('total_score', 0)):.1f}/100<br>
                <b>Top Businesses:</b>
                <ul style='margin: 5px 0; padding-left: 20px;'>
                    {self._format_top_businesses(businesses[:3])}
                </ul>
            </div>
        </div>
        """
    
    def _create_enhanced_cluster_popup(self, cluster: Dict, businesses: List[Dict], boundary_data: Dict) -> str:
        """Create enhanced popup with boundary accuracy information"""
        gdp_impact = cluster.get('projected_gdp_impact', 0) or cluster.get('metrics', {}).get('projected_gdp_impact', 0)
        job_impact = cluster.get('projected_jobs', 0) or cluster.get('metrics', {}).get('projected_jobs', 0)
        confidence = boundary_data.get('confidence', 0) * 100
        algorithm = boundary_data.get('algorithm', 'unknown')
        
        return f"""
        <div style='width: 380px; font-family: Arial, sans-serif;'>
            <h4 style='margin: 0 0 10px 0; color: #333;'>{cluster.get('name', 'Cluster')}</h4>
            
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px;'>
                <div style='background: #f5f5f5; padding: 8px; border-radius: 4px;'>
                    <div style='font-size: 11px; color: #666;'>Businesses</div>
                    <div style='font-size: 18px; font-weight: bold; color: #333;'>
                        {len(businesses)}
                    </div>
                </div>
                <div style='background: #e3f2fd; padding: 8px; border-radius: 4px;'>
                    <div style='font-size: 11px; color: #666;'>Boundary Confidence</div>
                    <div style='font-size: 18px; font-weight: bold; color: #1976d2;'>
                        {confidence:.0f}%
                    </div>
                </div>
            </div>
            
            <div style='background: #fff3e0; padding: 10px; border-radius: 4px; margin-bottom: 10px; border-left: 4px solid #ff9800;'>
                <h6 style='margin: 0 0 5px 0; color: #e65100;'>Spatial Accuracy</h6>
                <div style='font-size: 12px;'>
                    <strong>Algorithm:</strong> {algorithm.replace('_', ' ').title()}<br>
                    <strong>Precision:</strong> {boundary_data.get('precision', 'N/A')}<br>
                    <strong>Area Coverage:</strong> {boundary_data.get('area_coverage', 'N/A')}%
                </div>
            </div>
            
            <div style='background: #e8f5e8; padding: 10px; border-radius: 4px; margin-bottom: 10px;'>
                <h5 style='margin: 0 0 5px 0; color: #2e7d32;'>Economic Impact</h5>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 5px;'>
                    <div>
                        <span style='font-size: 11px; color: #666;'>GDP Impact:</span><br>
                        <span style='font-weight: bold;'>${gdp_impact:,.0f}</span>
                    </div>
                    <div>
                        <span style='font-size: 11px; color: #666;'>Job Creation:</span><br>
                        <span style='font-weight: bold;'>{job_impact:,}</span>
                    </div>
                </div>
            </div>
            
            <div style='font-size: 12px; color: #666;'>
                <b>Cluster Type:</b> {cluster.get('type', 'mixed').replace('_', ' ').title()}<br>
                <b>Score:</b> {cluster.get('cluster_score', cluster.get('total_score', 0)):.1f}/100<br>
                <b>Top Businesses:</b>
                <ul style='margin: 5px 0; padding-left: 20px;'>
                    {self._format_top_businesses(businesses[:3])}
                </ul>
            </div>
        </div>
        """
    
    def _format_top_businesses(self, businesses: List[Dict]) -> str:
        """Format top businesses for popup display"""
        items = []
        for b in businesses:
            name = b.get('name', 'Unknown Business')
            employees = b.get('employees', 0)
            items.append(f"<li>{name} ({employees} emp.)</li>")
        return "".join(items) if items else "<li>No business data available</li>"
    
    def _create_business_popup(self, business: Dict, cluster: Dict) -> str:
        """Create popup for individual business"""
        return f"""
        <div style='width: 250px;'>
            <h5 style='margin: 0 0 5px 0;'>{business.get('name', 'Business')}</h5>
            <div style='font-size: 12px;'>
                <b>Cluster:</b> {cluster.get('name', 'Unknown')}<br>
                <b>Industry:</b> {business.get('naics_code', 'N/A')}<br>
                <b>Employees:</b> {business.get('employees', 0)}<br>
                <b>Score:</b> {business.get('composite_score', business.get('score', 0)):.1f}/100<br>
                <b>Revenue Est:</b> ${business.get('revenue_estimate', 0):,.0f}
            </div>
        </div>
        """

    def _add_cluster_markers(self, m: folium.Map, results: Dict):
        """Add data-rich cluster visualizations"""
        clusters = results.get('steps', {}).get('cluster_optimization', {}).get('clusters', [])
        
        logger.info(f"Adding cluster markers for {len(clusters)} clusters")
        
        # Debug: Log cluster structure
        if clusters:
            logger.info(f"First cluster keys: {list(clusters[0].keys())}")
            businesses = clusters[0].get('businesses', [])
            if businesses:
                logger.info(f"First business in first cluster: {businesses[0]}")
        
        # Create a parent group for ALL discovered clusters
        discovered_clusters_group = folium.FeatureGroup(
            name=f"ðŸ“Š DISCOVERED CLUSTERS ({len(clusters)})",
            show=True,
            overlay=True,
            control=True
        )
        
        # Create individual toggleable layer for EACH discovered cluster
        cluster_layers = []
        for i, cluster in enumerate(clusters, 1):
            cluster_name = cluster.get('name', f'Cluster {i}')
            cluster_type = cluster.get('type', 'mixed')
            
            # Create a feature group for this specific cluster
            individual_cluster_layer = folium.FeatureGroup(
                name=f"  â””â”€ {i}. {cluster_name}",
                show=True,
                overlay=True,
                control=True
            )
            cluster_layers.append(individual_cluster_layer)
        
        # Create supporting layers (not individual clusters)
        business_layer = folium.FeatureGroup(
            name="ðŸ“ Business Locations", 
            show=False,
            overlay=True,
            control=True
        ) 
        opportunity_layer = folium.FeatureGroup(
            name="ðŸ’° Investment Opportunities",
            show=False,
            overlay=True,
            control=True
        )
        
        # For compatibility with existing code
        impact_layer = discovered_clusters_group
        
        # Use enhanced colors if available, fallback to basic colors
        if self.visualization_mode == 'improved':
            cluster_colors = self.enhanced_cluster_colors
        else:
            # Legacy color map for compatibility
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
                "mixed": "gray",
                "supply_chain": "darkblue",
                "biotech": "darkgreen",
                "advanced_manufacturing": "darkred",
                "technology_biosciences": "lightgreen",
                "logistics_manufacturing": "lightblue"
            }
        
        # Get user's optimization focus
        optimization_focus = results.get('parameters', {}).get('algorithm_params', {}).get('optimization_focus', 'balanced')
        
        # Log cluster analysis
        logger.info(f"Total clusters available: {len(clusters)}")
        clusters_to_show = len(clusters)
        logger.info(f"Showing all {clusters_to_show} clusters on map")
        
        # Process all clusters
        for rank, (cluster, cluster_layer) in enumerate(zip(clusters, cluster_layers), 1):
            logger.info(f"Processing cluster {rank}: {cluster.get('name', 'Unknown')}")
            cluster_type = cluster.get('type', 'mixed')
            color = cluster_colors.get(cluster_type, cluster_colors.get('mixed', 'gray'))
            
            # Check if we have coordinates for businesses
            businesses = cluster.get('businesses', [])
            has_coordinates = any(self._get_business_coordinates(business) for business in businesses)
            
            # Use improved visualization if available and we have coordinates
            if self.visualization_mode == 'improved' and has_coordinates:
                logger.info(f"Using improved visualization for cluster {rank}")
                self._add_cluster_visualization_improved(m, cluster, rank, color)
                continue
            
            # Calculate cluster metrics - handle both strategic and optimization formats
            if 'metrics' in cluster:
                # Strategic cluster format
                gdp_impact = cluster['metrics'].get('projected_gdp_impact', 0)
                job_count = cluster['metrics'].get('projected_jobs', 0)
            else:
                # Optimization cluster format
                gdp_impact = cluster.get('projected_gdp_impact', 0)
                job_count = cluster.get('projected_jobs', 0)
            
            business_count = cluster.get('business_count', 0)
            avg_score = cluster.get('cluster_score', cluster.get('total_score', 0))
            
            # Get businesses in cluster
            businesses = cluster.get('businesses', [])
            
            logger.info(f"Cluster {rank}: {cluster.get('name', 'Unknown')} has {len(businesses)} businesses")
            logger.info(f"Cluster data keys: {list(cluster.keys())}")
            
            if businesses:
                # Group businesses by county to create impact zones
                county_groups = {}
                for business in businesses:
                    county = business.get('county', 'Jackson County')
                    state = business.get('state', 'MO')
                    
                    # Handle empty or None county values
                    if not county or county == '':
                        county = 'Jackson County'
                        state = 'MO'
                    
                    # Use normalized county name
                    county_key = self.normalize_county_name(county, state)
                    
                    # Try to find match in county_coords
                    if county_key and county_key not in self.county_coords:
                        # Log for debugging
                        logger.warning(f"County key '{county_key}' not found in coordinates")
                        
                        # Try various formats
                        attempts = [
                            county_key,  # Original attempt
                            f"{county}, {state}",  # Without normalization
                            f"{county.replace(' County', '')} County, {state}",  # Ensure single " County"
                        ]
                        
                        for attempt in attempts:
                            if attempt in self.county_coords:
                                county_key = attempt
                                logger.info(f"Found match with '{attempt}'")
                                break
                        else:
                            # Still no match - try fuzzy matching
                            county_base = county.replace(' County', '').strip()
                            for coord_key in self.county_coords:
                                if county_base in coord_key and state in coord_key:
                                    county_key = coord_key
                                    logger.info(f"Fuzzy matched to '{coord_key}'")
                                    break
                    
                    # Log the mapping for debugging
                    logger.debug(f"Mapping business county '{county}' state '{state}' to key '{county_key}'")
                    
                    if county_key not in county_groups:
                        county_groups[county_key] = []
                    county_groups[county_key].append(business)
                
                # Log county groupings and matching
                logger.info(f"County groups created: {list(county_groups.keys())}")
                logger.info(f"Available county coords: {list(self.county_coords.keys())[:5]}...")
                
                # Log which counties matched and which didn't
                matched_counties = [k for k in county_groups.keys() if k in self.county_coords]
                unmatched_counties = [k for k in county_groups.keys() if k not in self.county_coords]
                
                if matched_counties:
                    logger.info(f"Matched counties: {matched_counties}")
                if unmatched_counties:
                    logger.warning(f"Unmatched counties (no coordinates): {unmatched_counties}")
                    # If we have unmatched counties, show the cluster at KC center as a fallback
                    if not matched_counties and unmatched_counties:
                        logger.info(f"No counties matched - showing cluster at KC center")
                        # Add cluster at KC center with info about unmatched counties
                        cluster_popup = f"""
                        <div style='width: 300px'>
                            <h4>{cluster.get('name', 'Cluster')} - Kansas City Region</h4>
                            <b>Cluster Type:</b> {cluster_type.title()}<br>
                            <b>Business Count:</b> {len(businesses)}<br>
                            <b>Counties:</b> {', '.join(unmatched_counties)}<br>
                            <hr>
                            <b>Projected Impact:</b><br>
                            â€¢ GDP: ${gdp_impact:,.0f}<br>
                            â€¢ Jobs: {job_count:,}<br>
                            <hr>
                            <small><i>Note: County coordinates not found for mapping</i></small>
                        </div>
                        """
                        
                        folium.Circle(
                            location=self.kc_center,
                            radius=10000 + rank * 2000,
                            popup=folium.Popup(cluster_popup, max_width=300),
                            color=color,
                            fill=True,
                            fillOpacity=0.4,
                            weight=3,
                            opacity=0.8,
                            tooltip=f"{cluster.get('name', 'Cluster')} - Click for details"
                        ).add_to(cluster_layer)
                
                # Create county overlays for each county with businesses
                for county_key, county_businesses in county_groups.items():
                    if county_key in KC_COUNTY_BOUNDARIES:
                        # Calculate impact metrics for this county
                        county_employees = sum(b.get('employees', 0) for b in county_businesses)
                        county_avg_score = sum(b.get('composite_score', 0) for b in county_businesses) / len(county_businesses)
                        
                        # Create cluster overlay popup
                        impact_popup = f"""
                        <div style='width: 300px'>
                            <h4>{cluster.get('name', 'Cluster')} - {county_key}</h4>
                            <b>Cluster Type:</b> {cluster_type.title()}<br>
                            <b>Businesses in County:</b> {len(county_businesses)}<br>
                            <b>Total Employees:</b> {county_employees:,}<br>
                            <b>Avg Business Score:</b> {county_avg_score:.1f}/100<br>
                            <hr>
                            <b>Projected Cluster Impact:</b><br>
                            â€¢ GDP: ${gdp_impact:,.0f}<br>
                            â€¢ Jobs: {job_count:,}<br>
                            â€¢ ROI: {(gdp_impact / max(cluster.get('total_revenue', 1), 1) - 1) * 100:.1f}%<br>
                            <hr>
                            <b>Rank:</b> #{rank} of {len(clusters)} clusters
                        </div>
                        """
                        
                        # Create GeoJSON feature for cluster overlay
                        feature = {
                            "type": "Feature",
                            "properties": {
                                "name": cluster.get('name', 'Cluster'),
                                "county": county_key,
                                "cluster_type": cluster_type,
                                "business_count": len(county_businesses),
                                "employees": county_employees
                            },
                            "geometry": KC_COUNTY_BOUNDARIES[county_key]
                        }
                        
                        # Calculate opacity based on cluster strength
                        base_opacity = 0.4
                        if optimization_focus == 'gdp':
                            opacity_boost = min(0.4, (gdp_impact / 1e10) * 0.3)  # More GDP = more visible
                        elif optimization_focus == 'jobs':
                            opacity_boost = min(0.4, (county_employees / 1000) * 0.3)  # More jobs = more visible
                        else:
                            opacity_boost = min(0.4, (county_avg_score / 100) * 0.3)  # Higher score = more visible
                        
                        fill_opacity = base_opacity + opacity_boost
                        
                        # Debug overlay color
                        logger.debug(f"Creating overlay for {county_key} with color: {color} (cluster type: {cluster_type})")
                        
                        # Add cluster overlay polygon
                        folium.GeoJson(
                            feature,
                            name=f"{cluster.get('name', 'Cluster')} - {county_key}",
                            style_function=lambda x, color=color, opacity=fill_opacity: {
                                'fillColor': color,
                                'color': color,
                                'weight': 3,
                                'fillOpacity': opacity,
                                'opacity': 0.8
                            },
                            highlight_function=lambda x: {'weight': 4, 'fillOpacity': 0.7},
                            tooltip=folium.Tooltip(f"{cluster.get('name', 'Cluster')} - {county_key}"),
                            popup=folium.Popup(impact_popup, max_width=300)
                        ).add_to(cluster_layer)
                    elif county_key in self.county_coords:
                        # Fallback to circles if no polygon boundary available
                        lat, lon = self.county_coords[county_key]
                        logger.warning(f"No polygon boundary for {county_key}, using circle at ({lat}, {lon})")
                        
                        # Calculate impact metrics
                        county_employees = sum(b.get('employees', 0) for b in county_businesses)
                        county_avg_score = sum(b.get('composite_score', 0) for b in county_businesses) / len(county_businesses)
                        
                        # Create simple popup
                        impact_popup = f"""
                        <div style='width: 300px'>
                            <h4>{cluster.get('name', 'Cluster')} - {county_key}</h4>
                            <b>Cluster Type:</b> {cluster_type.title()}<br>
                            <b>Businesses:</b> {len(county_businesses)}<br>
                            <b>Employees:</b> {county_employees:,}<br>
                            <hr>
                            <b>Note:</b> County boundary data not available
                        </div>
                        """
                        
                        # Add fallback circle
                        folium.Circle(
                            location=[lat, lon],
                            radius=8000,
                            popup=folium.Popup(impact_popup, max_width=300),
                            color=color,
                            fill=True,
                            fill_color=color,
                            fillOpacity=0.4,
                            weight=3,
                            opacity=0.8
                        ).add_to(cluster_layer)
                    
                    # Add business markers for counties with coordinates
                    if county_key in self.county_coords:
                        lat, lon = self.county_coords[county_key]
                        
                        # Add top 5 businesses as markers
                        top_businesses = sorted(county_businesses, 
                                              key=lambda x: x.get('composite_score', 
                                                           x.get('score', 
                                                           x.get('total_score', 0))), 
                                              reverse=True)[:5]
                        
                        for i, business in enumerate(top_businesses):
                            # Add slight offset for multiple markers
                            import random
                            marker_lat = lat + random.uniform(-0.02, 0.02)
                            marker_lon = lon + random.uniform(-0.02, 0.02)
                            
                            # Log business fields for debugging
                            if i == 0:  # Log first business only
                                logger.debug(f"Business fields: {list(business.keys())}")
                                logger.debug(f"Score fields - composite: {business.get('composite_score', 'N/A')}, " + 
                                           f"score: {business.get('score', 'N/A')}, total: {business.get('total_score', 'N/A')}")
                            
                            # Determine icon based on business strength
                            # Try multiple score fields for compatibility
                            score = business.get('composite_score', 
                                   business.get('score', 
                                   business.get('total_score', 0)))
                            
                            # Use different icon approach for better compatibility
                            if score >= 80:
                                icon = 'star'
                                icon_color = 'darkblue'  # Changed from 'gold' which might not work
                            elif score >= 60:
                                icon = 'certificate' 
                                icon_color = 'blue'
                            else:
                                icon = 'building'
                                icon_color = 'cadetblue'
                            
                            business_popup = f"""
                            <div style='width: 250px'>
                                <h5>{business.get('name', 'Business')}</h5>
                                <b>Industry:</b> {business.get('naics_code', 'N/A')}<br>
                                <b>Employees:</b> {business.get('employees', 0)}<br>
                                <b>Score:</b> {score:.1f}/100<br>
                                <b>Revenue Est:</b> ${business.get('revenue_estimate', 0):,.0f}<br>
                                <hr>
                                <b>Strengths:</b><br>
                                â€¢ Innovation: {business.get('innovation_score', 0):.0f}<br>
                                â€¢ Market Potential: {business.get('market_potential_score', 0):.0f}<br>
                                â€¢ Competition: {business.get('competition_score', 0):.0f}
                            </div>
                            """
                            
                            # Create marker with cluster-specific styling
                            marker_icon = folium.Icon(
                                color=color,  # Use cluster color instead of icon_color
                                icon=icon,
                                prefix='fa'
                            )
                            
                            folium.Marker(
                                location=[marker_lat, marker_lon],
                                popup=folium.Popup(business_popup, max_width=250),
                                icon=marker_icon,
                                tooltip=f"{business.get('name', 'Business')} (Score: {score:.0f})"
                            ).add_to(business_layer)
                        
                        logger.debug(f"Added {len(top_businesses)} business markers for {county_key}")
                
                # Add investment opportunity markers for high-potential areas
                # Check multiple score fields for compatibility
                cluster_score = cluster.get('total_score', cluster.get('cluster_score', 0))
                if cluster_score > 40 and len(businesses) >= 3:  # Further lowered thresholds
                    # Find the best county for this cluster
                    best_county = max(county_groups.items(), 
                                    key=lambda x: len(x[1]))
                    if best_county[0] in self.county_coords:
                        lat, lon = self.county_coords[best_county[0]]
                        
                        opp_popup = f"""
                        <div style='width: 280px'>
                            <h4>ðŸ’¡ Investment Opportunity</h4>
                            <b>Cluster:</b> {cluster.get('name', 'Unknown')}<br>
                            <b>Type:</b> {cluster_type.title()}<br>
                            <b>Location:</b> {best_county[0]}<br>
                            <hr>
                            <b>Why Invest Here:</b><br>
                            â€¢ High concentration of {cluster_type} businesses<br>
                            â€¢ Projected ROI: {(gdp_impact / max(cluster.get('total_revenue', 1), 1) - 1) * 100:.1f}%<br>
                            â€¢ Growth potential: {cluster.get('longevity_score', 0):.1f}/10<br>
                            â€¢ Risk level: {'Low' if cluster.get('risk_score', 100) < 30 else 'Medium'}
                        </div>
                        """
                        
                        folium.Marker(
                            location=[lat, lon],
                            popup=folium.Popup(opp_popup, max_width=280),
                            icon=folium.Icon(color='green', icon='dollar-sign', prefix='fa'),
                            tooltip="Investment Opportunity"
                        ).add_to(opportunity_layer)
                        
                        logger.info(f"Added investment opportunity marker for {cluster.get('name')} at {best_county[0]}")
            else:
                # Handle case when businesses list is empty - still show cluster on map
                logger.warning(f"Cluster {rank} '{cluster.get('name', 'Unknown')}' has no businesses to display on map")
                
                # Add a generic marker at KC center to indicate the cluster
                cluster_popup = f"""
                <div style='width: 300px'>
                    <h4>{cluster.get('name', 'Cluster')} - Kansas City Region</h4>
                    <b>Cluster Type:</b> {cluster_type.title()}<br>
                    <b>Business Count:</b> {cluster.get('business_count', 0)}<br>
                    <b>Total Employees:</b> {cluster.get('total_employees', 0):,}<br>
                    <hr>
                    <b>Projected Impact:</b><br>
                    â€¢ GDP: ${gdp_impact:,.0f}<br>
                    â€¢ Jobs: {job_count:,}<br>
                    <hr>
                    <b>Rank:</b> #{rank} of {len(clusters)} clusters<br>
                    <small><i>Note: Individual business locations not available</i></small>
                </div>
                """
                
                # Add a large circle at KC center to represent the cluster
                folium.Circle(
                    location=self.kc_center,
                    radius=8000 + rank * 1000,  # Vary radius by rank
                    popup=folium.Popup(cluster_popup, max_width=300),
                    color=color,
                    fill=True,
                    fillOpacity=0.3,
                    weight=3,
                    opacity=0.8,
                    tooltip=f"{cluster.get('name', 'Cluster')} - Click for details"
                ).add_to(cluster_layer)
        
        # Add all cluster layers to map
        discovered_clusters_group.add_to(m)
        for layer in cluster_layers:
            layer.add_to(m)
        
        # Add supporting layers to map
        business_layer.add_to(m)
        opportunity_layer.add_to(m)
        
        # Check if any markers were added
        any_markers_added = (len(list(impact_layer._children.values())) > 0 or 
                           len(list(business_layer._children.values())) > 0 or
                           len(list(opportunity_layer._children.values())) > 0)
        
        # If no markers were added, add a default marker
        if not any_markers_added and clusters:
            logger.warning("No cluster markers were added to map - adding default marker at KC center")
            default_popup = f"""
            <div style='width: 300px'>
                <h4>KC Cluster Analysis Results</h4>
                <b>Total Clusters Identified:</b> {len(clusters)}<br>
                <b>Status:</b> Geographic mapping unavailable<br>
                <hr>
                <p><b>Top Clusters:</b></p>
                <ul style='margin: 5px 0; padding-left: 20px;'>
            """
            for i, cluster in enumerate(clusters, 1):
                default_popup += f"<li>{cluster.get('name', 'Cluster')} ({cluster.get('business_count', 0)} businesses)</li>"
            default_popup += """
                </ul>
                <hr>
                <small><i>Note: Individual business locations could not be mapped. 
                See analysis results for full details.</i></small>
            </div>
            """
            
            folium.Marker(
                location=self.kc_center,
                popup=folium.Popup(default_popup, max_width=300),
                icon=folium.Icon(color='blue', icon='info-circle', prefix='fa'),
                tooltip="KC Cluster Analysis - Click for details"
            ).add_to(m)
        
        # Log what was added
        logger.info(f"Map layers added - Clusters: {len(clusters)}, Markers added: {any_markers_added}")
        
        # Add a control showing cluster count
        cluster_count_html = f'''
        <div style="position: absolute; top: 70px; left: 10px; z-index: 1000; 
                    background: white; padding: 10px; border-radius: 5px; 
                    border: 2px solid #666; font-size: 14px;">
            <strong>{len(clusters)}</strong> Viable Clusters Identified
            <br><small style="color: #666;">After applying all filters and thresholds</small>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(cluster_count_html))
    
    def _add_infrastructure_markers(self, layer_or_map, results: Dict):
        """Add infrastructure asset markers"""
        # Support both passing a layer or map
        if isinstance(layer_or_map, folium.Map):
            infra_group = folium.FeatureGroup(name="ðŸ—ï¸ Infrastructure Assets", overlay=True, control=True, show=False)
            target = infra_group
        else:
            target = layer_or_map
            infra_group = None
        
        # Major infrastructure points
        infrastructure = {
            "Kansas City International Airport": (39.2976, -94.7139, "airport"),
            "Union Station": (39.0855, -94.5859, "rail"),
            "BNSF Intermodal": (39.1167, -94.6331, "rail"),
            "I-35/I-70 Junction": (39.0748, -94.6076, "highway"),
            "UMKC": (39.0319, -94.5765, "university"),
            "KU Medical Center": (39.0565, -94.6093, "university")
        }
        
        for name, (lat, lon, infra_type) in infrastructure.items():
            icon = self.infrastructure_icons.get(infra_type, "info")
            
            folium.Marker(
                location=[lat, lon],
                popup=f"<b>{name}</b><br>Type: {infra_type}",
                icon=folium.Icon(color='darkblue', icon=icon, prefix='fa')
            ).add_to(target)
        
        # Only add to map if we created the group
        if infra_group and isinstance(layer_or_map, folium.Map):
            infra_group.add_to(layer_or_map)
    
    def _add_geopolitical_risk_layer(self, m: folium.Map, results: Dict):
        """Add geopolitical risk visualization layer"""
        risk_layer = folium.FeatureGroup(name="Geopolitical Risk Indicators", overlay=True, control=True, show=False)
        
        market_data = results.get('market_data', {})
        geo_risks = market_data.get('geopolitical_risks', {})
        
        # Risk color scale
        risk_colors = {
            'low': '#00ff00',      # Green
            'moderate': '#ffff00',  # Yellow
            'high': '#ff0000'      # Red
        }
        
        # Add risk overlay for each cluster type
        clusters = results.get('steps', {}).get('cluster_optimization', {}).get('clusters', [])
        
        for cluster in clusters:  # All clusters
            cluster_type = cluster.get('type', 'mixed')
            risk_score = geo_risks.get(cluster_type, 0.5)
            
            # Determine risk level
            if risk_score > 0.7:
                risk_level = 'high'
                risk_icon = 'exclamation-triangle'
            elif risk_score > 0.4:
                risk_level = 'moderate'
                risk_icon = 'exclamation-circle'
            else:
                risk_level = 'low'
                risk_icon = 'check-circle'
            
            # Get cluster location (use first business or KC center)
            businesses = cluster.get('businesses', [])
            if businesses and isinstance(businesses[0], dict):
                county = businesses[0].get('county', '')
                location = self.county_coords.get(county, self.kc_center)
            else:
                location = self.kc_center
            
            # Add risk indicator
            risk_popup = f"""
            <div style='width: 250px'>
                <h4>Geopolitical Risk: {cluster_type.title()}</h4>
                <b>Risk Level:</b> {risk_level.title()} ({risk_score:.0%})<br>
                <hr>
                <b>Key Factors:</b><br>
            """
            
            # Add specific risk factors
            supply_chain = market_data.get('supply_chain_risks', {})
            if cluster_type == 'logistics':
                port_congestion = supply_chain.get('port_congestion', {}).get('west_coast', {}).get('congestion_level', 0)
                risk_popup += f"â€¢ Port Congestion: {port_congestion:.0%}<br>"
                risk_popup += f"â€¢ Shipping Cost Trend: {'Rising' if supply_chain.get('shipping_costs', {}).get('container_20ft', {}).get('monthly_change', 0) > 0 else 'Stable'}<br>"
            elif cluster_type == 'technology':
                risk_popup += f"â€¢ Semiconductor Availability: {supply_chain.get('critical_materials', {}).get('semiconductors', {}).get('availability', 'Unknown')}<br>"
                risk_popup += f"â€¢ China Trade Exposure: High<br>"
            elif cluster_type == 'manufacturing':
                risk_popup += f"â€¢ Steel Availability: {supply_chain.get('critical_materials', {}).get('steel', {}).get('availability', 'Unknown')}<br>"
                risk_popup += f"â€¢ Tariff Impact: Medium<br>"
            
            # Add trade partner info
            trade_data = market_data.get('trade_data', {})
            if 'top_partners' in trade_data:
                risky_partners = [p for p in trade_data['top_partners'][:3] if p.get('risk_score', 0) > 0.5]
                if risky_partners:
                    risk_popup += f"<br><b>Risky Trade Partners:</b><br>"
                    for partner in risky_partners:
                        risk_popup += f"â€¢ {partner['country']} ({partner['export_share']:.0%} of exports)<br>"
            
            risk_popup += "</div>"
            
            # Add circular risk zone
            folium.Circle(
                location=location,
                radius=20000,  # 20km radius
                popup=folium.Popup(risk_popup, max_width=250),
                color=risk_colors[risk_level],
                fill=True,
                fillOpacity=0.2,
                weight=2,
                dashArray='10, 10',  # Dashed line
                tooltip=f"Geopolitical Risk: {risk_level.title()}"
            ).add_to(risk_layer)
            
            # Add risk marker
            folium.Marker(
                location=location,
                popup=folium.Popup(risk_popup, max_width=250),
                icon=folium.Icon(color='red' if risk_level == 'high' else 'orange' if risk_level == 'moderate' else 'green',
                               icon=risk_icon, prefix='fa'),
                tooltip=f"{cluster_type.title()} - Risk: {risk_level.title()}"
            ).add_to(risk_layer)
        
        risk_layer.add_to(m)
    
    def _add_legend(self, m: folium.Map, results: Dict = None):
        """Add collapsible legend to map with actual cluster names"""
        # Get actual clusters from results to build dynamic legend
        clusters = []
        cluster_info = []
        
        if results:
            clusters = results.get('steps', {}).get('cluster_optimization', {}).get('clusters', [])
            
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
                "mixed": "gray",
                "supply_chain": "darkblue",
                "biotech": "darkgreen",
                "advanced_manufacturing": "darkred",
                "technology_biosciences": "lightgreen",
                "logistics_manufacturing": "lightblue"
            }
            
            # Build cluster info for legend
            for i, cluster in enumerate(clusters):
                cluster_type = cluster.get('type', 'mixed')
                color = cluster_colors.get(cluster_type, 'gray')
                name = cluster.get('name', f'Cluster {i+1}')
                # Truncate long names for legend
                if len(name) > 25:
                    name = name[:22] + '...'
                cluster_info.append((name, color))
        
        # Build cluster legend items HTML
        cluster_legend_items = ""
        if cluster_info:
            for name, color in cluster_info:
                cluster_legend_items += f'<span style="color:{color}; font-size:16px;">â—</span> {name}<br>\n'
        else:
            # Fallback to generic types if no results
            cluster_legend_items = '''<span style="color:blue; font-size:16px;">â—</span> Logistics<br>
                <span style="color:green; font-size:16px;">â—</span> Biosciences<br>
                <span style="color:purple; font-size:16px;">â—</span> Technology<br>
                <span style="color:orange; font-size:16px;">â—</span> Manufacturing<br>
                <span style="color:red; font-size:16px;">â—</span> Animal Health<br>
                <span style="color:gray; font-size:16px;">â—</span> Mixed<br>'''
        
        legend_html = f'''
        <div id="legendContainer" style="position: absolute; 
                    bottom: 30px; right: 10px; z-index:1000;">
            <button id="legendToggle" onclick="toggleLegend()" style="
                background-color: white;
                border: 2px solid grey;
                border-radius: 5px;
                padding: 5px 10px;
                cursor: pointer;
                font-size: 12px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                margin-bottom: 5px;
                display: flex;
                align-items: center;
                gap: 5px;
            ">
                <i class="fa fa-info-circle"></i> <span id="legendButtonText">Show Legend</span>
            </button>
            
            <div id="legendContent" style="
                width: 250px; height: auto; 
                background-color: white; border:2px solid grey; 
                font-size:11px; padding: 8px; border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                display: none;">
                
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;
                            cursor: move;" id="legendHeader">
                    <h5 style="margin: 0; font-size: 14px;">KC Cluster Analysis</h5>
                    <button onclick="toggleLegend()" style="
                        background: none;
                        border: none;
                        cursor: pointer;
                        font-size: 14px;
                        color: #666;
                        padding: 0 5px;
                    ">Ã—</button>
                </div>
                
                <div style="background:#f0f8ff;padding:5px;margin-bottom:10px;border-radius:3px;">
                    <b style="color:#003366;">ðŸ“Š DISCOVERED CLUSTERS</b>
                </div>
                {cluster_legend_items}
                <small style="font-size:10px;"><i>Circle size = Economic Impact</i></small><br><br>
                
                <div style="background:#f5f5f5;padding:5px;margin-bottom:10px;border-radius:3px;">
                    <b style="color:#666;">ðŸ—ºï¸ MAP LAYERS</b>
                </div>
                
                <b>Toggle Layers:</b><br>
                <small style="font-size:10px;">
                <i class="fa fa-check-square" style="color:#4CAF50"></i> = Visible | 
                <i class="fa fa-square" style="color:#999"></i> = Hidden
                </small><br><br>
                
                <b>Business Markers:</b><br>
                <i class="fa fa-star" style="color:gold"></i> Top Performers (80+)<br>
                <i class="fa fa-certificate" style="color:lightblue"></i> Strong (60-79)<br>
                <i class="fa fa-building" style="color:lightgray"></i> Average (< 60)<br><br>
                
                <b>County Types:</b><br>
                <span style="background:#FF6B6B;color:white;padding:2px 6px;border-radius:3px;">Urban</span>
                <span style="background:#4ECDC4;color:white;padding:2px 6px;border-radius:3px;">Suburban</span>
                <span style="background:#95E1D3;color:white;padding:2px 6px;border-radius:3px;">Rural</span>
            </div>
        </div>
        
        <script>
        function toggleLegend() {{
            var content = document.getElementById('legendContent');
            var buttonText = document.getElementById('legendButtonText');
            if (content.style.display === 'none') {{
                content.style.display = 'block';
                buttonText.textContent = 'Hide Legend';
            }} else {{
                content.style.display = 'none';
                buttonText.textContent = 'Show Legend';
            }}
        }}
        
        // Auto-show legend after map loads
        setTimeout(function() {{
            var content = document.getElementById('legendContent');
            var buttonText = document.getElementById('legendButtonText');
            if (content && content.style.display === 'none') {{
                content.style.display = 'block';
                if (buttonText) buttonText.textContent = 'Hide Legend';
            }}
        }}, 1000);
        
        // Make legend draggable
        (function() {{
            var container = document.getElementById('legendContainer');
            var header = null;
            var isDragging = false;
            var currentX;
            var currentY;
            var initialX;
            var initialY;
            var xOffset = 0;
            var yOffset = 0;
            
            function dragStart(e) {{
                header = document.getElementById('legendHeader');
                if (e.target === header || header.contains(e.target)) {{
                    if (e.type === "touchstart") {{
                        initialX = e.touches[0].clientX - xOffset;
                        initialY = e.touches[0].clientY - yOffset;
                    }} else {{
                        initialX = e.clientX - xOffset;
                        initialY = e.clientY - yOffset;
                    }}
                    
                    if (e.target === header || e.target.tagName !== 'BUTTON') {{
                        isDragging = true;
                    }}
                }}
            }}
            
            function dragEnd(e) {{
                initialX = currentX;
                initialY = currentY;
                isDragging = false;
            }}
            
            function drag(e) {{
                if (isDragging) {{
                    e.preventDefault();
                    
                    if (e.type === "touchmove") {{
                        currentX = e.touches[0].clientX - initialX;
                        currentY = e.touches[0].clientY - initialY;
                    }} else {{
                        currentX = e.clientX - initialX;
                        currentY = e.clientY - initialY;
                    }}
                    
                    xOffset = currentX;
                    yOffset = currentY;
                    
                    container.style.transform = "translate(" + currentX + "px, " + currentY + "px)";
                }}
            }}
            
            container.addEventListener('touchstart', dragStart, false);
            container.addEventListener('touchend', dragEnd, false);
            container.addEventListener('touchmove', drag, false);
            
            container.addEventListener('mousedown', dragStart, false);
            container.addEventListener('mouseup', dragEnd, false);
            container.addEventListener('mousemove', drag, false);
        }})();
        </script>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
    
    def create_heat_map(self, results: Dict) -> str:
        """Create a growth potential map showing areas with highest opportunity"""
        try:
            logger.info("Creating growth potential map")
            m = folium.Map(
                location=self.kc_center, 
                zoom_start=9,
                tiles='OpenStreetMap',
                width='100%',
                height='100%'
            )
            try:
                plugins.Fullscreen(position='topright', title='Full Screen', title_cancel='Exit Full Screen').add_to(m)
            except Exception:
                pass
            
            clusters = results.get('steps', {}).get('cluster_optimization', {}).get('clusters', [])
            ml_predictions = results.get('ml_explanations', {})
            market_analysis = results.get('market_analysis', {})
            
            # Create feature groups
            growth_zones = folium.FeatureGroup(name='High Growth Zones')
            opportunity_areas = folium.FeatureGroup(name='Opportunity Areas')
            infrastructure_layer = folium.FeatureGroup(name='Infrastructure Access')
            
            # Calculate growth potential metrics for each area
            area_metrics = {}
            
            for cluster in clusters:
                cluster_name = cluster.get('name', 'Unknown')
                businesses = cluster.get('businesses', [])
                
                # Get ML confidence and predictions from cluster data
                ml_confidence = cluster.get('confidence_score', 0.5) * 100  # Convert to percentage
                ml_cluster_predictions = cluster.get('ml_predictions', {})
                # Calculate growth rate from predictions
                gdp_impact = ml_cluster_predictions.get('gdp_impact', 0)
                projected_gdp = cluster.get('projected_gdp_impact', 0)
                gdp_growth_rate = 0.1 if projected_gdp > 0 else 0.05  # 10% growth if positive impact
                
                # Get market scores
                market_score = market_analysis.get('market_scores', {}).get(cluster_name, 0.5)
                
                # Group by county for area analysis
                for business in businesses:
                    county = business.get('county', 'Jackson')
                    state = business.get('state', 'MO')
                    county_full = self.normalize_county_name(county, state)
                    
                    if county_full not in area_metrics:
                        area_metrics[county_full] = {
                            'growth_score': 0,
                            'opportunity_count': 0,
                            'avg_business_age': [],
                            'innovation_indicators': 0,
                            'market_favorability': 0,
                            'clusters': set()
                        }
                    
                    # Calculate growth indicators
                    from datetime import datetime as _dt
                    business_age = _dt.now().year - business.get('year_established', 2020)
                    area_metrics[county_full]['avg_business_age'].append(business_age)
                    area_metrics[county_full]['opportunity_count'] += 1
                    area_metrics[county_full]['innovation_indicators'] += business.get('patent_count', 0)
                    area_metrics[county_full]['growth_score'] += business.get('composite_score', 0) * (1 + gdp_growth_rate)
                    area_metrics[county_full]['market_favorability'] = max(area_metrics[county_full]['market_favorability'], market_score)
                    area_metrics[county_full]['clusters'].add(cluster_name)
            
            # Identify high-growth zones (bubble overlay)
            for county_full, metrics in area_metrics.items():
                if county_full in self.county_coords:
                    lat, lon = self.county_coords[county_full]
                    
                    # Calculate composite growth potential
                    avg_age = sum(metrics['avg_business_age']) / len(metrics['avg_business_age']) if metrics['avg_business_age'] else 10
                    youth_factor = max(0, (10 - avg_age) / 10)  # Younger businesses = more growth
                    
                    growth_potential = (
                        (metrics['growth_score'] / max(metrics['opportunity_count'], 1)) * 0.3 +
                        youth_factor * 100 * 0.2 +
                        min(metrics['innovation_indicators'] / 10, 1) * 100 * 0.2 +
                        metrics['market_favorability'] * 100 * 0.3
                    )
                    
                    # Determine zone type and styling
                    if growth_potential >= 70:
                        zone_color = '#e74c3c'  # Red - Hot growth zone
                        zone_name = "Hot Growth Zone"
                        opacity = 0.6
                    elif growth_potential >= 50:
                        zone_color = '#f39c12'  # Orange - High potential
                        zone_name = "High Potential Area"
                        opacity = 0.5
                    elif growth_potential >= 30:
                        zone_color = '#3498db'  # Blue - Emerging opportunity
                        zone_name = "Emerging Opportunity"
                        opacity = 0.4
                    else:
                        zone_color = '#95a5a6'  # Gray - Developing area
                        zone_name = "Developing Area"
                        opacity = 0.3
                    
                    # Add growth zone visualization
                    radius = 3000 + (growth_potential * 100)  # 3-13km based on potential
                    
                    folium.Circle(
                        location=[lat, lon],
                        radius=radius,
                        color=zone_color,
                        weight=3,
                        fill_color=zone_color,
                        fillOpacity=opacity,
                        popup=folium.Popup(
                            f"""<div style='width: 300px'>
                            <h5>{county_full} - {zone_name}</h5>
                            <b>Growth Potential Score:</b> {growth_potential:.1f}/100<br>
                            <b>Active Opportunities:</b> {metrics['opportunity_count']}<br>
                            <b>Average Business Age:</b> {avg_age:.1f} years<br>
                            <b>Innovation Activity:</b> {metrics['innovation_indicators']} patents<br>
                            <b>Market Favorability:</b> {metrics['market_favorability']*100:.0f}%<br>
                            <b>Clusters Present:</b> {len(metrics['clusters'])}<br>
                            <hr>
                            <small>Higher scores indicate better growth prospects</small>
                            </div>""",
                            max_width=350
                        )
                    ).add_to(growth_zones if growth_potential >= 50 else opportunity_areas)
                    
                    # Add markers for top growth areas
                    if growth_potential >= 70:
                        folium.Marker(
                            [lat, lon],
                            icon=folium.Icon(
                                color='red',
                                icon='fire',
                                prefix='fa'
                            ),
                            popup=f"Hot Growth Zone: {growth_potential:.0f}/100"
                        ).add_to(growth_zones)
                    
                    # Add growth potential labels
                    if growth_potential >= 50:
                        folium.Marker(
                            [lat + 0.05, lon],
                            icon=folium.DivIcon(
                                html=f"""<div style='font-size: 14px; font-weight: bold; 
                                         color: {zone_color}; text-align: center;
                                         background: white; padding: 2px 5px; border-radius: 3px;
                                         border: 2px solid {zone_color};'>
                                         {growth_potential:.0f}%
                                         </div>""",
                                icon_size=(50, 25),
                                icon_anchor=(25, 12)
                            )
                        ).add_to(growth_zones)
            
            # Add infrastructure indicators (airports, highways)
            # Major infrastructure points that boost growth potential
            infrastructure_points = [
                {"name": "KCI Airport", "lat": 39.2976, "lon": -94.7139, "type": "airport"},
                {"name": "Downtown Airport", "lat": 39.1231, "lon": -94.5928, "type": "airport"},
                {"name": "I-70/I-435 Junction", "lat": 39.0525, "lon": -94.4801, "type": "highway"},
                {"name": "I-35/I-70 Junction", "lat": 39.0689, "lon": -94.5795, "type": "highway"}
            ]
            
            for infra in infrastructure_points:
                icon_type = 'plane' if infra['type'] == 'airport' else 'road'
                folium.Marker(
                    [infra['lat'], infra['lon']],
                    icon=folium.Icon(
                        color='green',
                        icon=icon_type,
                        prefix='fa'
                    ),
                    popup=f"{infra['name']} - Key {infra['type']} infrastructure"
                ).add_to(infrastructure_layer)
            
            # Add layers to map and controls
            growth_zones.add_to(m)
            opportunity_areas.add_to(m)
            infrastructure_layer.add_to(m)
            
            # Add layer control (expanded for clarity)
            folium.LayerControl().add_to(m)
            # Keep default Leaflet control stacking

            # Layer control already added; keep map defaults (no search/fit logic)
            
            # Add legend
            legend_html = '''
            <div style="position: fixed; 
                        top: 10px; right: 10px; width: 220px;
                        background-color: white; z-index: 1000; 
                        border:2px solid grey; border-radius: 5px; padding: 10px">
                <h6 style="margin-top: 0;">Growth Potential</h6>
                <p style="margin: 0;"><span style="color: #e74c3c;">â—</span> Hot Growth Zone (70-100)</p>
                <p style="margin: 0;"><span style="color: #f39c12;">â—</span> High Potential (50-70)</p>
                <p style="margin: 0;"><span style="color: #3498db;">â—</span> Emerging (30-50)</p>
                <p style="margin: 0;"><span style="color: #95a5a6;">â—</span> Developing (<30)</p>
                <hr style="margin: 5px 0;">
                <p style="margin: 0; font-size: 0.85em;"><b>Factors:</b></p>
                <p style="margin: 0; font-size: 0.8em;">â€¢ Business growth scores</p>
                <p style="margin: 0; font-size: 0.8em;">â€¢ Innovation activity</p>
                <p style="margin: 0; font-size: 0.8em;">â€¢ Market conditions</p>
                <p style="margin: 0; font-size: 0.8em;">â€¢ Business age profile</p>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            return m._repr_html_()
            
        except Exception as e:
            logger.error(f"Error creating heat map: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return error map
            error_map = folium.Map(location=self.kc_center, zoom_start=9)
            folium.Marker(
                self.kc_center,
                popup=f"Error creating heatmap: {str(e)}",
                icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa')
            ).add_to(error_map)
            return error_map._repr_html_()
    
    def create_economic_impact_map(self, results: Dict) -> str:
        """Create an economic impact choropleth map showing cluster performance by county"""
        try:
            logger.info("Creating economic impact choropleth map")
            m = folium.Map(
                location=self.kc_center,
                zoom_start=9,
                tiles='OpenStreetMap',
                width='100%',
                height='100%'
            )
            try:
                plugins.Fullscreen(position='topright', title='Full Screen', title_cancel='Exit Full Screen').add_to(m)
            except Exception:
                pass
            
            clusters = results.get('steps', {}).get('cluster_optimization', {}).get('clusters', [])
            
            # Calculate county-level economic metrics
            county_metrics = {}
            
            for cluster in clusters:
                businesses = cluster.get('businesses', [])
                cluster_gdp = cluster.get('projected_gdp_impact', 0)
                cluster_jobs = cluster.get('projected_jobs', 0)
                # Extract cluster ROI (fraction) from known locations
                roi_frac = 0.0
                try:
                    roi_frac = float(cluster.get('roi', 0.0) or 0.0)
                    if roi_frac == 0.0 and 'ml_predictions' in cluster:
                        mp = cluster.get('ml_predictions') or {}
                        if 'expected_roi' in mp:
                            roi_frac = float(mp.get('expected_roi') or 0.0)
                        elif 'roi_percentage' in mp:
                            roi_frac = float(mp.get('roi_percentage') or 0.0) / 100.0
                except Exception:
                    roi_frac = 0.0
                
                # Count businesses per county
                county_business_count = {}
                for business in businesses:
                    county = business.get('county', 'Jackson')
                    state = business.get('state', 'MO')
                    county_full = self.normalize_county_name(county, state)
                    if county_full:
                        county_business_count[county_full] = county_business_count.get(county_full, 0) + 1
                
                # Distribute cluster metrics proportionally to counties
                total_businesses = len(businesses)
                if total_businesses > 0:
                    for county_full, count in county_business_count.items():
                        proportion = count / total_businesses
                        if county_full not in county_metrics:
                            county_metrics[county_full] = {
                                'gdp_impact': 0.0,
                                'jobs': 0,
                                'businesses': 0,
                                'roi_sum': 0.0,   # for weighted average
                                'roi_w': 0.0,
                                'clusters': []
                            }
                        county_metrics[county_full]['gdp_impact'] += cluster_gdp * proportion
                        county_metrics[county_full]['jobs'] += int(cluster_jobs * proportion)
                        county_metrics[county_full]['businesses'] += count
                        county_metrics[county_full]['clusters'].append(cluster.get('name', 'Unknown'))
                        # Accumulate ROI weighted by proportion
                        county_metrics[county_full]['roi_sum'] += roi_frac * proportion
                        county_metrics[county_full]['roi_w'] += proportion
            
            # Post-process: compute ROI average per county
            for k, v in county_metrics.items():
                w = max(1e-9, float(v.get('roi_w', 0.0)))
                v['roi_avg'] = float(v.get('roi_sum', 0.0)) / w

            # Create feature groups
            county_layer = folium.FeatureGroup(name='County Economic Impact (Circles)', show=False)
            bivariate_layer = folium.FeatureGroup(name='Economic Impact (Bivariate)', show=True)
            cluster_markers = folium.FeatureGroup(name='Cluster Centers')
            
            # Process each county
            max_gdp = max((m['gdp_impact'] for m in county_metrics.values()), default=1)
            jobs_values = [m['jobs'] for m in county_metrics.values()] or [0]
            roi_values = [m.get('roi_avg', 0.0) for m in county_metrics.values()] or [0.0]
            # Compute tercile thresholds for bivariate mapping
            def _terciles(vals):
                s = sorted(vals)
                if len(s) < 3:
                    return (min(vals), sum(vals)/len(vals) if vals else 0, max(vals))
                import math
                def q(p):
                    idx = int(p * (len(s)-1))
                    return s[idx]
                return (q(1/3), q(2/3))
            jobs_t1, jobs_t2 = _terciles(jobs_values)
            roi_t1, roi_t2 = _terciles(roi_values)

            # Bivariate 3x3 palette (ROI = columns low→high, Jobs = rows low→high)
            palette = {
                (0,0): '#e8e8e8', (0,1): '#b5c0da', (0,2): '#6c83b5',
                (1,0): '#b8d6be', (1,1): '#90b2b3', (1,2): '#567994',
                (2,0): '#73ae80', (2,1): '#5a9178', (2,2): '#2a5a5b'
            }
            def _bin(val, t1, t2):
                return 0 if val <= t1 else (1 if val <= t2 else 2)
            # Precompute county colors
            county_colors = {}
            for county_full, metrics in county_metrics.items():
                jb = _bin(metrics['jobs'], jobs_t1, jobs_t2)
                rb = _bin(metrics.get('roi_avg', 0.0), roi_t1, roi_t2)
                county_colors[county_full] = palette[(jb, rb)]
            
            for county_full, metrics in county_metrics.items():
                if county_full in self.county_coords:
                    lat, lon = self.county_coords[county_full]
                    
                    # Calculate color intensity based on GDP impact
                    gdp_billions = metrics['gdp_impact'] / 1e9
                    intensity = min(metrics['gdp_impact'] / max_gdp, 1.0)
                    
                    # Create circle for county with size based on GDP impact
                    radius = 5000 + (intensity * 15000)  # 5km to 20km radius
                    
                    # Color based on performance tiers
                    if gdp_billions >= 1.0:
                        color = '#2ecc71'  # Green - High impact
                        tier = "Tier 1: High Impact"
                    elif gdp_billions >= 0.5:
                        color = '#f39c12'  # Orange - Medium impact
                        tier = "Tier 2: Medium Impact"
                    elif gdp_billions >= 0.25:
                        color = '#3498db'  # Blue - Growing potential
                        tier = "Tier 3: Growing"
                    else:
                        color = '#9b59b6'  # Purple - Emerging
                        tier = "Tier 4: Emerging"
                    
                    folium.Circle(
                        location=[lat, lon],
                        radius=radius,
                        color=color,
                        weight=3,
                        fill_color=color,
                        fillOpacity=0.3 + (intensity * 0.4),
                        popup=folium.Popup(
                            f"""<div style='width: 350px'>
                            <h5>{county_full}</h5>
                            <span class='badge' style='background-color: {color}; color: white;'>{tier}</span>
                            <hr>
                            <div style='display: flex; justify-content: space-between;'>
                                <div>
                                    <b>Economic Impact:</b><br>
                                    <span style='font-size: 1.5em;'>${gdp_billions:.2f}B</span>
                                </div>
                                <div>
                                    <b>Projected Jobs:</b><br>
                                    <span style='font-size: 1.5em;'>{metrics['jobs']:,}</span>
                                </div>
                            </div>
                            <hr>
                            <b>Active Businesses:</b> {metrics['businesses']}<br>
                            <b>Clusters in County:</b> {len(metrics['clusters'])}<br>
                            <details style='margin-top: 10px;'>
                                <summary>View Clusters</summary>
                                {'<br>'.join(f"â€¢ {c}" for c in metrics['clusters'])}
                            </details>
                            </div>""",
                            max_width=400
                        )
                    ).add_to(county_layer)
                    
                    # Add text label for major counties
                    if gdp_billions >= 0.25:
                        folium.Marker(
                            [lat, lon],
                            icon=folium.DivIcon(
                                html=f"""<div style='font-size: 12px; font-weight: bold; 
                                         color: {color}; text-align: center;
                                         text-shadow: 2px 2px 2px white, -2px -2px 2px white,
                                                      2px -2px 2px white, -2px 2px 2px white;'>
                                         ${gdp_billions:.1f}B
                                         </div>""",
                                icon_size=(60, 20),
                                icon_anchor=(30, 10)
                            )
                        ).add_to(county_layer)
            
            # Add top cluster markers
            for i, cluster in enumerate(clusters[:10]):  # Top 10 clusters
                businesses = cluster.get('businesses', [])
                if businesses:
                    # Calculate cluster center
                    lats, lons = [], []
                    for business in businesses[:20]:  # Sample for performance
                        county = business.get('county', 'Jackson')
                        state = business.get('state', 'MO')
                        county_full = self.normalize_county_name(county, state)
                        if county_full and county_full in self.county_coords:
                            lat, lon = self.county_coords[county_full]
                            # Add small jitter
                            lat += (hash(business.get('name', '')) % 100 - 50) * 0.002
                            lon += (hash(business.get('name', '')) % 100 - 50) * 0.002
                            lats.append(lat)
                            lons.append(lon)
                    
                    if lats and lons:
                        center_lat = sum(lats) / len(lats)
                        center_lon = sum(lons) / len(lons)
                        
                        # Cluster type icons and colors
                        type_config = {
                            'logistics': {'icon': 'truck', 'color': 'cadetblue'},
                            'manufacturing': {'icon': 'industry', 'color': 'darkred'},
                            'technology': {'icon': 'microchip', 'color': 'darkpurple'},
                            'biosciences': {'icon': 'dna', 'color': 'green'},
                            'mixed': {'icon': 'building', 'color': 'orange'}
                        }
                        
                        cluster_type = cluster.get('type', 'mixed')
                        config = type_config.get(cluster_type, {'icon': 'building', 'color': 'gray'})
                        
                        folium.Marker(
                            [center_lat, center_lon],
                            popup=folium.Popup(
                                f"""<div style='width: 300px'>
                                <h5>#{i+1}: {cluster.get('name', f'Cluster {i+1}')}</h5>
                                <span class='badge' style='background-color: {config['color']}; color: white;'>
                                    <i class='fa fa-{config['icon']}'></i> {cluster_type.title()}
                                </span>
                                <hr>
                                <b>Businesses:</b> {cluster.get('business_count', 0)}<br>
                                <b>GDP Impact:</b> ${cluster.get('projected_gdp_impact', 0)/1e9:.2f}B<br>
                                <b>Jobs:</b> {cluster.get('projected_jobs', 0):,}<br>
                                <b>ML Confidence:</b> {cluster.get('confidence_score', 0.5)*100:.1f}%
                                </div>""",
                                max_width=350
                            ),
                            icon=folium.Icon(
                                color=config['color'],
                                icon=config['icon'],
                                prefix='fa'
                            )
                        ).add_to(cluster_markers)
            
            # (Original version had no choropleth overlay)
            # Add bivariate choropleth overlay
            for county_name, geojson in KC_COUNTY_BOUNDARIES.items():
                feature = {
                    "type": "Feature",
                    "properties": {"name": county_name},
                    "geometry": geojson
                }
                color = county_colors.get(county_name, '#e8e8e8')
                folium.GeoJson(
                    feature,
                    name=county_name,
                    style_function=lambda x, c=color: {
                        'fillColor': c,
                        'color': '#333333',
                        'weight': 1,
                        'fillOpacity': 0.6,
                        'opacity': 0.6
                    },
                    tooltip=folium.Tooltip(
                        f"{county_name}: ROI {county_metrics.get(county_name,{}).get('roi_avg',0)*100:.1f}% | Jobs {county_metrics.get(county_name,{}).get('jobs',0):,}",
                        sticky=True
                    )
                ).add_to(bivariate_layer)

            # Add layers to map
            county_layer.add_to(m)
            bivariate_layer.add_to(m)
            cluster_markers.add_to(m)
            
            # Add layer control (expanded for clarity)
            folium.LayerControl().add_to(m)
            # Keep default Leaflet control stacking
            
            # Add comprehensive legend
            legend_html = '''
            <div style="position: fixed; 
                        top: 10px; right: 10px; width: 250px;
                        background-color: white; z-index: 1000; 
                        border:2px solid grey; border-radius: 5px; padding: 10px">
                <h6 style="margin-top: 0;">Economic Impact Tiers (Circles)</h6>
                <p style="margin: 2px 0;"><span style="color: #2ecc71; font-size: 1.2em;">â—</span> Tier 1: High Impact (â‰¥$1B)</p>
                <p style="margin: 2px 0;"><span style="color: #f39c12; font-size: 1.2em;">â—</span> Tier 2: Medium ($0.5-1B)</p>
                <p style="margin: 2px 0;"><span style="color: #3498db; font-size: 1.2em;">â—</span> Tier 3: Growing ($0.25-0.5B)</p>
                <p style="margin: 2px 0;"><span style="color: #9b59b6; font-size: 1.2em;">â—</span> Tier 4: Emerging (<$0.25B)</p>
                <hr style="margin: 8px 0;">
                <h6 style="margin: 5px 0;">Bivariate (ROI vs Jobs)</h6>
                <div style="display:flex; align-items:center; gap:10px;">
                  <div>
                    <div style="display:grid; grid-template-columns: repeat(3, 16px); grid-gap:2px;">
                      <div style="width:16px; height:16px; background:#e8e8e8"></div>
                      <div style="width:16px; height:16px; background:#b5c0da"></div>
                      <div style="width:16px; height:16px; background:#6c83b5"></div>
                      <div style="width:16px; height:16px; background:#b8d6be"></div>
                      <div style="width:16px; height:16px; background:#90b2b3"></div>
                      <div style="width:16px; height:16px; background:#567994"></div>
                      <div style="width:16px; height:16px; background:#73ae80"></div>
                      <div style="width:16px; height:16px; background:#5a9178"></div>
                      <div style="width:16px; height:16px; background:#2a5a5b"></div>
                    </div>
                  </div>
                  <div style="font-size: 11px;">
                    <div>Rows: Jobs (low→high)</div>
                    <div>Cols: ROI (low→high)</div>
                  </div>
                </div>
                <hr style="margin: 8px 0;">
                <h6 style="margin: 5px 0;">Cluster Types</h6>
                <p style="margin: 2px 0; font-size: 0.85em;"><i class="fa fa-truck"></i> Logistics</p>
                <p style="margin: 2px 0; font-size: 0.85em;"><i class="fa fa-industry"></i> Manufacturing</p>
                <p style="margin: 2px 0; font-size: 0.85em;"><i class="fa fa-microchip"></i> Technology</p>
                <p style="margin: 2px 0; font-size: 0.85em;"><i class="fa fa-dna"></i> Biosciences</p>
                <p style="margin: 2px 0; font-size: 0.85em;"><i class="fa fa-building"></i> Mixed</p>
                <hr style="margin: 8px 0;">
                <p style="margin: 0; font-size: 0.8em;"><i>Circle size = GDP impact</i></p>
                <p style="margin: 0; font-size: 0.8em;"><i>Opacity = relative strength</i></p>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            return m._repr_html_()
            
        except Exception as e:
            logger.error(f"Error creating economic impact map: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return error map
            error_map = folium.Map(location=self.kc_center, zoom_start=9)
            folium.Marker(
                self.kc_center,
                popup=f"Error creating economic impact map: {str(e)}",
                icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa')
            ).add_to(error_map)
            return error_map._repr_html_()
    
    def create_cluster_comparison_map(self, scenarios: List[Dict]) -> str:
        """Create map comparing multiple scenarios"""
        m = folium.Map(location=self.kc_center, zoom_start=9)
        
        # Create feature groups for each scenario
        for i, scenario in enumerate(scenarios):
            fg = folium.FeatureGroup(name=f"Scenario {i+1}")
            
            clusters = scenario.get('steps', {}).get('cluster_optimization', {}).get('clusters', [])
            
            # Add markers for this scenario
            for cluster in clusters[:3]:
                businesses = cluster.get('businesses', [])
                cluster_type = cluster.get('type', 'mixed')
                
                for business in businesses[:10]:
                    county = business.get('county', 'Jackson County, MO')
                    if county in self.county_coords:
                        lat, lon = self.county_coords[county]
                        
                        folium.CircleMarker(
                            location=[lat, lon],
                            radius=5,
                            popup=f"Scenario {i+1}: {cluster_type}",
                            color=f"#{i*50:02x}{100:02x}{200-i*50:02x}",  # Different colors
                            fill=True,
                            fillOpacity=0.7
                        ).add_to(fg)
            
            fg.add_to(m)
        
        folium.LayerControl().add_to(m)
        
        return m._repr_html_()
