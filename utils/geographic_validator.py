"""Geographic data validation and correction utilities for KC Cluster Prediction Tool"""

import logging
import math
import json
from typing import Dict, List, Tuple, Optional, Union
from shapely.geometry import Point, Polygon, MultiPoint
from shapely.ops import unary_union
import numpy as np

logger = logging.getLogger(__name__)

class GeographicValidator:
    """Comprehensive geographic data validation and correction system"""
    
    def __init__(self):
        # Define KC MSA bounds with proper coordinate system
        self.kc_bounds = {
            'min_lat': 38.2, 'max_lat': 39.7,
            'min_lon': -95.2, 'max_lon': -94.0
        }
        
        # Load county boundaries (will be enhanced with precise data)
        self.county_boundaries = self._load_county_boundaries()
        
        # Coordinate precision thresholds
        self.precision_thresholds = {
            'high_precision': 5,  # Decimal places for high precision
            'medium_precision': 4,  # Decimal places for medium precision
            'low_precision': 3     # Decimal places for low precision
        }
    
    def validate_coordinates(self, lat: float, lon: float) -> Dict:
        """Comprehensive coordinate validation with detailed feedback"""
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'corrected': None,
            'precision_level': 'unknown',
            'county_match': None
        }
        
        # Basic type and range validation
        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            result['errors'].append(f"Invalid coordinate format: {lat}, {lon}")
            return result
        
        # Check for NaN or infinite values
        if math.isnan(lat) or math.isnan(lon) or math.isinf(lat) or math.isinf(lon):
            result['errors'].append("Coordinates contain NaN or infinite values")
            return result
        
        # Basic geographic range validation
        if not (-90 <= lat <= 90):
            result['errors'].append(f"Latitude {lat} outside valid range [-90, 90]")
        
        if not (-180 <= lon <= 180):
            result['errors'].append(f"Longitude {lon} outside valid range [-180, 180]")
        
        # KC MSA bounds validation
        if not (self.kc_bounds['min_lat'] <= lat <= self.kc_bounds['max_lat']):
            result['warnings'].append(f"Latitude {lat} outside KC MSA bounds")
        
        if not (self.kc_bounds['min_lon'] <= lon <= self.kc_bounds['max_lon']):
            result['warnings'].append(f"Longitude {lon} outside KC MSA bounds")
        
        # Precision analysis
        lat_str = str(abs(lat)).split('.')[-1] if '.' in str(abs(lat)) else ''
        lon_str = str(abs(lon)).split('.')[-1] if '.' in str(abs(lon)) else ''
        
        lat_precision = len(lat_str)
        lon_precision = len(lon_str)
        avg_precision = (lat_precision + lon_precision) / 2
        
        if avg_precision >= self.precision_thresholds['high_precision']:
            result['precision_level'] = 'high'
        elif avg_precision >= self.precision_thresholds['medium_precision']:
            result['precision_level'] = 'medium'
        else:
            result['precision_level'] = 'low'
            result['warnings'].append(f"Low coordinate precision: {avg_precision} decimal places")
        
        # Check for common coordinate system issues
        if abs(lat) < 1 and abs(lon) < 1:
            result['warnings'].append("Coordinates may be in wrong format (too small)")
            # Try to correct - might be in degrees decimal format but missing leading zeros
            if abs(lat) < 1 and abs(lon) < 1:
                corrected_lat = lat * 100 if lat < 1 else lat
                corrected_lon = lon * 100 if lon < 1 else lon
                
                if (self.kc_bounds['min_lat'] <= corrected_lat <= self.kc_bounds['max_lat'] and
                    self.kc_bounds['min_lon'] <= corrected_lon <= self.kc_bounds['max_lon']):
                    result['corrected'] = (corrected_lat, corrected_lon)
                    result['warnings'].append("Applied coordinate correction (multiplied by 100)")
        
        # Validate against county boundaries
        county_match = self._point_in_county(lat, lon)
        result['county_match'] = county_match
        
        if not county_match:
            result['warnings'].append("Point doesn't fall within known county boundaries")
        
        result['valid'] = len(result['errors']) == 0
        return result
    
    def _load_county_boundaries(self) -> Dict:
        """Load precise county boundaries from enhanced data source"""
        # For now, use existing boundaries but this should be replaced
        # with precise GeoJSON from US Census TIGER/Line data
        try:
            from utils.kc_county_boundaries import KC_COUNTY_BOUNDARIES
            return KC_COUNTY_BOUNDARIES
        except ImportError:
            logger.warning("Could not load county boundaries")
            return {}
    
    def _point_in_county(self, lat: float, lon: float) -> Optional[str]:
        """Check if point falls within any KC county boundary"""
        point = Point(lon, lat)
        
        for county_name, boundary_data in self.county_boundaries.items():
            try:
                if boundary_data['type'] == 'Polygon':
                    coords = boundary_data['coordinates'][0]
                    polygon = Polygon([(lon, lat) for lon, lat in coords])
                    
                    if polygon.contains(point) or polygon.touches(point):
                        return county_name
            except Exception as e:
                logger.debug(f"Error checking county {county_name}: {e}")
                continue
        
        return None
    
    def correct_coordinates(self, lat: float, lon: float, business_data: Dict = None) -> Tuple[float, float]:
        """Attempt to correct coordinates using multiple strategies"""
        
        # Strategy 1: Use county centroid if coordinates are invalid
        validation = self.validate_coordinates(lat, lon)
        if not validation['valid']:
            if business_data and 'county' in business_data:
                county_name = self._normalize_county_name(business_data['county'])
                centroid = self._get_county_centroid(county_name)
                if centroid:
                    logger.info(f"Corrected coordinates using county centroid: {centroid}")
                    return centroid
        
        # Strategy 2: Geocode from address if available
        if business_data and 'address' in business_data:
            try:
                from data_preparation.smart_geocoder import SmartGeocoder
                geocoder = SmartGeocoder()
                geocoded = geocoder.geocode_address(business_data['address'])
                if geocoded:
                    logger.info(f"Corrected coordinates using address geocoding")
                    return geocoded
            except ImportError:
                logger.warning("SmartGeocoder not available for coordinate correction")
            except Exception as e:
                logger.debug(f"Address geocoding failed: {e}")
        
        # Strategy 3: Snap to nearest valid coordinate in same county
        if validation['county_match']:
            county_name = validation['county_match']
            snapped = self._snap_to_county(lat, lon, county_name)
            if snapped:
                logger.info(f"Corrected coordinates by snapping to county boundary")
                return snapped
        
        # If all correction strategies fail, return original
        return lat, lon
    
    def _normalize_county_name(self, county: str) -> str:
        """Normalize county name to match boundary data"""
        if not county:
            return ""
        
        county_str = str(county).strip()
        
        # Remove 'County' suffix if present and add it back in standard format
        if county_str.endswith('County'):
            county_core = county_str[:-6].strip()
        else:
            county_core = county_str
        
        # Add state if missing (assume MO for Jackson, etc.)
        if ',' not in county_core:
            # Simple state inference based on common KC counties
            if county_core in ['Jackson', 'Clay', 'Platte', 'Cass', 'Ray', 'Lafayette', 'Bates', 'Clinton', 'Caldwell']:
                county_str = f"{county_core} County, MO"
            elif county_core in ['Johnson', 'Wyandotte', 'Leavenworth', 'Miami', 'Linn']:
                county_str = f"{county_core} County, KS"
            else:
                county_str = f"{county_core} County"
        else:
            county_str = f"{county_core} County"
        
        return county_str
    
    def _get_county_centroid(self, county_name: str) -> Optional[Tuple[float, float]]:
        """Get centroid coordinates for a county"""
        if county_name in self.county_boundaries:
            boundary_data = self.county_boundaries[county_name]
            
            if boundary_data['type'] == 'Polygon':
                coords = boundary_data['coordinates'][0]
                # Calculate simple centroid
                lats = [lat for lon, lat in coords]
                lons = [lon for lon, lat in coords]
                
                centroid_lat = sum(lats) / len(lats)
                centroid_lon = sum(lons) / len(lons)
                
                return centroid_lat, centroid_lon
        
        return None
    
    def _snap_to_county(self, lat: float, lon: float, county_name: str) -> Optional[Tuple[float, float]]:
        """Snap coordinates to nearest point within county boundary"""
        if county_name not in self.county_boundaries:
            return None
        
        try:
            boundary_data = self.county_boundaries[county_name]
            
            if boundary_data['type'] == 'Polygon':
                coords = boundary_data['coordinates'][0]
                polygon = Polygon([(lon, lat) for lon, lat in coords])
                
                point = Point(lon, lat)
                
                # If point is already in polygon, return as-is
                if polygon.contains(point) or polygon.touches(point):
                    return lat, lon
                
                # Find nearest point on polygon boundary
                nearest_point = polygon.exterior.interpolate(polygon.exterior.project(point))
                
                return nearest_point.y, nearest_point.x
        except Exception as e:
            logger.debug(f"Error snapping to county {county_name}: {e}")
        
        return None
    
    def validate_cluster_boundaries(self, boundaries: List[List[Tuple[float, float]]]) -> Dict:
        """Validate cluster boundary polygons for common issues"""
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'topology_issues': [],
            'area_calculations': []
        }
        
        for i, boundary in enumerate(boundaries):
            if len(boundary) < 3:
                result['errors'].append(f"Boundary {i} has fewer than 3 points")
                result['valid'] = False
                continue
            
            try:
                # Create polygon and check for self-intersection
                polygon = Polygon([(lon, lat) for lat, lon in boundary])
                
                if not polygon.is_valid:
                    result['topology_issues'].append(f"Boundary {i} is self-intersecting")
                    result['valid'] = False
                
                # Check for extremely small area (likely error)
                area = polygon.area
                if area < 1e-8:  # Very small area in degrees
                    result['warnings'].append(f"Boundary {i} has extremely small area: {area}")
                
                result['area_calculations'].append({
                    'boundary_index': i,
                    'area': area,
                    'perimeter': polygon.length
                })
                
            except Exception as e:
                result['errors'].append(f"Error processing boundary {i}: {e}")
                result['valid'] = False
        
        return result
    
    def enhance_coordinate_precision(self, businesses: List[Dict]) -> List[Dict]:
        """Enhance coordinate precision for business locations"""
        enhanced = []
        
        for business in businesses:
            enhanced_business = business.copy()
            
            if 'lat' in business and 'lon' in business:
                lat, lon = business['lat'], business['lon']
                
                # Round to appropriate precision level
                validation = self.validate_coordinates(lat, lon)
                
                if validation['precision_level'] == 'low':
                    # Enhance to medium precision
                    enhanced_business['lat'] = round(lat, 4)
                    enhanced_business['lon'] = round(lon, 4)
                    enhanced_business['precision_enhanced'] = True
                elif validation['precision_level'] == 'medium':
                    # Enhance to high precision if we have confidence
                    if validation['county_match']:
                        enhanced_business['lat'] = round(lat, 6)
                        enhanced_business['lon'] = round(lon, 6)
                        enhanced_business['precision_enhanced'] = True
                
                # Validate enhanced coordinates
                new_validation = self.validate_coordinates(
                    enhanced_business['lat'], enhanced_business['lon']
                )
                
                if not new_validation['valid']:
                    # Fall back to original if enhancement breaks validity
                    enhanced_business['lat'] = lat
                    enhanced_business['lon'] = lon
                    enhanced_business['precision_enhanced'] = False
            
            enhanced.append(enhanced_business)
        
        return enhanced