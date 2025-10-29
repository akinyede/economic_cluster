"""Spatial accuracy enhancement techniques for cluster visualization"""

import logging
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from shapely.geometry import Point, Polygon, MultiPoint, LineString
from shapely.ops import unary_union, nearest_points
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import json

logger = logging.getLogger(__name__)

class SpatialAccuracyEnhancer:
    """Advanced spatial accuracy enhancement for cluster boundaries"""
    
    def __init__(self):
        self.kc_counties = self._load_precise_county_boundaries()
        self.road_network = None  # Will be loaded if available
        self.land_use_data = None  # Will be loaded if available
        
        # Enhancement parameters
        self.default_buffer_distance = 0.005  # degrees (~500m)
        self.min_points_for_boundary = 3
        self.max_points_for_simplification = 100
        
    def enhance_cluster_boundaries(self, clusters: List[Dict]) -> List[Dict]:
        """Apply multiple spatial accuracy enhancement techniques"""
        
        enhanced_clusters = []
        
        for i, cluster in enumerate(clusters):
            businesses = cluster.get('businesses', [])
            
            if not businesses:
                enhanced_clusters.append(cluster)
                continue
            
            logger.debug(f"Enhancing spatial accuracy for cluster {i+1} with {len(businesses)} businesses")
            
            # 1. Coordinate precision improvement
            enhanced_businesses = self._improve_coordinate_precision(businesses)
            
            # 2. Geographic sub-clustering with adaptive parameters
            sub_clusters = self._adaptive_geographic_clustering(enhanced_businesses)
            
            # 3. Boundary refinement using multiple algorithms
            refined_boundaries = []
            for j, sub_cluster in enumerate(sub_clusters):
                boundary = self._create_refined_boundary(sub_cluster)
                if boundary:
                    refined_boundaries.append({
                        'sub_cluster_id': j,
                        'boundary': boundary,
                        'business_count': len(sub_cluster),
                        'confidence': self._calculate_boundary_confidence(boundary, sub_cluster)
                    })
            
            # 4. Topology validation and correction
            validated_boundaries = self._validate_and_correct_topology(refined_boundaries)
            
            # 5. Context-aware boundary adjustment
            context_adjusted = self._apply_contextual_adjustments(
                validated_boundaries, cluster.get('type', 'mixed')
            )
            
            # 6. Boundary smoothing and optimization
            optimized_boundaries = self._optimize_boundaries(context_adjusted)
            
            enhanced_cluster = cluster.copy()
            enhanced_cluster['businesses'] = enhanced_businesses
            enhanced_cluster['boundaries'] = optimized_boundaries
            enhanced_cluster['spatial_enhancement'] = {
                'original_business_count': len(businesses),
                'enhanced_business_count': len(enhanced_businesses),
                'sub_clusters_created': len(sub_clusters),
                'boundaries_created': len(optimized_boundaries),
                'average_confidence': np.mean([b.get('confidence', 0.5) for b in optimized_boundaries])
            }
            
            enhanced_clusters.append(enhanced_cluster)
        
        return enhanced_clusters
    
    def _improve_coordinate_precision(self, businesses: List[Dict]) -> List[Dict]:
        """Improve coordinate precision through multiple methods"""
        
        enhanced = []
        
        for business in businesses:
            enhanced_business = business.copy()
            
            # 1. Reverse geocoding validation
            if 'lat' in business and 'lon' in business:
                lat, lon = business['lat'], business['lon']
                
                # Check if coordinates match address
                if 'address' in business:
                    validated_coords = self._validate_address_coordinates(
                        business['address'], lat, lon
                    )
                    if validated_coords:
                        enhanced_business['lat'] = validated_coords[0]
                        enhanced_business['lon'] = validated_coords[1]
                        enhanced_business['coordinate_source'] = 'validated'
                
                # 2. Snap to nearest road network for businesses
                if self._should_snap_to_road(business):
                    snapped_coords = self._snap_to_road_network(lat, lon)
                    if snapped_coords:
                        enhanced_business['lat'] = snapped_coords[0]
                        enhanced_business['lon'] = snapped_coords[1]
                        enhanced_business['coordinate_source'] = 'road_snapped'
                
                # 3. Precision enhancement
                enhanced_business = self._enhance_coordinate_precision(enhanced_business)
            
            enhanced.append(enhanced_business)
        
        return enhanced
    
    def _adaptive_geographic_clustering(self, businesses: List[Dict]) -> List[List[Dict]]:
        """Adaptive clustering based on business density and distribution"""
        
        # Extract coordinates
        coords = []
        valid_businesses = []
        
        for business in businesses:
            if 'lat' in business and 'lon' in business:
                lat, lon = float(business['lat']), float(business['lon'])
                
                # Validate coordinates are in reasonable range
                if (38.0 <= lat <= 40.0) and (-96.0 <= lon <= -94.0):
                    coords.append([lat, lon])
                    valid_businesses.append(business)
        
        if len(coords) < 3:
            return [valid_businesses]
        
        coords_array = np.array(coords)
        
        # Calculate adaptive epsilon based on point density
        distances = []
        for i in range(min(len(coords), 100)):  # Sample for performance
            point_distances = np.sqrt(np.sum((coords_array - coords_array[i])**2, axis=1))
            distances.extend(sorted(point_distances)[1:6])  # 5 nearest neighbors
        
        if distances:
            adaptive_eps = np.percentile(distances, 75) * 1.5
        else:
            adaptive_eps = 0.03  # Fallback
        
        # Apply DBSCAN with adaptive parameters
        clustering = DBSCAN(
            eps=adaptive_eps,
            min_samples=max(3, int(len(valid_businesses) * 0.02)),  # Adaptive min_samples
            algorithm='ball_tree',
            metric='haversine'  # Better for geographic data
        ).fit(np.radians(coords_array))  # Convert to radians for haversine
        
        # Group businesses by cluster
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label == -1:  # Noise points - assign to nearest cluster
                label = self._assign_to_nearest_cluster(idx, coords_array, clustering.labels_)
            
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(valid_businesses[idx])
        
        # Filter out very small clusters
        min_cluster_size = max(3, int(len(valid_businesses) * 0.05))
        filtered_clusters = [cluster for cluster in clusters.values() if len(cluster) >= min_cluster_size]
        
        return filtered_clusters if filtered_clusters else [valid_businesses]
    
    def _create_refined_boundary(self, businesses: List[Dict]) -> Optional[Dict]:
        """Create refined boundary using multiple algorithms"""
        
        coords = []
        for business in businesses:
            if 'lat' in business and 'lon' in business:
                coords.append([float(business['lon']), float(business['lat'])])
        
        if len(coords) < 3:
            return None
        
        try:
            # Method 1: Alpha shape for natural boundaries
            alpha_boundary = self._create_alpha_shape(coords)
            
            # Method 2: Concave hull for better fitting
            concave_boundary = self._create_concave_hull(coords)
            
            # Method 3: Kernel density estimation boundary
            kde_boundary = self._create_kde_boundary(coords)
            
            # Method 4: Weighted boundary based on business importance
            weighted_boundary = self._create_weighted_boundary(businesses, coords)
            
            # Combine methods using weighted approach
            combined_boundary = self._combine_boundaries([
                (alpha_boundary, 0.3),
                (concave_boundary, 0.3),
                (kde_boundary, 0.2),
                (weighted_boundary, 0.2)
            ])
            
            return {
                'geometry': combined_boundary,
                'method': 'refined',
                'confidence': self._calculate_boundary_confidence(combined_boundary, coords),
                'algorithm_scores': {
                    'alpha_shape': self._score_boundary(alpha_boundary, coords),
                    'concave_hull': self._score_boundary(concave_boundary, coords),
                    'kde_boundary': self._score_boundary(kde_boundary, coords),
                    'weighted_boundary': self._score_boundary(weighted_boundary, coords)
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating refined boundary: {e}")
            return self._fallback_convex_hull(coords)
    
    def _create_alpha_shape(self, coords: List[List[float]]) -> Optional[Polygon]:
        """Create alpha shape boundary for natural cluster boundaries"""
        try:
            from shapely.geometry import MultiPoint
            from shapely.ops import alpha_shape
            
            points = [Point(lon, lat) for lon, lat in coords]
            multipoint = MultiPoint(points)
            
            # Calculate optimal alpha value based on point distribution
            alpha = self._calculate_optimal_alpha(coords)
            
            # Create alpha shape
            boundary = alpha_shape(multipoint, alpha)
            
            if boundary and boundary.is_valid:
                return boundary
            
        except ImportError:
            logger.debug("Alpha shape not available, falling back to convex hull")
        except Exception as e:
            logger.debug(f"Alpha shape creation failed: {e}")
        
        return None
    
    def _create_concave_hull(self, coords: List[List[float]]) -> Optional[Polygon]:
        """Create concave hull boundary for better cluster fitting"""
        try:
            # Use Delaunay triangulation for concave hull
            from scipy.spatial import Delaunay
            
            points = np.array(coords)
            tri = Delaunay(points)
            
            # Find edges that are only used once (boundary edges)
            edges = set()
            edge_points = []
            
            for ia, ib, ic in tri.simplices:
                edges.add(tuple(sorted((ia, ib))))
                edges.add(tuple(sorted((ib, ic))))
                edges.add(tuple(sorted((ic, ia))))
            
            # Find boundary edges (used only once)
            boundary_edges = []
            for edge in edges:
                if list(edges).count(edge) == 1:
                    boundary_edges.append(edge)
            
            # Order boundary points
            if boundary_edges:
                boundary_points = self._order_boundary_points(boundary_edges, points)
                if len(boundary_points) >= 3:
                    return Polygon(boundary_points)
            
        except Exception as e:
            logger.debug(f"Concave hull creation failed: {e}")
        
        return None
    
    def _create_kde_boundary(self, coords: List[List[float]]) -> Optional[Polygon]:
        """Create boundary using kernel density estimation"""
        try:
            from scipy.stats import gaussian_kde
            from scipy.ndimage import gaussian_filter
            
            points = np.array(coords)
            
            # Create KDE
            kde = gaussian_kde(points.T)
            
            # Create grid
            xmin, ymin = points.min(axis=0)
            xmax, ymax = points.max(axis=0)
            
            # Add padding
            padding = 0.01
            xmin -= padding
            ymin -= padding
            xmax += padding
            ymax += padding
            
            # Generate grid
            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            grid_points = np.vstack([xx.ravel(), yy.ravel()])
            
            # Evaluate KDE on grid
            density = kde(grid_points).reshape(xx.shape)
            
            # Apply Gaussian filter for smoothing
            density_smooth = gaussian_filter(density, sigma=2)
            
            # Find contour at threshold
            threshold = np.percentile(density_smooth[density_smooth > 0], 20)
            
            from skimage import measure
            contours = measure.find_contours(density_smooth, threshold)
            
            if contours:
                # Take the largest contour
                largest_contour = max(contours, key=len)
                if len(largest_contour) >= 3:
                    return Polygon(largest_contour)
            
        except ImportError:
            logger.debug("KDE boundary not available, missing scipy/skimage")
        except Exception as e:
            logger.debug(f"KDE boundary creation failed: {e}")
        
        return None
    
    def _create_weighted_boundary(self, businesses: List[Dict], coords: List[List[float]]) -> Optional[Polygon]:
        """Create boundary weighted by business importance"""
        try:
            # Calculate weights based on business metrics
            weights = []
            for i, business in enumerate(businesses):
                if 'lat' in business and 'lon' in business:
                    # Weight by composite score and employee count
                    score = business.get('composite_score', 50)
                    employees = business.get('employees', 1)
                    
                    # Combined weight (normalized)
                    weight = (score / 100) * np.log1p(employees)
                    weights.append(weight)
                else:
                    weights.append(1.0)
            
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            
            # Create weighted convex hull
            points = np.array(coords)
            
            # Use weighted centroid as reference
            weighted_centroid = np.average(points, weights=weights, axis=0)
            
            # Find points farthest from centroid in different directions
            angles = np.arctan2(points[:, 1] - weighted_centroid[1], 
                               points[:, 0] - weighted_centroid[0])
            
            # Sort by angle and select boundary points
            sorted_indices = np.argsort(angles)
            
            # Select points that contribute most to boundary
            boundary_points = []
            for idx in sorted_indices:
                if len(boundary_points) == 0:
                    boundary_points.append(points[idx])
                else:
                    # Check if this point significantly expands the boundary
                    test_polygon = Polygon(boundary_points + [points[idx]])
                    if test_polygon.is_valid and test_polygon.area > 0:
                        boundary_points.append(points[idx])
            
            if len(boundary_points) >= 3:
                return Polygon(boundary_points)
            
        except Exception as e:
            logger.debug(f"Weighted boundary creation failed: {e}")
        
        return None
    
    def _combine_boundaries(self, boundaries: List[Tuple[Optional[Polygon], float]]) -> Polygon:
        """Combine multiple boundary methods using weighted approach"""
        
        valid_boundaries = [(geom, weight) for geom, weight in boundaries if geom is not None]
        
        if not valid_boundaries:
            # Fallback to convex hull
            return self._create_convex_hull_fallback()
        
        if len(valid_boundaries) == 1:
            return valid_boundaries[0][0]
        
        # Weighted combination using intersection and union
        total_weight = sum(weight for _, weight in valid_boundaries)
        
        # Start with the highest weight boundary
        primary_boundary, primary_weight = max(valid_boundaries, key=lambda x: x[1])
        combined = primary_boundary
        
        # Incorporate other boundaries
        for boundary, weight in valid_boundaries:
            if boundary is primary_boundary:
                continue
            
            # Weighted intersection (conservative approach)
            intersection_weight = weight / total_weight
            if intersection_weight > 0.3:  # Only intersect if significant weight
                try:
                    intersection = combined.intersection(boundary)
                    if intersection.is_valid and intersection.area > 0:
                        combined = intersection
                except:
                    pass
        
        return combined
    
    def _validate_and_correct_topology(self, boundaries: List[Dict]) -> List[Dict]:
        """Validate and correct topological issues in boundaries"""
        
        validated = []
        
        for boundary_data in boundaries:
            boundary = boundary_data.get('geometry')
            if not boundary:
                continue
            
            try:
                # Check for self-intersection
                if not boundary.is_valid:
                    # Try to fix self-intersection
                    boundary = boundary.buffer(0)
                
                # Check for very small area
                if boundary.area < 1e-8:
                    logger.warning(f"Boundary has very small area: {boundary.area}")
                    continue
                
                # Check for excessive complexity
                if hasattr(boundary, 'exterior'):
                    n_points = len(boundary.exterior.coords)
                    if n_points > self.max_points_for_simplification:
                        # Simplify boundary
                        boundary = boundary.simplify(0.001, preserve_topology=True)
                
                # Validate against KC county boundaries
                clipped_boundary = self._clip_to_kc_counties(boundary)
                
                if clipped_boundary and clipped_boundary.is_valid:
                    boundary_data['geometry'] = clipped_boundary
                    boundary_data['topology_valid'] = True
                    validated.append(boundary_data)
                else:
                    logger.warning("Boundary failed county clipping validation")
                    
            except Exception as e:
                logger.error(f"Topology validation failed: {e}")
                continue
        
        return validated
    
    def _apply_contextual_adjustments(self, boundaries: List[Dict], cluster_type: str) -> List[Dict]:
        """Apply context-aware boundary adjustments based on cluster type"""
        
        adjusted = []
        
        for boundary_data in boundaries:
            boundary = boundary_data.get('geometry')
            if not boundary:
                adjusted.append(boundary_data)
                continue
            
            try:
                # Apply cluster-type specific adjustments
                if cluster_type == 'logistics':
                    # Expand boundaries slightly for logistics (need more space)
                    boundary = boundary.buffer(self.default_buffer_distance * 0.5)
                
                elif cluster_type == 'manufacturing':
                    # Use tighter boundaries for manufacturing
                    boundary = boundary.buffer(-self.default_buffer_distance * 0.2)
                    if boundary.is_empty:
                        boundary = boundary_data['geometry']  # Fallback
                
                elif cluster_type == 'technology':
                    # Create more organic boundaries for tech clusters
                    boundary = self._create_organic_boundary(boundary)
                
                # Apply land-use considerations if available
                if self.land_use_data:
                    boundary = self._adjust_for_land_use(boundary)
                
                # Apply road network alignment if available
                if self.road_network:
                    boundary = self._align_to_road_network(boundary)
                
                boundary_data['geometry'] = boundary
                boundary_data['context_adjusted'] = True
                adjusted.append(boundary_data)
                
            except Exception as e:
                logger.debug(f"Contextual adjustment failed: {e}")
                adjusted.append(boundary_data)
        
        return adjusted
    
    def _optimize_boundaries(self, boundaries: List[Dict]) -> List[Dict]:
        """Final optimization of boundaries for performance and accuracy"""
        
        optimized = []
        
        for boundary_data in boundaries:
            boundary = boundary_data.get('geometry')
            if not boundary:
                optimized.append(boundary_data)
                continue
            
            try:
                # Final simplification for performance
                if hasattr(boundary, 'exterior'):
                    n_points = len(boundary.exterior.coords)
                    if n_points > 50:  # Simplify if too complex
                        boundary = boundary.simplify(0.0005, preserve_topology=True)
                
                # Ensure proper orientation (clockwise for exterior)
                if hasattr(boundary, 'exterior_coords'):
                    coords = list(boundary.exterior.coords)
                    if not self._is_clockwise(coords):
                        coords = coords[::-1]  # Reverse orientation
                        boundary = Polygon(coords)
                
                # Calculate final metrics
                boundary_data['geometry'] = boundary
                boundary_data['final_area'] = boundary.area
                boundary_data['final_perimeter'] = boundary.length
                boundary_data['optimization_applied'] = True
                
                optimized.append(boundary_data)
                
            except Exception as e:
                logger.debug(f"Boundary optimization failed: {e}")
                optimized.append(boundary_data)
        
        return optimized
    
    def _calculate_optimal_alpha(self, coords: List[List[float]]) -> float:
        """Calculate optimal alpha value for alpha shape based on point distribution"""
        n_points = len(coords)
        
        if n_points < 10:
            return 0.5  # More conservative for small clusters
        elif n_points < 50:
            return 0.3
        else:
            return 0.1  # More detailed for large clusters
    
    def _calculate_boundary_confidence(self, boundary: Polygon, coords: List[List[float]]) -> float:
        """Calculate confidence score for boundary quality"""
        if not boundary or not boundary.is_valid:
            return 0.0
        
        try:
            # Factors affecting confidence:
            # 1. Point coverage (how many points are inside/on boundary)
            points = [Point(lon, lat) for lon, lat in coords]
            inside_count = sum(1 for p in points if boundary.contains(p) or boundary.touches(p))
            coverage_score = inside_count / len(points)
            
            # 2. Boundary regularity (penalty for very irregular shapes)
            if hasattr(boundary, 'area') and hasattr(boundary, 'length'):
                compactness = 4 * math.pi * boundary.area / (boundary.length ** 2)
                regularity_score = min(1.0, compactness)
            else:
                regularity_score = 0.5
            
            # 3. Area appropriateness (not too small or too large)
            if hasattr(boundary, 'area'):
                # Expected area based on point distribution
                points_array = np.array(coords)
                expected_area = self._estimate_expected_area(points_array)
                area_ratio = min(1.0, boundary.area / expected_area)
                area_score = 1.0 - abs(1.0 - area_ratio)
            else:
                area_score = 0.5
            
            # Combine scores with weights
            confidence = (
                coverage_score * 0.5 +
                regularity_score * 0.3 +
                area_score * 0.2
            )
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.debug(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _estimate_expected_area(self, points: np.ndarray) -> float:
        """Estimate expected area based on point distribution"""
        if len(points) < 3:
            return 0.001
        
        # Use convex hull area as baseline
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume  # In 2D, volume is area
            
            # Adjust for point density
            density = len(points) / hull_area
            density_factor = min(1.0, density / 1000)  # Normalize density
            
            return hull_area * (1.0 + density_factor * 0.2)
            
        except:
            return 0.001  # Fallback
    
    def _is_clockwise(self, coords: List[Tuple[float, float]]) -> bool:
        """Check if polygon coordinates are in clockwise order"""
        if len(coords) < 4:
            return True
        
        # Calculate signed area
        area = 0
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            area += (x2 - x1) * (y2 + y1)
        
        return area > 0
    
    def _load_precise_county_boundaries(self) -> Dict:
        """Load precise county boundaries from enhanced data source"""
        try:
            from utils.kc_county_boundaries import KC_COUNTY_BOUNDARIES
            return KC_COUNTY_BOUNDARIES
        except ImportError:
            logger.warning("Could not load county boundaries")
            return {}
    
    def _clip_to_kc_counties(self, boundary: Polygon) -> Optional[Polygon]:
        """Clip boundary to KC metropolitan area"""
        if not self.kc_counties:
            return boundary
        
        try:
            # Create union of all KC counties
            county_polygons = []
            for county_name, boundary_data in self.kc_counties.items():
                if boundary_data['type'] == 'Polygon':
                    coords = boundary_data['coordinates'][0]
                    county_polygon = Polygon([(lon, lat) for lon, lat in coords])
                    county_polygons.append(county_polygon)
            
            if county_polygons:
                kc_union = unary_union(county_polygons)
                
                # Ensure proper intersection
                if boundary.intersects(kc_union):
                    clipped = boundary.intersection(kc_union)
                    
                    # Handle MultiPolygon results
                    if hasattr(clipped, 'geoms'):
                        # Return largest component
                        return max(clipped.geoms, key=lambda g: g.area)
                    
                    return clipped
            
        except Exception as e:
            logger.debug(f"County clipping failed: {e}")
        
        return boundary
    
    def _create_convex_hull_fallback(self) -> Polygon:
        """Create fallback convex hull boundary"""
        # Simple fallback - return a small polygon at KC center
        kc_center = (39.0997, -94.5786)  # Kansas City center
        center_point = Point(kc_center[1], kc_center[0])
        
        # Create small buffer around center
        return center_point.buffer(0.01)  # ~1km radius
    
    # Helper methods for coordinate improvement
    def _validate_address_coordinates(self, address: str, lat: float, lon: float) -> Optional[Tuple[float, float]]:
        """Validate coordinates against address using reverse geocoding"""
        # Placeholder for address validation
        # In implementation, this would use a geocoding service
        return None
    
    def _should_snap_to_road(self, business: Dict) -> bool:
        """Determine if business should be snapped to road network"""
        # Businesses that should be road-aligned
        road_aligned_types = ['logistics', 'transportation', 'manufacturing', 'retail']
        
        cluster_type = business.get('cluster_type', '').lower()
        return cluster_type in road_aligned_types
    
    def _snap_to_road_network(self, lat: float, lon: float) -> Optional[Tuple[float, float]]:
        """Snap coordinates to nearest road network"""
        # Placeholder for road network snapping
        # In implementation, this would use road network data
        return None
    
    def _enhance_coordinate_precision(self, business: Dict) -> Dict:
        """Enhance coordinate precision for a business"""
        # Round coordinates to appropriate precision
        if 'lat' in business and 'lon' in business:
            business['lat'] = round(float(business['lat']), 6)
            business['lon'] = round(float(business['lon']), 6)
        
        return business
    
    def _assign_to_nearest_cluster(self, point_idx: int, coords: np.ndarray, labels: np.ndarray) -> int:
        """Assign noise point to nearest cluster"""
        point = coords[point_idx]
        
        # Find cluster centers
        unique_labels = set(labels) - {-1}  # Exclude noise
        if not unique_labels:
            return 0
        
        min_distance = float('inf')
        nearest_label = 0
        
        for label in unique_labels:
            # Get points in this cluster
            cluster_points = coords[labels == label]
            if len(cluster_points) == 0:
                continue
            
            # Find nearest point in cluster
            distances = np.sqrt(np.sum((cluster_points - point) ** 2, axis=1))
            min_dist_to_cluster = distances.min()
            
            if min_dist_to_cluster < min_distance:
                min_distance = min_dist_to_cluster
                nearest_label = label
        
        return nearest_label
    
    def _order_boundary_points(self, edges: List[Tuple[int, int]], points: np.ndarray) -> List[Tuple[float, float]]:
        """Order boundary points to form a proper polygon"""
        # Simple ordering - start with a point and follow connected edges
        if not edges:
            return []
        
        # Start with first edge
        ordered_points = []
        used_edges = set()
        
        # Find starting point
        current_edge = edges[0]
        current_point = current_edge[0]
        
        ordered_points.append(tuple(points[current_point]))
        used_edges.add(tuple(sorted(current_edge)))
        
        # Follow edges around the boundary
        while len(used_edges) < len(edges):
            found_next = False
            
            for edge in edges:
                edge_tuple = tuple(sorted(edge))
                if edge_tuple in used_edges:
                    continue
                
                if current_point in edge:
                    # Find the other point in this edge
                    next_point = edge[1] if edge[0] == current_point else edge[0]
                    ordered_points.append(tuple(points[next_point]))
                    current_point = next_point
                    used_edges.add(edge_tuple)
                    found_next = True
                    break
            
            if not found_next:
                break
        
        return ordered_points
    
    def _score_boundary(self, boundary: Optional[Polygon], coords: List[List[float]]) -> float:
        """Score boundary quality for algorithm comparison"""
        if not boundary or not boundary.is_valid:
            return 0.0
        
        try:
            # Coverage score
            points = [Point(lon, lat) for lon, lat in coords]
            inside_count = sum(1 for p in points if boundary.contains(p) or boundary.touches(p))
            coverage = inside_count / len(points)
            
            # Compactness score
            if hasattr(boundary, 'area') and hasattr(boundary, 'length'):
                compactness = 4 * math.pi * boundary.area / (boundary.length ** 2)
            else:
                compactness = 0.5
            
            # Combined score
            return coverage * 0.7 + compactness * 0.3
            
        except:
            return 0.0
    
    def _create_organic_boundary(self, boundary: Polygon) -> Polygon:
        """Create more organic, less geometric boundary"""
        try:
            # Apply slight smoothing to create more organic shape
            smoothed = boundary.buffer(0.001).buffer(-0.001)
            
            if smoothed.is_valid:
                return smoothed
            else:
                return boundary
                
        except:
            return boundary
    
    def _adjust_for_land_use(self, boundary: Polygon) -> Polygon:
        """Adjust boundary based on land use considerations"""
        # Placeholder for land use adjustment
        # In implementation, this would consider zoning, land use patterns, etc.
        return boundary
    
    def _align_to_road_network(self, boundary: Polygon) -> Polygon:
        """Align boundary to road network where appropriate"""
        # Placeholder for road network alignment
        # In implementation, this would snap boundary segments to roads
        return boundary
