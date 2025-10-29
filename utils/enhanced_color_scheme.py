"""Enhanced color scheme with perceptual uniformity for cluster differentiation"""

import colorsys
import logging
from typing import Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class EnhancedColorScheme:
    """Perceptually uniform color system optimized for cluster visualization"""
    
    def __init__(self):
        # Primary colors optimized for color vision deficiency and contrast
        self.cluster_colors = {
            "logistics": "#0066CC",      # Strong Blue
            "biosciences": "#00A652",    # Deep Green  
            "technology": "#7B3F99",      # Purple
            "manufacturing": "#E87722",    # Orange
            "animal_health": "#D62D20",    # Red
            "professional_services": "#5D4037",  # Brown
            "community_services": "#00796B",  # Teal
            "mixed": "#616161",            # Gray
            # Additional types for better differentiation
            "finance": "#0288D1",          # Cyan
            "healthcare": "#C2185B",        # Pink
            "education": "#FFA000",          # Amber
            "retail": "#7CB342",            # Light Green
            "transportation": "#303F9F",     # Indigo
            "agriculture": "#689F38",        # Light Green
            "construction": "#F57C00",       # Deep Orange
            "utilities": "#01579B",          # Dark Blue
        }
        
        # Generate sub-cluster colors for each cluster type
        self.sub_cluster_colors = self._generate_sub_cluster_palette()
        
        # Color accessibility settings
        self.contrast_threshold = 4.5  # WCAG AA standard
        self.color_blind_friendly = True
    
    def _generate_sub_cluster_palette(self) -> Dict[str, List[str]]:
        """Generate perceptually distinct colors for sub-clusters"""
        sub_colors = {}
        
        for cluster_type, base_color in self.cluster_colors.items():
            # Generate variations using HSL color space for consistent perception
            variations = self._create_color_variations(base_color, 5)
            sub_colors[cluster_type] = variations
        
        return sub_colors
    
    def _create_color_variations(self, base_hex: str, n_variations: int = 5) -> List[str]:
        """Create perceptually distinct variations of base color"""
        # Convert hex to RGB
        r, g, b = tuple(int(base_hex[i:i+2], 16) for i in (1, 3, 5))
        
        # Convert to HSL
        h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
        
        variations = []
        for i in range(n_variations):
            # Vary lightness and saturation while keeping hue
            # Use golden ratio for optimal distribution
            golden_ratio = 0.618033988749895
            
            # Calculate variation parameters
            lightness_factor = 1.0 + (i - n_variations/2) * 0.15
            saturation_factor = 1.0 + (i - n_variations/2) * 0.1
            
            # Apply factors with bounds checking
            new_l = max(0.2, min(0.8, l * lightness_factor))
            new_s = max(0.3, min(1.0, s * saturation_factor))
            
            # Convert back to RGB
            new_r, new_g, new_b = colorsys.hls_to_rgb(h, new_l, new_s)
            
            # Convert to hex
            new_hex = "#{:02x}{:02x}{:02x}".format(
                int(new_r * 255), int(new_g * 255), int(new_b * 255)
            )
            variations.append(new_hex)
        
        return variations
    
    def get_contrasting_colors(self, n_colors: int) -> List[str]:
        """Generate n maximally contrasting colors using color theory"""
        colors = []
        
        # Use golden ratio for hue distribution
        golden_ratio = 0.618033988749895
        
        for i in range(n_colors):
            # Distribute hues using golden ratio
            hue = (i * golden_ratio) % 1.0
            
            # Use consistent saturation and lightness for good contrast
            saturation = 0.7
            lightness = 0.5
            
            # Convert HSL to RGB
            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            
            # Convert to hex
            hex_color = "#{:02x}{:02x}{:02x}".format(
                int(r * 255), int(g * 255), int(b * 255)
            )
            colors.append(hex_color)
        
        return colors
    
    def get_color_for_cluster(self, cluster_type: str, sub_cluster_id: int = 0) -> str:
        """Get color for a cluster type and optional sub-cluster"""
        # Get base color for cluster type
        base_color = self.cluster_colors.get(cluster_type, "#616161")
        
        # If sub-cluster specified, get variation
        if sub_cluster_id > 0 and cluster_type in self.sub_cluster_colors:
            variations = self.sub_cluster_colors[cluster_type]
            if sub_cluster_id <= len(variations):
                return variations[sub_cluster_id - 1]
        
        return base_color
    
    def calculate_contrast_ratio(self, color1: str, color2: str) -> float:
        """Calculate contrast ratio between two colors for accessibility"""
        # Convert hex to RGB
        r1, g1, b1 = tuple(int(color1[i:i+2], 16) for i in (1, 3, 5))
        r2, g2, b2 = tuple(int(color2[i:i+2], 16) for i in (1, 3, 5))
        
        # Calculate relative luminance
        l1 = self._relative_luminance(r1, g1, b1)
        l2 = self._relative_luminance(r2, g2, b2)
        
        # Calculate contrast ratio
        if l1 > l2:
            return (l1 + 0.05) / (l2 + 0.05)
        else:
            return (l2 + 0.05) / (l1 + 0.05)
    
    def _relative_luminance(self, r: int, g: int, b: int) -> float:
        """Calculate relative luminance for contrast calculation"""
        # Normalize RGB values
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0
        
        # Apply gamma correction
        def gamma_correct(c):
            if c <= 0.03928:
                return c / 12.92
            else:
                return pow((c + 0.055) / 1.055, 2.4)
        
        r_corrected = gamma_correct(r_norm)
        g_corrected = gamma_correct(g_norm)
        b_corrected = gamma_correct(b_norm)
        
        # Calculate luminance
        return 0.2126 * r_corrected + 0.7152 * g_corrected + 0.0722 * b_corrected
    
    def get_text_color(self, background_color: str) -> str:
        """Get appropriate text color (black or white) for background"""
        # Calculate contrast with white and black
        contrast_with_white = self.calculate_contrast_ratio(background_color, "#FFFFFF")
        contrast_with_black = self.calculate_contrast_ratio(background_color, "#000000")
        
        # Return the color with better contrast
        if contrast_with_white > contrast_with_black:
            return "#FFFFFF"
        else:
            return "#000000"
    
    def create_gradient_colors(self, base_color: str, n_steps: int = 5) -> List[str]:
        """Create gradient colors from base color"""
        # Convert hex to RGB
        r, g, b = tuple(int(base_color[i:i+2], 16) for i in (1, 3, 5))
        
        # Convert to HSL
        h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
        
        gradient_colors = []
        for i in range(n_steps):
            # Vary lightness from light to dark
            lightness_factor = 0.9 - (i * 0.15)  # From 0.9 to 0.15
            new_l = max(0.15, min(0.9, l * lightness_factor))
            
            # Keep hue and saturation constant
            new_r, new_g, new_b = colorsys.hls_to_rgb(h, new_l, s)
            
            # Convert to hex
            new_hex = "#{:02x}{:02x}{:02x}".format(
                int(new_r * 255), int(new_g * 255), int(new_b * 255)
            )
            gradient_colors.append(new_hex)
        
        return gradient_colors
    
    def validate_color_scheme(self) -> Dict:
        """Validate color scheme for accessibility and contrast"""
        validation_result = {
            'valid': True,
            'issues': [],
            'contrast_ratios': {},
            'color_blind_simulation': {}
        }
        
        # Check contrast between adjacent colors in palette
        color_list = list(self.cluster_colors.values())
        
        for i in range(len(color_list) - 1):
            color1 = color_list[i]
            color2 = color_list[i + 1]
            
            contrast = self.calculate_contrast_ratio(color1, color2)
            validation_result['contrast_ratios'][f"{i}-{i+1}"] = contrast
            
            if contrast < self.contrast_threshold:
                validation_result['issues'].append(
                    f"Low contrast between colors {i} and {i+1}: {contrast:.2f}"
                )
                validation_result['valid'] = False
        
        # Check text contrast for each color
        for color_name, color_value in self.cluster_colors.items():
            text_color = self.get_text_color(color_value)
            text_contrast = self.calculate_contrast_ratio(color_value, text_color)
            
            if text_contrast < self.contrast_threshold:
                validation_result['issues'].append(
                    f"Poor text contrast for {color_name}: {text_contrast:.2f}"
                )
                validation_result['valid'] = False
        
        return validation_result
    
    def get_color_legend_html(self) -> str:
        """Generate HTML for color legend with accessibility features"""
        legend_items = []
        
        for cluster_type, color in self.cluster_colors.items():
            text_color = self.get_text_color(color)
            
            legend_item = f"""
            <div class="legend-item" style="
                display: flex;
                align-items: center;
                margin-bottom: 5px;
                padding: 3px;
                border-radius: 3px;
                background-color: {color}20;
            ">
                <div class="color-swatch" style="
                    width: 16px;
                    height: 16px;
                    background-color: {color};
                    border: 1px solid #ccc;
                    margin-right: 8px;
                    border-radius: 2px;
                "></div>
                <div class="cluster-label" style="
                    color: {text_color};
                    font-weight: bold;
                    font-size: 12px;
                ">{cluster_type.replace('_', ' ').title()}</div>
            </div>
            """
            legend_items.append(legend_item)
        
        return f"""
        <div class="enhanced-color-legend" style="
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            border: 1px solid #ddd;
        ">
            <h4 style="margin: 0 0 10px 0; font-size: 14px;">Cluster Types</h4>
            {''.join(legend_items)}
        </div>
        """
    
    def generate_legend_html(self, cluster_types: List[str]) -> str:
        """Generate HTML legend for cluster types"""
        legend_items = []
        for cluster_type in sorted(cluster_types):
            color = self.get_color_for_cluster(cluster_type)
            display_name = cluster_type.replace('_', ' ').title()
            legend_items.append(f"""
                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <div style="width: 20px; height: 12px; background: {color}; margin-right: 8px; border-radius: 2px;"></div>
                    <span style="font-size: 12px;">{display_name}</span>
                </div>
            """)
        
        return f"""
        <div style="
            background: white;
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            font-family: Arial, sans-serif;
        ">
            <div style="font-weight: bold; margin-bottom: 8px; font-size: 14px;">Cluster Types</div>
            {''.join(legend_items)}
        </div>
        """
    
    def export_color_scheme(self) -> Dict:
        """Export color scheme for use in other components"""
        return {
            'cluster_colors': self.cluster_colors,
            'sub_cluster_colors': self.sub_cluster_colors,
            'contrast_threshold': self.contrast_threshold,
            'color_blind_friendly': self.color_blind_friendly,
            'validation': self.validate_color_scheme()
        }