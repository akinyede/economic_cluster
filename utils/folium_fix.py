"""Fix for Folium CDN issues with Content Security Policy"""

import folium
from folium import branca

def patch_folium_cdns():
    """Patch Folium to use CDNs that are allowed by our CSP"""
    
    # Override the default CDN paths in branca (used by Folium)
    # Update Bootstrap CSS to use cdn.jsdelivr.net instead of netdna.bootstrapcdn.com
    if hasattr(branca.element, 'CssLink'):
        # Replace old Bootstrap CDN with new one
        old_bootstrap = 'https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap.min.css'
        new_bootstrap = 'https://cdn.jsdelivr.net/npm/bootstrap@3.0.0/dist/css/bootstrap.min.css'
        
        # Patch the default CSS links
        original_render = branca.element.CssLink.render
        
        def patched_render(self, **kwargs):
            # Replace old CDN with new one
            if hasattr(self, 'url') and self.url == old_bootstrap:
                self.url = new_bootstrap
            return original_render(self, **kwargs)
        
        branca.element.CssLink.render = patched_render
    
    # Also patch Folium's default tile layers to ensure HTTPS
    if hasattr(folium, 'TileLayer'):
        original_init = folium.TileLayer.__init__
        
        def patched_init(self, tiles='OpenStreetMap', *args, **kwargs):
            # Ensure HTTPS for OpenStreetMap tiles
            if tiles == 'OpenStreetMap':
                tiles = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
                kwargs['attr'] = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            # Handle CartoDB tiles
            elif tiles == 'CartoDB Positron' or tiles == 'cartodbpositron':
                tiles = 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'
                kwargs['attr'] = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
            original_init(self, tiles, *args, **kwargs)
        
        folium.TileLayer.__init__ = patched_init

# Apply the patch when this module is imported
patch_folium_cdns()