"""Interactive map features for enhanced cluster visualization"""

import folium
from folium import plugins
import json
import logging
from typing import Dict, List, Tuple, Optional
from utils.enhanced_color_scheme import EnhancedColorScheme

logger = logging.getLogger(__name__)

class InteractiveMapFeatures:
    """Enhanced interactive features for cluster maps"""
    
    def __init__(self):
        self.color_scheme = EnhancedColorScheme()
        self.active_cluster = None
        self.highlighted_boundary = None
        self.tooltip_data = {}
        
    def add_interactive_features(self, m: folium.Map, clusters: List[Dict]):
        """Add all interactive features to cluster map"""
        
        # 1. Enhanced hover tooltips with rich information
        self._add_enhanced_tooltips(m, clusters)
        
        # 2. Click-to-highlight functionality
        self._add_boundary_highlighting(m, clusters)
        
        # 3. Cluster comparison tool
        self._add_cluster_comparison(m, clusters)
        
        # 4. Dynamic filtering controls
        self._add_dynamic_filters(m, clusters)
        
        # 5. Mini-map overview
        self._add_minimap(m)
        
        # 6. Measurement tools
        self._add_measurement_tools(m)
    
    def _add_enhanced_tooltips(self, m: folium.Map, clusters: List[Dict]):
        """Add rich, informative tooltips with enhanced styling"""
        
        # Prepare tooltip data for all clusters
        for i, cluster in enumerate(clusters):
            cluster_id = f"cluster_{i}"
            
            # Calculate additional metrics for tooltip
            businesses = cluster.get('businesses', [])
            business_count = len(businesses)
            
            # Calculate average scores
            avg_score = 0
            total_employees = 0
            if businesses:
                scores = [b.get('composite_score', 0) for b in businesses if 'composite_score' in b]
                avg_score = sum(scores) / len(scores) if scores else 0
                total_employees = sum(b.get('employees', 0) for b in businesses if 'employees' in b)
            
            # Calculate economic impact metrics
            gdp_impact = cluster.get('projected_gdp_impact', 0)
            jobs_created = cluster.get('projected_jobs', 0)
            
            # Format numbers for display
            gdp_display = self._format_number(gdp_impact)
            jobs_display = f"{jobs_created:,}"
            
            # Get cluster color
            cluster_type = cluster.get('type', 'mixed')
            cluster_color = self.color_scheme.get_color_for_cluster(cluster_type)
            
            # Store tooltip data
            self.tooltip_data[cluster_id] = {
                'name': cluster.get('name', f'Cluster {i+1}'),
                'type': cluster_type,
                'color': cluster_color,
                'business_count': business_count,
                'avg_score': avg_score,
                'gdp_impact': gdp_display,
                'jobs_created': jobs_display,
                'total_employees': total_employees,
                'confidence': cluster.get('confidence_score', 0.5),
                'rank': cluster.get('rank', i+1)
            }
        
        # Create enhanced tooltip HTML and JavaScript
        tooltip_html = f"""
        <div id="enhanced-cluster-tooltip" style="
            position: fixed;
            background: linear-gradient(135deg, rgba(0, 0, 0, 0.95), rgba(0, 0, 0, 0.85));
            color: white;
            padding: 15px;
            border-radius: 12px;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 13px;
            z-index: 10000;
            max-width: 320px;
            display: none;
            box-shadow: 0 8px 25px rgba(0,0,0,0.4);
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            transform: translateY(-5px);
            transition: all 0.3s ease;
        ">
            <div id="tooltip-content"></div>
            <div style="
                position: absolute;
                bottom: 8px;
                right: 8px;
                font-size: 10px;
                opacity: 0.7;
            ">Click for details →</div>
        </div>
        
        <style>
        .tooltip-metric {{
            display: inline-block;
            margin: 4px 8px 4px 0;
            padding: 6px 10px;
            background: rgba(255,255,255,0.1);
            border-radius: 6px;
            text-align: center;
            min-width: 80px;
        }}
        
        .tooltip-metric-value {{
            display: block;
            font-size: 16px;
            font-weight: bold;
            margin-top: 2px;
        }}
        
        .tooltip-metric-label {{
            font-size: 10px;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .cluster-type-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }}
        
        .confidence-indicator {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-left: 5px;
        }}
        
        .confidence-high {{ background: #4CAF50; }}
        .confidence-medium {{ background: #FF9800; }}
        .confidence-low {{ background: #F44336; }}
        </style>
        
        <script>
        // Enhanced tooltip system with smooth animations
        const tooltip = document.getElementById('enhanced-cluster-tooltip');
        const tooltipContent = document.getElementById('tooltip-content');
        let hideTimeout = null;
        
        // Tooltip data from server
        const tooltipData = {json.dumps(self.tooltip_data)};
        
        function showEnhancedTooltip(clusterId, event) {{
            // Clear any pending hide timeout
            if (hideTimeout) {{
                clearTimeout(hideTimeout);
                hideTimeout = null;
            }}
            
            const data = tooltipData[clusterId];
            if (!data) return;
            
            // Create rich tooltip content
            const confidenceClass = data.confidence >= 0.8 ? 'confidence-high' : 
                                   data.confidence >= 0.6 ? 'confidence-medium' : 'confidence-low';
            
            tooltipContent.innerHTML = `
                <div style="margin-bottom: 12px;">
                    <strong style="font-size: 16px; color: #FFD700; display: block; margin-bottom: 5px;">
                        ${{data.name}}
                    </strong>
                    <span class="cluster-type-badge" style="background: ${{data.color}}; color: white;">
                        ${{data.type.replace('_', ' ').toUpperCase()}}
                    </span>
                    <span style="float: right; font-size: 12px; opacity: 0.8;">
                        Rank #${{data.rank}}
                        <span class="confidence-indicator ${{confidenceClass}}"></span>
                    </span>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px;">
                    <div class="tooltip-metric">
                        <div class="tooltip-metric-label">Businesses</div>
                        <div class="tooltip-metric-value">${{data.business_count}}</div>
                    </div>
                    <div class="tooltip-metric">
                        <div class="tooltip-metric-label">Avg Score</div>
                        <div class="tooltip-metric-value">${{data.avg_score.toFixed(1)}}</div>
                    </div>
                </div>
                
                <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 8px; margin-bottom: 10px;">
                    <div style="font-size: 12px; margin-bottom: 6px; opacity: 0.9; font-weight: bold;">ECONOMIC IMPACT</div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                        <div class="tooltip-metric">
                            <div class="tooltip-metric-label">GDP Impact</div>
                            <div class="tooltip-metric-value">$${{data.gdp_impact}}</div>
                        </div>
                        <div class="tooltip-metric">
                            <div class="tooltip-metric-label">Jobs Created</div>
                            <div class="tooltip-metric-value">${{data.jobs_created}}</div>
                        </div>
                    </div>
                </div>
                
                <div style="font-size: 11px; opacity: 0.7; text-align: center; margin-top: 8px;">
                    <i class="fa fa-info-circle"></i> Click cluster for detailed analysis
                </div>
            `;
            
            // Show tooltip with animation
            tooltip.style.display = 'block';
            tooltip.style.opacity = '0';
            tooltip.style.transform = 'translateY(-5px) scale(0.95)';
            
            // Position tooltip near cursor but keep on screen
            const x = Math.min(event.pageX + 20, window.innerWidth - 350);
            const y = Math.min(event.pageY + 20, window.innerHeight - 250);
            
            tooltip.style.left = x + 'px';
            tooltip.style.top = y + 'px';
            
            // Animate in
            setTimeout(() => {{
                tooltip.style.opacity = '1';
                tooltip.style.transform = 'translateY(0) scale(1)';
            }}, 10);
        }}
        
        function hideTooltipWithDelay() {{
            // Add delay to prevent flickering when moving between elements
            if (hideTimeout) clearTimeout(hideTimeout);
            hideTimeout = setTimeout(() => {{
                tooltip.style.opacity = '0';
                tooltip.style.transform = 'translateY(-5px) scale(0.95)';
                setTimeout(() => {{
                    tooltip.style.display = 'none';
                }}, 200);
            }}, 100);
        }}
        
        function hideTooltipImmediately() {{
            if (hideTimeout) {{
                clearTimeout(hideTimeout);
                hideTimeout = null;
            }}
            tooltip.style.display = 'none';
        }}
        
        // Add event listeners when DOM is ready
        document.addEventListener('DOMContentLoaded', function() {{
            const clusterElements = document.querySelectorAll('[data-cluster-id]');
            
            clusterElements.forEach(element => {{
                const clusterId = element.getAttribute('data-cluster-id');
                
                element.addEventListener('mouseenter', (e) => showEnhancedTooltip(clusterId, e));
                element.addEventListener('mouseleave', hideTooltipWithDelay);
                element.addEventListener('click', () => {{
                    hideTooltipImmediately();
                    highlightCluster(clusterId);
                }});
            }});
        }});
        </script>
        """
        
        m.get_root().html.add_child(folium.Element(tooltip_html))
    
    def _add_boundary_highlighting(self, m: folium.Map, clusters: List[Dict]):
        """Add click-to-highlight functionality for cluster boundaries"""
        
        highlight_script = f"""
        <style>
        .cluster-boundary {{
            transition: all 0.3s ease;
            cursor: pointer;
        }}
        
        .cluster-boundary:hover {{
            fill-opacity: 0.3 !important;
            stroke-width: 4px !important;
            filter: brightness(1.1);
        }}
        
        .cluster-boundary.highlighted {{
            fill-opacity: 0.5 !important;
            stroke-width: 5px !important;
            stroke: #FFD700 !important;
            filter: brightness(1.2) drop-shadow(0 0 10px rgba(255, 215, 0, 0.5));
            z-index: 1000 !important;
        }}
        
        .cluster-marker {{
            transition: all 0.3s ease;
        }}
        
        .cluster-marker.highlighted {{
            transform: scale(1.2);
            filter: brightness(1.3) drop-shadow(0 0 15px rgba(255, 215, 0, 0.7));
            z-index: 1001 !important;
        }}
        
        .cluster-details-panel {{
            position: fixed;
            top: 50%;
            right: -400px;
            transform: translateY(-50%);
            width: 380px;
            max-height: 80vh;
            background: white;
            box-shadow: -5px 0 20px rgba(0,0,0,0.3);
            border-radius: 8px 0 0 8px;
            z-index: 10000;
            overflow-y: auto;
            transition: right 0.4s ease;
        }}
        
        .cluster-details-panel.show {{
            right: 20px;
        }}
        
        .details-panel-header {{
            background: linear-gradient(135deg, #2196F3, #1976D2);
            color: white;
            padding: 15px;
            border-radius: 8px 8px 0 0;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        
        .details-panel-close {{
            position: absolute;
            top: 15px;
            right: 15px;
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .details-panel-close:hover {{
            background: rgba(255,255,255,0.3);
        }}
        </style>
        
        <script>
        let highlightedClusterId = null;
        let originalStyles = {{}};
        const tooltipData = {json.dumps(self.tooltip_data)};
        
        function highlightCluster(clusterId) {{
            // Reset previous highlight
            if (highlightedClusterId && highlightedClusterId !== clusterId) {{
                resetHighlight(highlightedClusterId);
            }}
            
            // Find and highlight new cluster elements
            const clusterElements = document.querySelectorAll(`[data-cluster-id="${{clusterId}}"]`);
            
            clusterElements.forEach(element => {{
                // Store original styles if not already stored
                if (!originalStyles[clusterId]) {{
                    originalStyles[clusterId] = {{
                        fillOpacity: element.style.fillOpacity || '',
                        strokeWidth: element.style.strokeWidth || '',
                        stroke: element.style.stroke || '',
                        filter: element.style.filter || ''
                    }};
                }}
                
                // Apply highlight styles
                element.classList.add('highlighted');
            }});
            
            highlightedClusterId = clusterId;
            
            // Show cluster details panel
            showClusterDetailsPanel(clusterId);
            
            // Update URL hash for bookmarking
            window.location.hash = `cluster=${{clusterId}}`;
        }}
        
        function resetHighlight(clusterId) {{
            const clusterElements = document.querySelectorAll(`[data-cluster-id="${{clusterId}}"]`);
            
            clusterElements.forEach(element => {{
                element.classList.remove('highlighted');
                
                // Restore original styles
                if (originalStyles[clusterId]) {{
                    Object.assign(element.style, originalStyles[clusterId]);
                }}
            }});
        }}
        
        function showClusterDetailsPanel(clusterId) {{
            // Create or update details panel
            let panel = document.getElementById('cluster-details-panel');
            
            if (!panel) {{
                panel = document.createElement('div');
                panel.id = 'cluster-details-panel';
                document.body.appendChild(panel);
            }}
            
            const data = tooltipData[clusterId];
            if (!data) return;
            
            // Generate detailed panel content
            panel.innerHTML = `
                <div class="details-panel-header">
                    <h3 style="margin: 0; font-size: 18px;">${{data.name}}</h3>
                    <button class="details-panel-close" onclick="hideClusterDetailsPanel()">×</button>
                    <div style="margin-top: 8px; font-size: 14px; opacity: 0.9;">
                        <span class="cluster-type-badge" style="background: ${{data.color}};">
                            ${{data.type.replace('_', ' ').toUpperCase()}}
                        </span>
                        <span style="margin-left: 10px;">Rank #${{data.rank}}</span>
                    </div>
                </div>
                
                <div style="padding: 20px;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">
                        <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 12px; color: #666; margin-bottom: 5px;">BUSINESSES</div>
                            <div style="font-size: 24px; font-weight: bold; color: #2196F3;">${{data.business_count}}</div>
                        </div>
                        <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 12px; color: #666; margin-bottom: 5px;">AVG SCORE</div>
                            <div style="font-size: 24px; font-weight: bold; color: #4CAF50;">${{data.avg_score.toFixed(1)}}</div>
                        </div>
                    </div>
                    
                    <div style="background: linear-gradient(135deg, #E3F2FD, #BBDEFB); padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                        <h4 style="margin: 0 0 10px 0; color: #1565C0;">ECONOMIC IMPACT</h4>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                            <div>
                                <div style="font-size: 12px; color: #666;">GDP Impact</div>
                                <div style="font-size: 20px; font-weight: bold; color: #1565C0;">${{data.gdp_impact}}</div>
                            </div>
                            <div>
                                <div style="font-size: 12px; color: #666;">Jobs Created</div>
                                <div style="font-size: 20px; font-weight: bold; color: #1565C0;">${{data.jobs_created}}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div style="text-align: center; margin-top: 20px;">
                        <button onclick="exportClusterData('${{clusterId}}')" style="
                            background: #2196F3;
                            color: white;
                            border: none;
                            padding: 10px 20px;
                            border-radius: 5px;
                            cursor: pointer;
                            font-size: 14px;
                            margin-right: 10px;
                        ">
                            <i class="fa fa-download"></i> Export Data
                        </button>
                        <button onclick="shareCluster('${{clusterId}}')" style="
                            background: #4CAF50;
                            color: white;
                            border: none;
                            padding: 10px 20px;
                            border-radius: 5px;
                            cursor: pointer;
                            font-size: 14px;
                        ">
                            <i class="fa fa-share"></i> Share
                        </button>
                    </div>
                </div>
            `;
            
            // Show panel with animation
            setTimeout(() => panel.classList.add('show'), 10);
        }}
        
        function hideClusterDetailsPanel() {{
            const panel = document.getElementById('cluster-details-panel');
            if (panel) {{
                panel.classList.remove('show');
            }}
            
            // Reset highlight
            if (highlightedClusterId) {{
                resetHighlight(highlightedClusterId);
                highlightedClusterId = null;
            }}
            
            // Clear URL hash
            if (window.location.hash) {{
                history.pushState("", document.title, window.location.pathname);
            }}
        }}
        
        function exportClusterData(clusterId) {{
            // Implementation for cluster data export
            console.log('Exporting data for cluster:', clusterId);
            // This would trigger download of cluster data as CSV/JSON
        }}
        
        function shareCluster(clusterId) {{
            // Implementation for cluster sharing
            console.log('Sharing cluster:', clusterId);
            // This would generate shareable link or social media post
        }}
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') {{
                hideClusterDetailsPanel();
            }}
        }});
        
        // Check URL hash on load for direct linking
        document.addEventListener('DOMContentLoaded', function() {{
            if (window.location.hash) {{
                const match = window.location.hash.match(/cluster=([^&]+)/);
                if (match && match[1]) {{
                    setTimeout(() => highlightCluster(match[1]), 500);
                }}
            }}
        }});
        </script>
        """
        
        m.get_root().html.add_child(folium.Element(highlight_script))
    
    def _add_cluster_comparison(self, m: folium.Map, clusters: List[Dict]):
        """Add cluster comparison functionality"""
        
        comparison_html = f"""
        <div id="cluster-comparison-panel" style="
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            z-index: 9999;
            max-width: 300px;
            display: none;
        ">
            <h4 style="margin: 0 0 10px 0; font-size: 16px;">Compare Clusters</h4>
            <div id="comparison-clusters">
                <!-- Comparison items will be added dynamically -->
            </div>
            <div style="margin-top: 10px; text-align: center;">
                <button onclick="clearComparison()" style="
                    background: #f44336;
                    color: white;
                    border: none;
                    padding: 8px 15px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 12px;
                ">Clear All</button>
            </div>
        </div>
        
        <div id="comparison-toggle" style="
            position: fixed;
            bottom: 20px;
            left: 20px;
            background: #2196F3;
            color: white;
            padding: 10px 15px;
            border-radius: 8px 8px 8px 0;
            cursor: pointer;
            z-index: 9998;
            font-size: 14px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        " onclick="toggleComparisonPanel()">
            <i class="fa fa-balance-scale"></i> Compare
        </div>
        
        <script>
        let comparisonClusters = [];
        const tooltipData = {json.dumps(self.tooltip_data)};
        
        function toggleComparisonPanel() {{
            const panel = document.getElementById('cluster-comparison-panel');
            const toggle = document.getElementById('comparison-toggle');
            
            if (panel.style.display === 'block') {{
                panel.style.display = 'none';
                toggle.style.borderRadius = '8px';
            }} else {{
                panel.style.display = 'block';
                toggle.style.borderRadius = '8px 8px 0 0';
            }}
        }}
        
        function addToComparison(clusterId) {{
            if (comparisonClusters.includes(clusterId)) return;
            
            if (comparisonClusters.length >= 3) {{
                alert('Maximum 3 clusters can be compared at once');
                return;
            }}
            
            comparisonClusters.push(clusterId);
            updateComparisonDisplay();
        }}
        
        function removeFromComparison(clusterId) {{
            comparisonClusters = comparisonClusters.filter(id => id !== clusterId);
            updateComparisonDisplay();
        }}
        
        function updateComparisonDisplay() {{
            const container = document.getElementById('comparison-clusters');
            container.innerHTML = '';
            
            comparisonClusters.forEach(clusterId => {{
                const data = tooltipData[clusterId];
                if (!data) return;
                
                const item = document.createElement('div');
                item.style.cssText = `
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: 8px;
                    margin-bottom: 5px;
                    background: #f5f5f5;
                    border-radius: 4px;
                    border-left: 4px solid ${{data.color}};
                `;
                
                item.innerHTML = `
                    <span style="font-weight: bold;">${{data.name}}</span>
                    <button onclick="removeFromComparison('${{clusterId}}')" style="
                        background: #f44336;
                        color: white;
                        border: none;
                        padding: 2px 8px;
                        border-radius: 3px;
                        cursor: pointer;
                        font-size: 11px;
                    ">×</button>
                `;
                
                container.appendChild(item);
            }});
            
            // Show comparison chart if 2+ clusters
            if (comparisonClusters.length >= 2) {{
                showComparisonChart();
            }} else {{
                hideComparisonChart();
            }}
        }}
        
        function clearComparison() {{
            comparisonClusters = [];
            updateComparisonDisplay();
        }}
        
        function showComparisonChart() {{
            // Implementation for side-by-side comparison chart
            console.log('Showing comparison for:', comparisonClusters);
        }}
        
        function hideComparisonChart() {{
            // Hide comparison chart
        }}
        
        // Add comparison buttons to cluster tooltips
        document.addEventListener('DOMContentLoaded', function() {{
            // This would be integrated with the tooltip system
        }});
        </script>
        """
        
        m.get_root().html.add_child(folium.Element(comparison_html))
    
    def _add_dynamic_filters(self, m: folium.Map, clusters: List[Dict]):
        """Add dynamic filtering controls for clusters"""
        
        # Get unique cluster types
        cluster_types = list(set(cluster.get('type', 'mixed') for cluster in clusters))
        
        filter_html = f"""
        <div id="cluster-filter-panel" style="
            position: fixed;
            top: 80px;
            right: 20px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            z-index: 9999;
            width: 250px;
        ">
            <h4 style="margin: 0 0 15px 0; font-size: 16px;">Filter Clusters</h4>
            
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: bold;">By Type:</label>
                <div id="cluster-type-filters">
                    {self._generate_type_checkboxes(cluster_types)}
                </div>
            </div>
            
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: bold;">By Size:</label>
                <select id="cluster-size-filter" style="
                    width: 100%;
                    padding: 8px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                ">
                    <option value="all">All Sizes</option>
                    <option value="small">Small (< 50 businesses)</option>
                    <option value="medium">Medium (50-200 businesses)</option>
                    <option value="large">Large (> 200 businesses)</option>
                </select>
            </div>
            
            <div style="margin-bottom: 15px;">
                <label style="display: block; margin-bottom: 5px; font-weight: bold;">By Impact:</label>
                <input type="range" id="cluster-impact-filter" min="0" max="100" value="0" style="
                    width: 100%;
                ">
                <div style="display: flex; justify-content: space-between; font-size: 11px; color: #666;">
                    <span>Low</span>
                    <span id="impact-value">All</span>
                    <span>High</span>
                </div>
            </div>
            
            <div style="text-align: center;">
                <button onclick="applyFilters()" style="
                    background: #2196F3;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 14px;
                    width: 100%;
                ">Apply Filters</button>
            </div>
        </div>
        
        <script>
        const tooltipData = {json.dumps(self.tooltip_data)};
        
        function updateImpactValue(value) {{
            const impactLabel = document.getElementById('impact-value');
            if (value == 0) {{
                impactLabel.textContent = 'All';
            }} else {{
                impactLabel.textContent = value + '+';
            }}
        }}
        
        function applyFilters() {{
            // Get selected cluster types
            const selectedTypes = [];
            document.querySelectorAll('#cluster-type-filters input:checked').forEach(checkbox => {{
                selectedTypes.push(checkbox.value);
            }});
            
            // Get selected size
            const selectedSize = document.getElementById('cluster-size-filter').value;
            
            // Get impact threshold
            const impactThreshold = document.getElementById('cluster-impact-filter').value;
            
            // Filter clusters
            Object.keys(tooltipData).forEach(clusterId => {{
                const data = tooltipData[clusterId];
                const element = document.querySelector(`[data-cluster-id="${{clusterId}}"]`);
                
                if (!element) return;
                
                let show = true;
                
                // Check type filter
                if (selectedTypes.length > 0 && !selectedTypes.includes(data.type)) {{
                    show = false;
                }}
                
                // Check size filter
                if (selectedSize !== 'all') {{
                    if (selectedSize === 'small' && data.business_count >= 50) show = false;
                    if (selectedSize === 'medium' && (data.business_count < 50 || data.business_count > 200)) show = false;
                    if (selectedSize === 'large' && data.business_count <= 200) show = false;
                }}
                
                // Check impact filter
                if (impactThreshold > 0) {{
                    const impactScore = (data.avg_score / 100) * 100; // Normalize to 0-100
                    if (impactScore < impactThreshold) show = false;
                }}
                
                // Apply filter
                element.style.display = show ? '' : 'none';
            }});
        }}
        
        // Update impact value display
        document.getElementById('cluster-impact-filter').addEventListener('input', (e) => {{
            updateImpactValue(e.target.value);
        }});
        </script>
        """
        
        m.get_root().html.add_child(folium.Element(filter_html))
    
    def _generate_type_checkboxes(self, cluster_types: List[str]) -> str:
        """Generate checkbox HTML for cluster type filters"""
        checkboxes = []
        
        for cluster_type in sorted(cluster_types):
            color = self.color_scheme.get_color_for_cluster(cluster_type)
            display_name = cluster_type.replace('_', ' ').title()
            
            checkbox = f"""
            <label style="
                display: flex;
                align-items: center;
                margin-bottom: 8px;
                cursor: pointer;
            ">
                <input type="checkbox" value="{cluster_type}" checked style="margin-right: 8px;">
                <span style="
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    background: {color};
                    border-radius: 2px;
                    margin-right: 8px;
                "></span>
                {display_name}
            </label>
            """
            checkboxes.append(checkbox)
        
        return ''.join(checkboxes)
    
    def _add_minimap(self, m: folium.Map):
        """Add minimap overview"""
        minimap_html = """
        <div id="minimap-container" style="
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 200px;
            height: 150px;
            border: 2px solid #2196F3;
            border-radius: 8px;
            overflow: hidden;
            z-index: 9999;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        ">
            <div id="minimap" style="
                width: 100%;
                height: 100%;
                background: #f0f0f0;
            "></div>
        </div>
        
        <script>
        // Minimap implementation would sync with main map
        // This is a placeholder for the minimap functionality
        </script>
        """
        
        m.get_root().html.add_child(folium.Element(minimap_html))
    
    def _add_measurement_tools(self, m: folium.Map):
        """Add measurement tools to the map"""
        measurement_html = """
        <div id="measurement-tools" style="
            position: fixed;
            top: 80px;
            left: 20px;
            background: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            z-index: 9999;
        ">
            <button onclick="enableDistanceMeasurement()" style="
                background: #4CAF50;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                cursor: pointer;
                margin-right: 5px;
                font-size: 12px;
            " title="Measure Distance">
                <i class="fa fa-ruler"></i>
            </button>
            <button onclick="enableAreaMeasurement()" style="
                background: #FF9800;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                cursor: pointer;
                margin-right: 5px;
                font-size: 12px;
            " title="Measure Area">
                <i class="fa fa-draw-polygon"></i>
            </button>
            <button onclick="clearMeasurements()" style="
                background: #f44336;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
            " title="Clear Measurements">
                <i class="fa fa-trash"></i>
            </button>
        </div>
        
        <script>
        // Measurement tools implementation
        function enableDistanceMeasurement() {
            console.log('Distance measurement enabled');
            // Implementation for distance measurement
        }
        
        function enableAreaMeasurement() {
            console.log('Area measurement enabled');
            // Implementation for area measurement
        }
        
        function clearMeasurements() {
            console.log('Measurements cleared');
            // Implementation for clearing measurements
        }
        </script>
        """
        
        m.get_root().html.add_child(folium.Element(measurement_html))
    
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