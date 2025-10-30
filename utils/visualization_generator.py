"""Visualization generator for KC Cluster Prediction Tool"""
import json
import plotly.graph_objects as go
import plotly.utils
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import logging
import numpy as np
import pandas as pd
import math
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.inter_cluster_synergy import InterClusterSynergyAnalyzer

logger = logging.getLogger(__name__)

class VisualizationGenerator:
    """Generate Plotly visualizations for cluster analysis results"""
    
    def __init__(self):
        self.color_scheme = {
            'primary': '#003366',
            'secondary': '#4ECDC4',
            'success': '#2ECC71',
            'warning': '#F39C12',
            'danger': '#E74C3C',
            'info': '#3498DB'
        }
    
    def generate_all_visualizations(self, results: Dict) -> Dict[str, str]:
        """Generate all visualizations for the analysis results
        
        Returns:
            Dict mapping visualization names to JSON-encoded Plotly figures
        """
        visualizations = {}
        
        try:
            # Generate basic visualizations
            visualizations['gdp_chart'] = self._create_gdp_chart(results)
            visualizations['jobs_chart'] = self._create_jobs_chart(results)
            visualizations['spider_chart'] = self._create_spider_chart(results)
            visualizations['business_pie'] = self._create_business_pie(results)
            visualizations['gauge_chart'] = self._create_gauge_chart(results)
            
            # Generate ML-specific visualizations
            visualizations['ml_predictions_chart'] = self._create_ml_predictions_chart(results)
            visualizations['confidence_chart'] = self._create_confidence_chart(results)
            visualizations['network_graph'] = self._create_network_graph(results)
            visualizations['feature_importance_chart'] = self._create_feature_importance_chart(results)
            visualizations['synergy_heatmap'] = self._create_synergy_heatmap(results)
            visualizations['prediction_intervals_chart'] = self._create_prediction_intervals_chart(results)
            
            # Generate geographic visualizations
            try:
                from utils.map_generator import MapGenerator
                map_gen = MapGenerator()
                
                # Respect user's visualization mode selection if provided
                user_viz_mode = results.get('parameters', {}).get('visualization_mode', 'improved')
                if user_viz_mode:
                    map_gen.visualization_mode = user_viz_mode
                    logger.info(f"Using user-selected visualization mode: {user_viz_mode}")
                
                visualizations['cluster_map'] = map_gen.create_cluster_map(results)
                visualizations['heat_map'] = map_gen.create_heat_map(results)  # Now Growth Potential Map
                visualizations['economic_impact_map'] = map_gen.create_economic_impact_map(results)
                logger.info("Generated geographic visualizations successfully")
            except Exception as e:
                logger.error(f"Error generating geographic visualizations: {e}")
                visualizations['cluster_map'] = self._create_empty_map("Cluster Map")
                visualizations['heat_map'] = self._create_empty_map("Growth Potential Map")
                visualizations['economic_impact_map'] = self._create_empty_map("Economic Impact Map")
            
            logger.info(f"Generated {len(visualizations)} visualizations successfully")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            # Return empty visualizations rather than failing
            visualizations = {
                'gdp_chart': self._create_empty_chart("GDP Impact Chart"),
                'jobs_chart': self._create_empty_chart("Jobs Creation Chart"),
                'spider_chart': self._create_empty_chart("Industry Spider Chart"),
                'business_pie': self._create_empty_chart("Business Distribution"),
                'gauge_chart': self._create_empty_chart("Target Achievement Gauge"),
                'ml_predictions_chart': self._create_empty_chart("ML Predictions Comparison"),
                'confidence_chart': self._create_empty_chart("Prediction Confidence"),
                'network_graph': self._create_empty_chart("Business Network"),
                'feature_importance_chart': self._create_empty_chart("Feature Importance"),
                'synergy_heatmap': self._create_empty_chart("Cluster Synergies"),
                'prediction_intervals_chart': self._create_empty_chart("Prediction Intervals")
            }
        
        return visualizations
    
    def _create_gdp_chart(self, results: Dict) -> str:
        """Create GDP impact bar chart"""
        try:
            impact = results.get('economic_impact', {})
            
            # Extract GDP data with validation
            projected_gdp = impact.get('projected_gdp_impact', 0)
            if not isinstance(projected_gdp, (int, float)) or math.isnan(projected_gdp):
                projected_gdp = 0
            # Get target from results or economic_targets
            economic_targets = results.get('economic_targets', {})
            target_gdp = economic_targets.get('gdp_growth', impact.get('gdp_target', 2.87e9))
            
            fig = go.Figure()
            
            # Add bars
            fig.add_trace(go.Bar(
                x=['Projected GDP Impact', 'Target GDP Impact'],
                y=[projected_gdp, target_gdp],
                marker_color=[self.color_scheme['primary'], self.color_scheme['secondary']],
                text=[f'${projected_gdp:,.0f}', f'${target_gdp:,.0f}'],
                textposition='outside'
            ))
            
            fig.update_layout(
                title='GDP Impact Analysis',
                xaxis_title='Category',
                yaxis_title='GDP Impact ($)',
                yaxis=dict(tickformat='$,.0f'),
                showlegend=False,
                height=400
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating GDP chart: {e}")
            return self._create_empty_chart("GDP Impact Chart")
    
    def _create_jobs_chart(self, results: Dict) -> str:
        """Create jobs creation chart"""
        try:
            impact = results.get('economic_impact', {})
            
            # Extract jobs data with validation
            direct_jobs = impact.get('projected_direct_jobs', 0)
            total_jobs = impact.get('projected_total_jobs', 0)
            
            # Validate values
            if not isinstance(direct_jobs, (int, float)) or math.isnan(direct_jobs):
                direct_jobs = 0
            if not isinstance(total_jobs, (int, float)) or math.isnan(total_jobs):
                total_jobs = direct_jobs
            
            indirect_jobs = max(0, total_jobs - direct_jobs)  # Ensure non-negative
            # Get target from results or economic_targets
            economic_targets = results.get('economic_targets', {})
            target_jobs = economic_targets.get('direct_jobs', impact.get('jobs_target', 1000))
            
            fig = go.Figure()
            
            # Stacked bar chart
            fig.add_trace(go.Bar(
                name='Direct Jobs',
                x=['Projected Jobs', 'Target Jobs'],
                y=[direct_jobs, target_jobs],
                marker_color=self.color_scheme['primary']
            ))
            
            fig.add_trace(go.Bar(
                name='Indirect Jobs',
                x=['Projected Jobs', 'Target Jobs'],
                y=[indirect_jobs, 0],
                marker_color=self.color_scheme['secondary']
            ))
            
            fig.update_layout(
                title='Jobs Creation Analysis',
                xaxis_title='Category',
                yaxis_title='Number of Jobs',
                barmode='stack',
                height=400,
                showlegend=True
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating jobs chart: {e}")
            return self._create_empty_chart("Jobs Creation Chart")
    
    def _create_spider_chart(self, results: Dict) -> str:
        """Create spider/radar chart for cluster performance"""
        try:
            # Extract cluster data (try both locations for compatibility)
            clusters = results.get('clusters', [])
            if not clusters and 'steps' in results:
                clusters = results.get('steps', {}).get('cluster_optimization', {}).get('clusters', [])
            
            if not clusters:
                return self._create_empty_chart("Industry Performance Spider Chart")
            
            # Use first 5 clusters for visualization
            top_clusters = clusters[:5]
            
            categories = ['GDP Impact', 'Job Creation', 'Innovation', 'Market Access', 'Strategic Score']
            
            fig = go.Figure()
            
            for cluster in top_clusters:
                # Look for scores in multiple possible locations
                # First check if there's a nested 'scores' object
                scores = cluster.get('scores', {})
                
                # Extract values from various possible locations
                # GDP Impact
                gdp_val = cluster.get('projected_gdp_impact', 0)
                if gdp_val == 0 and 'metrics' in cluster:
                    gdp_val = cluster['metrics'].get('projected_gdp_impact', 0)
                
                # Job Creation
                job_val = cluster.get('projected_jobs', 0)
                if job_val == 0 and 'metrics' in cluster:
                    job_val = cluster['metrics'].get('projected_jobs', 0)
                
                # Innovation Score
                innov_val = cluster.get('innovation_score', 0)
                if innov_val == 0:
                    innov_val = cluster.get('innovation_capacity_score', 0)
                if innov_val == 0 and 'metrics' in cluster:
                    innov_val = cluster['metrics'].get('innovation_score', 0)
                
                # Market Access Score
                market_val = cluster.get('market_access_score', 0)
                if market_val == 0:
                    market_val = cluster.get('synergy_score', 0)
                if market_val == 0:
                    market_val = cluster.get('synergy_potential', 0)
                
                # Strategic/Cluster Score
                strategic_val = cluster.get('cluster_score', 0)
                if strategic_val == 0:
                    strategic_val = cluster.get('total_score', 0)
                if strategic_val == 0:
                    strategic_val = cluster.get('strategic_score', 0)
                
                # Validate and normalize values
                gdp_val = gdp_val if isinstance(gdp_val, (int, float)) and not math.isnan(gdp_val) else 0
                job_val = job_val if isinstance(job_val, (int, float)) and not math.isnan(job_val) else 0
                innov_val = innov_val if isinstance(innov_val, (int, float)) and not math.isnan(innov_val) else 0
                market_val = market_val if isinstance(market_val, (int, float)) and not math.isnan(market_val) else 0
                strategic_val = strategic_val if isinstance(strategic_val, (int, float)) and not math.isnan(strategic_val) else 0
                
                # Normalize values to 0-100 scale for radar chart
                # GDP: Scale to millions, then to 0-100 (assuming max 5B)
                gdp_normalized = min(100, (gdp_val / 5e9) * 100) if gdp_val > 0 else 0
                
                # Jobs: Scale to 0-100 (assuming max 10k jobs)
                jobs_normalized = min(100, (job_val / 10000) * 100) if job_val > 0 else 0
                
                # Innovation: Already on 0-100 scale or needs scaling
                innov_normalized = innov_val if innov_val <= 100 else min(100, innov_val / 10)
                
                # Market/Synergy: Already on 0-100 scale
                market_normalized = market_val if market_val <= 100 else min(100, market_val / 10)
                
                # Strategic: Already on 0-100 scale
                strategic_normalized = strategic_val if strategic_val <= 100 else min(100, strategic_val / 10)
                
                values = [
                    gdp_normalized,
                    jobs_normalized,
                    innov_normalized,
                    market_normalized,
                    strategic_normalized
                ]
                
                # Only add trace if we have meaningful data
                if any(v > 0 for v in values):
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=cluster.get('name', 'Cluster')[:30]  # Truncate long names
                    ))
            
            # Check if we added any traces
            if not fig.data:
                return self._create_empty_chart("Industry Performance Spider Chart - No data to display")
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100],
                        ticksuffix='%'
                    )),
                showlegend=True,
                title="Cluster Performance Analysis (Normalized Scores)",
                height=500
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating spider chart: {e}")
            return self._create_empty_chart("Industry Performance Spider Chart")
    
    def _create_business_pie(self, results: Dict) -> str:
        """Create pie chart for business distribution"""
        try:
            # Extract business distribution data
            steps = results.get('steps', {})
            scoring = steps.get('business_scoring', {})
            
            # Get business counts by score range from actual data
            score_distribution = scoring.get('score_distribution', {})
            
            # If no distribution provided, calculate from businesses
            if not score_distribution:
                total_businesses = scoring.get('total_businesses', 0)
                if total_businesses > 0:
                    # Use realistic distribution percentages
                    score_distribution = {
                        'elite': int(total_businesses * 0.05),
                        'high': int(total_businesses * 0.15),
                        'medium': int(total_businesses * 0.30),
                        'low': int(total_businesses * 0.50)
                    }
                else:
                    # No data available
                    score_distribution = {
                        'elite': 0,
                        'high': 0,
                        'medium': 0,
                        'low': 0
                    }
            
            labels = ['Elite (90-100)', 'High (70-89)', 'Medium (50-69)', 'Low (0-49)']
            values = [
                score_distribution.get('elite', 0),
                score_distribution.get('high', 0),
                score_distribution.get('medium', 0),
                score_distribution.get('low', 0)
            ]
            colors = [
                self.color_scheme['success'],
                self.color_scheme['info'],
                self.color_scheme['warning'],
                self.color_scheme['danger']
            ]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.3,
                marker_colors=colors
            )])
            
            fig.update_layout(
                title='Business Score Distribution',
                height=400,
                showlegend=True
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating business pie chart: {e}")
            return self._create_empty_chart("Business Distribution")
    
    def _create_gauge_chart(self, results: Dict) -> str:
        """Create gauge chart for target achievement"""
        try:
            impact = results.get('economic_impact', {})
            
            # Calculate overall achievement percentage with validation
            gdp_achievement = impact.get('gdp_target_achievement', 0)
            if not isinstance(gdp_achievement, (int, float)) or math.isnan(gdp_achievement):
                gdp_achievement = 0
            
            # Get jobs achievement directly from results (already calculated correctly in main.py)
            jobs_achievement = impact.get('jobs_target_achievement', 0)
            if not isinstance(jobs_achievement, (int, float)) or math.isnan(jobs_achievement):
                jobs_achievement = 0
            
            # Calculate overall achievement as average of GDP and jobs achievement
            overall_achievement = min(200, max(0, (gdp_achievement + jobs_achievement) / 2))  # Cap at 200%
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=overall_achievement,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Target Achievement (%)"},
                delta={'reference': 100, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [None, 200], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': self._get_gauge_color(overall_achievement)},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50], 'color': '#ffcccc'},
                        {'range': [50, 100], 'color': '#ffffcc'},
                        {'range': [100, 150], 'color': '#ccffcc'},
                        {'range': [150, 200], 'color': '#99ff99'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=400)
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating gauge chart: {e}")
            return self._create_empty_chart("Target Achievement Gauge")
    
    def _get_gauge_color(self, value: float) -> str:
        """Get color for gauge based on value"""
        if value >= 75:
            return self.color_scheme['success']
        elif value >= 50:
            return self.color_scheme['warning']
        else:
            return self.color_scheme['danger']
    
    def _create_empty_chart(self, title: str) -> str:
        """Create an empty chart with a message"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=f"{title}<br>No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        
        fig.update_layout(
            xaxis={'visible': False},
            yaxis={'visible': False},
            height=400
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def _create_ml_predictions_chart(self, results: Dict) -> str:
        """Create comparison chart of strategic vs ML predictions"""
        try:
            clusters = results.get('clusters', [])
            if not clusters and 'steps' in results:
                clusters = results.get('steps', {}).get('cluster_optimization', {}).get('clusters', [])
            clusters = clusters[:5]  # Top 5 clusters
            
            if not clusters:
                return self._create_empty_chart("ML Predictions Comparison")
            
            cluster_names = []
            strategic_gdp = []
            ml_gdp = []
            strategic_jobs = []
            ml_jobs = []
            
            for cluster in clusters:
                cluster_names.append(cluster.get('name', 'Unknown')[:30])
                
                # Get strategic predictions (handle both formats)
                if 'metrics' in cluster:
                    # Strategic cluster format
                    strategic_gdp.append(cluster['metrics'].get('projected_gdp_impact', 0))
                    strategic_jobs.append(cluster['metrics'].get('projected_jobs', 0))
                else:
                    # Optimization cluster format
                    strategic_gdp.append(cluster.get('projected_gdp_impact', 0))
                    strategic_jobs.append(cluster.get('projected_jobs', 0))
                
                # Get ML predictions
                ml_preds = cluster.get('ml_predictions', {})
                ml_gdp.append(ml_preds.get('gdp_impact', 0))
                ml_jobs.append(ml_preds.get('job_creation', 0))
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('GDP Impact Predictions', 'Job Creation Predictions')
            )
            
            # GDP comparison
            fig.add_trace(
                go.Bar(name='Strategic', x=cluster_names, y=strategic_gdp,
                       marker_color=self.color_scheme['primary']),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(name='ML Enhanced', x=cluster_names, y=ml_gdp,
                       marker_color=self.color_scheme['secondary']),
                row=1, col=1
            )
            
            # Jobs comparison
            fig.add_trace(
                go.Bar(name='Strategic', x=cluster_names, y=strategic_jobs,
                       marker_color=self.color_scheme['primary'], showlegend=False),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(name='ML Enhanced', x=cluster_names, y=ml_jobs,
                       marker_color=self.color_scheme['secondary'], showlegend=False),
                row=1, col=2
            )
            
            fig.update_xaxes(tickangle=-45)
            fig.update_layout(
                title='Strategic vs ML-Enhanced Predictions',
                barmode='group',
                height=500,
                showlegend=True
            )
            fig.update_yaxes(title_text="GDP Impact ($)", row=1, col=1)
            fig.update_yaxes(title_text="Number of Jobs", row=1, col=2)
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating ML predictions chart: {e}")
            return self._create_empty_chart("ML Predictions Comparison")
    
    def _create_confidence_chart(self, results: Dict) -> str:
        """Create confidence score visualization for ML predictions"""
        try:
            clusters = results.get('clusters', [])
            if not clusters and 'steps' in results:
                clusters = results.get('steps', {}).get('cluster_optimization', {}).get('clusters', [])
            
            if not clusters:
                return self._create_empty_chart("Prediction Confidence")
            
            cluster_names = []
            confidence_scores = []
            colors = []
            
            for cluster in clusters[:10]:  # Top 10 clusters
                cluster_names.append(cluster.get('name', 'Unknown')[:30])
                confidence = cluster.get('confidence_score', 0) * 100
                confidence_scores.append(confidence)
                
                # Color based on confidence level
                if confidence >= 80:
                    colors.append(self.color_scheme['success'])
                elif confidence >= 60:
                    colors.append(self.color_scheme['warning'])
                else:
                    colors.append(self.color_scheme['danger'])
            
            fig = go.Figure(data=[
                go.Bar(
                    x=cluster_names,
                    y=confidence_scores,
                    marker_color=colors,
                    text=[f'{c:.0f}%' for c in confidence_scores],
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title='ML Prediction Confidence Scores by Cluster',
                xaxis_title='Cluster',
                yaxis_title='Confidence Score (%)',
                yaxis=dict(range=[0, 105]),
                height=400
            )
            fig.update_xaxes(tickangle=-45)
            
            # Add reference lines
            fig.add_hline(y=80, line_dash="dash", line_color="green", 
                         annotation_text="High Confidence")
            fig.add_hline(y=60, line_dash="dash", line_color="orange", 
                         annotation_text="Medium Confidence")
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating confidence chart: {e}")
            return self._create_empty_chart("Prediction Confidence")
    
    def _create_network_graph(self, results: Dict) -> str:
        """Create network visualization showing business relationships"""
        try:
            # Get top cluster with network metrics
            clusters = results.get('clusters', [])
            if not clusters and 'steps' in results:
                clusters = results.get('steps', {}).get('cluster_optimization', {}).get('clusters', [])
            cluster_with_network = None
            
            for cluster in clusters:
                if cluster.get('network_metrics'):
                    cluster_with_network = cluster
                    break
            
            if not cluster_with_network:
                return self._create_empty_chart("Business Network Analysis")
            
            network = cluster_with_network.get('network_metrics', {})
            
            # Create a simple network visualization using scatter plot
            # In a real implementation, you'd use actual network data
            n_nodes = min(cluster_with_network.get('business_count', 20), 20)
            
            # Generate node positions using circular layout
            node_trace = go.Scatter(
                x=[],
                y=[],
                mode='markers+text',
                hoverinfo='text',
                marker=dict(
                    size=[],
                    color=[],
                    colorscale='YlOrRd',
                    line_width=2
                ),
                text=[],
                textposition="top center"
            )
            
            # Get business names from the cluster
            businesses = cluster_with_network.get('businesses', [])
            business_names = []
            
            # Extract business names
            for business in businesses[:n_nodes]:
                if isinstance(business, dict):
                    name = business.get('name', business.get('company_name', 'Unknown Business'))
                    # Truncate long names
                    name = str(name)[:30] if pd.notna(name) else 'Unknown Business'
                    business_names.append(name)
                else:
                    business_names.append('Unknown Business')
            
            # If we don't have enough business names, use central businesses
            central = network.get('central_businesses', [])
            if not business_names and central:
                business_names = central[:n_nodes]
            
            # Fill remaining with generic names if needed
            while len(business_names) < n_nodes:
                business_names.append(f'Business {len(business_names) + 1}')
            
            # Define business types and colors based on cluster type
            cluster_type = cluster_with_network.get('type', 'manufacturing')
            
            if cluster_type == 'manufacturing':
                business_types = ['Heavy Manufacturing', 'Light Manufacturing', 'Assembly/Packaging', 'Industrial Services', 'Supply Chain']
                colors_map = {
                    'Heavy Manufacturing': '#d62728',  # Red
                    'Light Manufacturing': '#ff7f0e',  # Orange  
                    'Assembly/Packaging': '#2ca02c',   # Green
                    'Industrial Services': '#1f77b4',  # Blue
                    'Supply Chain': '#9467bd'          # Purple
                }
            elif cluster_type == 'logistics':
                business_types = ['Transportation', 'Warehousing', 'Distribution', 'Freight Services', 'Supply Chain']
                colors_map = {
                    'Transportation': '#1f77b4',       # Blue
                    'Warehousing': '#ff7f0e',         # Orange
                    'Distribution': '#2ca02c',         # Green
                    'Freight Services': '#d62728',     # Red
                    'Supply Chain': '#9467bd'          # Purple
                }
            else:
                business_types = ['Professional Services', 'Tech Services', 'Support Services', 'Consulting', 'Other']
                colors_map = {
                    'Professional Services': '#1f77b4', # Blue
                    'Tech Services': '#2ca02c',        # Green
                    'Support Services': '#ff7f0e',     # Orange
                    'Consulting': '#9467bd',           # Purple
                    'Other': '#d62728'                 # Red
                }
            
            # Create nodes in circular layout
            for i in range(n_nodes):
                angle = 2 * math.pi * i / n_nodes
                x = math.cos(angle)
                y = math.sin(angle)
                node_trace['x'] += tuple([x])
                node_trace['y'] += tuple([y])
                
                # Assign business type
                business_type = business_types[i % len(business_types)]
                color = colors_map[business_type]
                
                # Size based on position (central businesses are larger)
                if i < len(central) and business_names[i] in central:
                    size = 35 + np.random.randint(10, 15)
                else:
                    size = 20 + np.random.randint(5, 15)
                node_trace['marker']['size'] += tuple([size])
                node_trace['marker']['color'] += tuple([color])
                
                # Add business name with type
                node_trace['text'] += tuple([business_names[i]])
            
            # Create edges (simplified)
            edge_trace = []
            density = network.get('network_density', 0.3)
            n_edges = int(n_nodes * (n_nodes - 1) * density / 2)
            
            for _ in range(n_edges):
                i = np.random.randint(0, n_nodes)
                j = np.random.randint(0, n_nodes)
                if i != j:
                    angle_i = 2 * math.pi * i / n_nodes
                    angle_j = 2 * math.pi * j / n_nodes
                    edge_trace.append(go.Scatter(
                        x=[math.cos(angle_i), math.cos(angle_j)],
                        y=[math.sin(angle_i), math.sin(angle_j)],
                        mode='lines',
                        line=dict(width=0.5, color='#888'),
                        hoverinfo='none',
                        showlegend=False
                    ))
            
            fig = go.Figure(data=edge_trace + [node_trace])
            
            # Add legend traces for business types
            for business_type, color in colors_map.items():
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=15, color=color),
                    showlegend=True,
                    name=business_type
                ))
            
            # Add metrics as annotations
            metrics_text = f"""<b>Network Metrics:</b><br>
            Network Density: {network.get('network_density', 0):.2f}<br>
            Clustering: {network.get('avg_clustering', 0):.2f}<br>
            Synergy Score: {network.get('synergy_score', 0):.0f}/100<br>
            Resilience: {network.get('network_resilience', 0):.0f}/100
            """
            
            fig.add_annotation(
                x=1.3, y=1,
                text=metrics_text,
                showarrow=False,
                align='left',
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            )
            
            # Add size legend annotation
            fig.add_annotation(
                x=1.3, y=-0.8,
                text="<b>Node Size:</b><br>Larger = Central/Hub Business<br>Smaller = Standard Business",
                showarrow=False,
                align='left',
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='black',
                borderwidth=1
            )
            
            fig.update_layout(
                title=f'Business Network: {cluster_with_network.get("name", "Cluster")}',
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=200, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=500,
                legend=dict(
                    title="Business Types",
                    yanchor="top",
                    y=0.5,
                    xanchor="left",
                    x=1.05,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating network graph: {e}")
            return self._create_empty_chart("Business Network Analysis")
    
    def _create_feature_importance_chart(self, results: Dict) -> str:
        """Create feature importance visualization from ML explanations"""
        try:
            ml_explanations = results.get('ml_explanations', {})
            
            if not ml_explanations:
                # No ML explanations available - return empty chart
                return self._create_empty_chart("Feature Importance - No ML data available")
            else:
                # Aggregate across clusters (average absolute importance)
                agg: Dict[str, float] = {}
                count: Dict[str, int] = {}
                for expl in ml_explanations.values():
                    fi = expl.get('feature_importance', {}) or {}
                    for k, v in fi.items():
                        try:
                            val = float(abs(v))
                        except Exception:
                            val = 0.0
                        agg[k] = agg.get(k, 0.0) + val
                        count[k] = count.get(k, 0) + 1
                if not agg:
                    return self._create_empty_chart("Feature Importance - No ML data available")
                # Average and pick top 20
                for k in list(agg.keys()):
                    c = max(1, count.get(k, 1))
                    agg[k] = agg[k] / c
                top_pairs = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:20]
                features = [k for k, _ in top_pairs]
                importance = [v for _, v in top_pairs]
            
            # Sort by importance
            sorted_pairs = sorted(zip(features, importance), key=lambda x: abs(x[1]), reverse=True)
            features, importance = zip(*sorted_pairs)
            
            colors = [self.color_scheme['success'] if imp > 0 else self.color_scheme['danger'] 
                     for imp in importance]
            
            fig = go.Figure(data=[
                go.Bar(
                    y=features,
                    x=importance,
                    orientation='h',
                    marker_color=colors,
                    text=[f'{abs(imp):.3f}' for imp in importance],
                    textposition='outside'
                )
            ])
            
            subtitle = ""
            fig.update_layout(
                title='ML model Feature importance',
                xaxis_title='Impact on Predictions (normalized, SHAP/permutation-inspired)',
                yaxis_title='Features',
                height=400,
                margin=dict(l=150)
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating feature importance chart: {e}")
            return self._create_empty_chart("Feature Importance")
    
    def _create_synergy_heatmap(self, results: Dict) -> str:
        """Create heatmap showing synergies between clusters using best practices"""
        try:
            clusters = results.get('clusters', [])
            if not clusters and 'steps' in results:
                clusters = results.get('steps', {}).get('cluster_optimization', {}).get('clusters', [])
            clusters = clusters[:8]  # Top 8 clusters
            
            if len(clusters) < 2:
                return self._create_empty_chart("Cluster Synergies")
            
            # Initialize the synergy analyzer
            synergy_analyzer = InterClusterSynergyAnalyzer()
            
            # Ensure we use cluster names, not types
            cluster_names = []
            for c in clusters:
                name = c.get('name', '')
                # Fallback to type if name is empty or looks like a type
                if not name or name in ['mixed', 'logistics', 'technology', 'manufacturing', 'biosciences', 'mixed_services']:
                    name = f"{c.get('type', 'Unknown').replace('_', ' ').title()} Cluster {clusters.index(c) + 1}"
                cluster_names.append(name[:20])
            
            # Create synergy matrix using comprehensive analysis
            synergy_matrix = synergy_analyzer.create_synergy_matrix(clusters)
            z_values = synergy_matrix.tolist()
            text_values = np.round(synergy_matrix, 0).astype(int).tolist()
            
            # Store detailed synergy information for tooltips
            hover_text = []
            for i in range(len(clusters)):
                row_text = []
                for j in range(len(clusters)):
                    if i == j:
                        row_text.append("Self: 100")
                    else:
                        synergy_result = synergy_analyzer.calculate_comprehensive_synergy(
                            clusters[i], clusters[j]
                        )
                        components = synergy_result['components']
                        text = f"Total: {synergy_result['total_score']:.0f}<br>"
                        text += f"Critical Mass: {components.get('critical_mass', 0):.0f}<br>"
                        text += f"Supply Chain: {components.get('supply_chain', 0):.0f}<br>"
                        text += f"Geographic: {components.get('geographic', 0):.0f}<br>"
                        text += f"Workforce: {components.get('workforce', 0):.0f}<br>"
                        text += f"Innovation: {components.get('innovation', 0):.0f}"
                        row_text.append(text)
                hover_text.append(row_text)
            
            # Create color scale with better thresholds
            # 70+ = Dark Green (Strong), 40-70 = Yellow/Light Green (Moderate), <40 = Red (Weak)
            fig = go.Figure(data=go.Heatmap(
                x=cluster_names,
                y=cluster_names,
                colorscale=[
                    [0, '#d32f2f'],      # Red for 0-20
                    [0.2, '#f57c00'],    # Orange for 20-40
                    [0.4, '#fbc02d'],    # Yellow for 40-60
                    [0.6, '#689f38'],    # Light Green for 60-70
                    [0.7, '#388e3c'],    # Green for 70-80
                    [1, '#1b5e20']       # Dark Green for 80-100
                ],
                z=z_values,
                text=text_values,
                texttemplate='%{text}',
                textfont={"size": 10},
                hovertext=hover_text,
                hovertemplate='%{hovertext}<extra></extra>',
                colorbar=dict(
                    title="Synergy Score",
                    tickmode='array',
                    tickvals=[0, 20, 40, 60, 70, 80, 100],
                    ticktext=['0', '20', '40', '60', '70', '80', '100']
                )
            ))
            
            fig.update_layout(
                title={
                    'text': 'Inter-Cluster Synergy Analysis<br><sub>Based on Critical Mass, Supply Chains, Geography, Workforce & Innovation</sub>',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title='Cluster',
                yaxis_title='Cluster',
                height=550,
                font=dict(size=11)
            )
            fig.update_xaxes(tickangle=-45)
            
            # Add annotations for synergy levels
            fig.add_annotation(
                text="<b>Synergy Levels:</b> 70+ Strong | 40-70 Moderate | <40 Weak",
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=10),
                xanchor='center'
            )
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating synergy heatmap: {e}")
            return self._create_empty_chart("Cluster Synergies")
    
    def _create_prediction_intervals_chart(self, results: Dict) -> str:
        """Create chart showing prediction intervals for ML estimates"""
        try:
            clusters = results.get('clusters', [])
            if not clusters and 'steps' in results:
                clusters = results.get('steps', {}).get('cluster_optimization', {}).get('clusters', [])
            clusters = clusters[:6]  # Top 6 clusters
            
            if not clusters:
                return self._create_empty_chart("Prediction Intervals")
            
            cluster_names = []
            gdp_predictions = []
            gdp_low = []
            gdp_high = []
            confidence_colors = []
            
            for cluster in clusters:
                cluster_names.append(cluster.get('name', 'Unknown')[:30])
                ml_preds = cluster.get('ml_predictions', {})
                
                gdp = ml_preds.get('gdp_impact', 0)
                gdp_predictions.append(gdp)
                
                # Get prediction intervals
                gdp_range = ml_preds.get('gdp_impact_range', {})
                gdp_low.append(gdp_range.get('low', gdp * 0.8))
                gdp_high.append(gdp_range.get('high', gdp * 1.2))
                
                # Color based on confidence
                confidence = cluster.get('confidence_score', 0.5)
                if confidence >= 0.8:
                    confidence_colors.append('rgba(46, 204, 113, 0.3)')  # Green
                elif confidence >= 0.6:
                    confidence_colors.append('rgba(243, 156, 18, 0.3)')  # Orange
                else:
                    confidence_colors.append('rgba(231, 76, 60, 0.3)')   # Red
            
            fig = go.Figure()
            
            # Add prediction intervals as filled areas
            for i, name in enumerate(cluster_names):
                fig.add_trace(go.Scatter(
                    x=[name, name],
                    y=[gdp_low[i], gdp_high[i]],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[name, name],
                    y=[gdp_high[i], gdp_low[i]],
                    fill='tonexty',
                    fillcolor=confidence_colors[i],
                    line=dict(width=0),
                    showlegend=False,
                    name=name,
                    hovertemplate='%{fullData.name}<br>Range: $%{y:,.0f}<extra></extra>'
                ))
            
            # Add point predictions
            fig.add_trace(go.Scatter(
                x=cluster_names,
                y=gdp_predictions,
                mode='markers',
                marker=dict(
                    size=12,
                    color=self.color_scheme['primary'],
                    line=dict(width=2, color='white')
                ),
                name='ML Prediction',
                hovertemplate='%{x}<br>Prediction: $%{y:,.0f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='ML GDP Impact Predictions with Confidence Intervals',
                xaxis_title='Cluster',
                yaxis_title='GDP Impact ($)',
                yaxis=dict(tickformat='$,.0f'),
                height=450,
                showlegend=True
            )
            fig.update_xaxes(tickangle=-45)
            
            return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            
        except Exception as e:
            logger.error(f"Error creating prediction intervals chart: {e}")
            return self._create_empty_chart("Prediction Intervals")
    
    def _create_empty_map(self, title: str) -> str:
        """Create an empty map with a message"""
        # Return a simple HTML map placeholder
        return f'''
        <div style="width: 100%; height: 400px; display: flex; align-items: center; justify-content: center; 
                    background-color: #f0f0f0; border: 1px solid #ddd; border-radius: 4px;">
            <div style="text-align: center; color: #666;">
                <i class="fa fa-map-marker-alt" style="font-size: 48px; margin-bottom: 10px;"></i>
                <h4>{title}</h4>
                <p>No geographic data available</p>
            </div>
        </div>
        '''
