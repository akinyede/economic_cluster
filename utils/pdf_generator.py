"""PDF report generation module for KC Cluster Prediction Tool"""
import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.widgets.markers import makeMarker
import plotly.graph_objs as go
import plotly.io as pio
from typing import Dict, List
import base64
import logging

logger = logging.getLogger(__name__)

class PDFReportGenerator:
    """Generate professional PDF reports for cluster analysis results"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
        
    def _create_custom_styles(self):
        """Create custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#003366'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#003366'),
            spaceAfter=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#003366'),
            spaceAfter=8
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['BodyText'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=12
        ))
        
        self.styles.add(ParagraphStyle(
            name='ExecutiveSummary',
            parent=self.styles['BodyText'],
            fontSize=11,
            firstLineIndent=0,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=12,
            alignment=TA_JUSTIFY,
            textColor=colors.HexColor('#333333')
        ))
    
    def generate_report(self, results: Dict, output_buffer: io.BytesIO) -> io.BytesIO:
        """Generate complete PDF report"""
        doc = SimpleDocTemplate(
            output_buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )
        
        # Build the PDF content
        story = []
        
        # Title Page
        story.extend(self._create_title_page(results))
        story.append(PageBreak())
        
        # Executive Summary
        story.extend(self._create_executive_summary(results))
        story.append(PageBreak())
        
        # Economic Impact Analysis
        story.extend(self._create_economic_impact_section(results))
        story.append(PageBreak())
        
        # Cluster Details
        story.extend(self._create_cluster_details_section(results))
        story.append(PageBreak())
        
        # Recommendations
        story.extend(self._create_recommendations_section(results))
        
        # Build PDF
        doc.build(story)
        output_buffer.seek(0)
        
        return output_buffer
    
    def _create_title_page(self, results: Dict) -> List:
        """Create title page elements"""
        elements = []
        
        # Title
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph(
            "Kansas City MSA Economic Cluster Analysis",
            self.styles['CustomTitle']
        ))
        
        elements.append(Spacer(1, 0.5*inch))
        
        # Subtitle
        elements.append(Paragraph(
            "Strategic Economic Development Recommendations",
            self.styles['CustomHeading2']
        ))
        
        elements.append(Spacer(1, 1*inch))
        
        # Date
        date_str = datetime.now().strftime("%B %d, %Y")
        elements.append(Paragraph(
            f"Report Generated: {date_str}",
            self.styles['Normal']
        ))
        
        # Status
        status = results.get('status', 'Unknown')
        elements.append(Paragraph(
            f"Analysis Status: {status.title()}",
            self.styles['Normal']
        ))
        
        return elements
    
    def _create_executive_summary(self, results: Dict) -> List:
        """Create executive summary section"""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['CustomHeading1']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Key findings
        impact = results.get('economic_impact', {})
        clusters = results.get('steps', {}).get('cluster_optimization', {})
        
        summary_text = f"""
        The comprehensive analysis of the Kansas City Metropolitan Statistical Area has identified 
        {clusters.get('clusters_identified', 0)} viable economic development clusters with significant 
        potential for growth and job creation.
        
        <b>Key Economic Projections:</b><br/>
        • Total GDP Impact: ${impact.get('projected_gdp_impact', 0):,.0f}<br/>
        • Direct Jobs Created: {impact.get('projected_direct_jobs', 0):,}<br/>
        • Total Jobs (Direct + Indirect): {impact.get('projected_total_jobs', 0):,}<br/>
        • Target Achievement: {'Meets all targets' if impact.get('meets_targets', False) else 'Partial target achievement'}<br/>
        
        The analysis evaluated {results.get('steps', {}).get('data_collection', {}).get('businesses_collected', 0):,} 
        businesses across the region, considering factors including innovation capacity, market potential, 
        competitive positioning, and infrastructure alignment.
        """
        
        elements.append(Paragraph(summary_text, self.styles['ExecutiveSummary']))
        
        return elements
    
    def _create_economic_impact_section(self, results: Dict) -> List:
        """Create economic impact analysis section"""
        elements = []
        
        elements.append(Paragraph("Economic Impact Analysis", self.styles['CustomHeading1']))
        elements.append(Spacer(1, 0.2*inch))
        
        # Create impact table
        impact = results.get('economic_impact', {})
        
        data = [
            ['Metric', 'Projected Value', 'Target', 'Achievement %'],
            ['GDP Impact', f"${impact.get('projected_gdp_impact', 0):,.0f}", 
             '$2.87B', f"{impact.get('gdp_target_achievement', 0):.1f}%"],
            ['Direct Jobs', f"{impact.get('projected_direct_jobs', 0):,}", 
             '1,000', f"{(impact.get('projected_direct_jobs', 0) / 1000 * 100):.1f}%"],
            ['Total Jobs', f"{impact.get('projected_total_jobs', 0):,}", 
             '3,000', f"{impact.get('jobs_target_achievement', 0):.1f}%"],
        ]
        
        table = Table(data, colWidths=[2*inch, 1.5*inch, 1*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Add GDP impact chart
        elements.append(self._create_gdp_chart(results))
        
        return elements
    
    def _create_cluster_details_section(self, results: Dict) -> List:
        """Create detailed cluster information section"""
        elements = []
        
        elements.append(Paragraph("Top Economic Clusters", self.styles['CustomHeading1']))
        elements.append(Spacer(1, 0.2*inch))
        
        clusters = results.get('steps', {}).get('cluster_optimization', {}).get('clusters', [])
        
        for i, cluster in enumerate(clusters[:3]):
            # Cluster header
            elements.append(Paragraph(
                f"{i+1}. {cluster.get('name', 'Unknown Cluster')}",
                self.styles['CustomHeading2']
            ))
            
            # Cluster details
            details_text = f"""
            <b>Type:</b> {cluster.get('type', 'mixed').title()}<br/>
            <b>Total Score:</b> {cluster.get('total_score', 0):.1f}/100<br/>
            <b>Businesses:</b> {cluster.get('business_count', 0)}<br/>
            <b>Projected GDP Impact:</b> ${cluster.get('projected_gdp_impact', 0):,.0f}<br/>
            <b>Projected Jobs:</b> {cluster.get('projected_jobs', 0):,}<br/>
            <b>Longevity Score:</b> {cluster.get('longevity_score', 0):.1f}/10<br/>
            <b>Risk Level:</b> {'Low' if cluster.get('risk_score', 100) < 30 else 'Medium' if cluster.get('risk_score', 100) < 60 else 'High'}
            """
            
            elements.append(Paragraph(details_text, self.styles['CustomBody']))
            
            # Cluster scores table
            scores_data = [
                ['Factor', 'Score'],
                ['Natural Assets', f"{cluster.get('natural_assets_score', 0):.0f}"],
                ['Infrastructure', f"{cluster.get('infrastructure_score', 0):.0f}"],
                ['Workforce', f"{cluster.get('workforce_score', 0):.0f}"],
                ['Innovation', f"{cluster.get('innovation_capacity_score', 0):.0f}"],
                ['Market Access', f"{cluster.get('market_access_score', 0):.0f}"],
            ]
            
            scores_table = Table(scores_data, colWidths=[2*inch, 1*inch])
            scores_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(scores_table)
            elements.append(Spacer(1, 0.3*inch))
        
        return elements
    
    def _create_recommendations_section(self, results: Dict) -> List:
        """Create recommendations section"""
        elements = []
        
        elements.append(Paragraph("Strategic Recommendations", self.styles['CustomHeading1']))
        elements.append(Spacer(1, 0.2*inch))
        
        recs = results.get('steps', {}).get('recommendations', {})
        
        # Entrepreneur recommendations
        if recs.get('entrepreneurs'):
            elements.append(Paragraph("For Entrepreneurs", self.styles['CustomHeading2']))
            for rec in recs['entrepreneurs'][:3]:
                text = f"<b>{rec['opportunity']}</b>: {rec['rationale']} (Investment: {rec['investment_range']})"
                elements.append(Paragraph(f"• {text}", self.styles['CustomBody']))
            elements.append(Spacer(1, 0.2*inch))
        
        # Investor recommendations
        if recs.get('investors'):
            elements.append(Paragraph("For Investors", self.styles['CustomHeading2']))
            for rec in recs['investors'][:3]:
                text = f"<b>{rec['cluster']}</b>: Projected ROI {rec['projected_roi']}, Risk: {rec['risk_level']}"
                elements.append(Paragraph(f"• {text}", self.styles['CustomBody']))
            elements.append(Spacer(1, 0.2*inch))
        
        # Policymaker recommendations
        if recs.get('policymakers'):
            elements.append(Paragraph("For Policymakers", self.styles['CustomHeading2']))
            
            policy_data = [['Action', 'Impact', 'Priority', 'Timeline', 'Funding']]
            for rec in recs['policymakers'][:4]:
                policy_data.append([
                    rec['action'],
                    rec['impact'],
                    rec['priority'],
                    rec.get('timeline', 'N/A'),
                    rec.get('funding_needed', 'N/A')
                ])
            
            policy_table = Table(policy_data, colWidths=[2*inch, 2*inch, 0.8*inch, 0.8*inch, 0.8*inch])
            policy_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            
            elements.append(policy_table)
        
        return elements
    
    def _create_gdp_chart(self, results: Dict) -> Drawing:
        """Create GDP impact bar chart"""
        drawing = Drawing(400, 200)
        
        clusters = results.get('steps', {}).get('cluster_optimization', {}).get('clusters', [])[:3]
        
        if clusters:
            bc = VerticalBarChart()
            bc.x = 50
            bc.y = 50
            bc.height = 125
            bc.width = 300
            
            data = [[c.get('projected_gdp_impact', 0) / 1e9 for c in clusters]]  # Convert to billions
            bc.data = data
            bc.categoryAxis.categoryNames = [c.get('name', f'Cluster {i+1}') for i, c in enumerate(clusters)]
            bc.valueAxis.valueMin = 0
            bc.valueAxis.valueMax = max(data[0]) * 1.2 if data[0] else 10
            bc.valueAxis.valueStep = bc.valueAxis.valueMax / 5
            
            bc.bars[0].fillColor = colors.HexColor('#003366')
            
            drawing.add(bc)
            
            # Add title
            from reportlab.graphics.shapes import String
            drawing.add(String(200, 180, 'Projected GDP Impact by Cluster (Billions $)', 
                             fontSize=12, fillColor=colors.HexColor('#003366'), textAnchor='middle'))
        
        return drawing