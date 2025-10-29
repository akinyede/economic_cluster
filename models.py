"""Data models for KC Cluster Prediction Tool"""
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Boolean, JSON, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Business(Base):
    __tablename__ = 'businesses'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    naics_code = Column(String(10))
    state = Column(String(2))
    county = Column(String(100))
    city = Column(String(100))
    year_established = Column(Integer)
    employees = Column(Integer)
    revenue_estimate = Column(Float)
    
    # Scoring metrics
    innovation_score = Column(Float, default=0)
    market_potential_score = Column(Float, default=0)
    competition_score = Column(Float, default=0)
    composite_score = Column(Float, default=0)
    
    # Innovation indicators
    patent_count = Column(Integer, default=0)
    sbir_awards = Column(Integer, default=0)
    r_and_d_investment = Column(Float)
    
    # Additional metadata
    data_source = Column(String(100))
    last_updated = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    cluster_memberships = relationship("ClusterMembership", back_populates="business")

class Cluster(Base):
    __tablename__ = 'clusters'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    type = Column(String(50))  # logistics, biosciences, technology, etc.
    
    # Scoring metrics
    natural_assets_score = Column(Float, default=0)
    infrastructure_score = Column(Float, default=0)
    workforce_score = Column(Float, default=0)
    innovation_score = Column(Float, default=0)
    market_access_score = Column(Float, default=0)
    geopolitical_score = Column(Float, default=0)
    resilience_score = Column(Float, default=0)
    total_score = Column(Float, default=0)
    
    # Financial metrics
    # ROI is stored as a fraction (e.g., 0.235 for 23.5%)
    roi = Column(Float)
    
    # Economic projections
    projected_gdp_impact = Column(Float)
    projected_direct_jobs = Column(Integer)
    projected_indirect_jobs = Column(Integer)
    projected_wage_impact = Column(Float)
    
    # Longevity assessment
    longevity_score = Column(Float)
    risk_factors = Column(JSON)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    memberships = relationship("ClusterMembership", back_populates="cluster")
    performance_metrics = relationship("ClusterPerformance", back_populates="cluster")

class ClusterMembership(Base):
    __tablename__ = 'cluster_memberships'
    
    id = Column(Integer, primary_key=True)
    business_id = Column(Integer, ForeignKey('businesses.id'))
    cluster_id = Column(Integer, ForeignKey('clusters.id'))
    role = Column(String(50))  # anchor, supplier, support, etc.
    synergy_score = Column(Float)
    
    # Relationships
    business = relationship("Business", back_populates="cluster_memberships")
    cluster = relationship("Cluster", back_populates="memberships")

class ClusterPerformance(Base):
    __tablename__ = 'cluster_performance'
    
    id = Column(Integer, primary_key=True)
    cluster_id = Column(Integer, ForeignKey('clusters.id'))
    
    # Actual performance metrics
    actual_gdp_impact = Column(Float)
    actual_direct_jobs = Column(Integer)
    actual_indirect_jobs = Column(Integer)
    actual_wage_change = Column(Float)
    
    # Performance tracking
    measurement_date = Column(DateTime, default=datetime.utcnow)
    performance_vs_target = Column(Float)
    
    # Relationships
    cluster = relationship("Cluster", back_populates="performance_metrics")

class DataSource(Base):
    __tablename__ = 'data_sources'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    url = Column(String(500))
    type = Column(String(50))  # business_registry, economic_data, patent_data, etc.
    last_scraped = Column(DateTime)
    scraping_frequency_hours = Column(Integer, default=24)
    is_active = Column(Boolean, default=True)
    api_key = Column(String(255))
    
class ScrapingLog(Base):
    __tablename__ = 'scraping_logs'
    
    id = Column(Integer, primary_key=True)
    data_source_id = Column(Integer, ForeignKey('data_sources.id'))
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    records_scraped = Column(Integer)
    errors = Column(JSON)
    status = Column(String(50))  # success, partial, failed

class MarketTrend(Base):
    __tablename__ = 'market_trends'
    
    id = Column(Integer, primary_key=True)
    industry = Column(String(100))
    naics_code = Column(String(10))
    trend_type = Column(String(50))  # growth_rate, demand, commodity_price
    value = Column(Float)
    measurement_date = Column(DateTime)
    source = Column(String(100))
    
class InfrastructureAsset(Base):
    __tablename__ = 'infrastructure_assets'
    
    id = Column(Integer, primary_key=True)
    type = Column(String(50))  # rail, highway, port, airport, utility
    name = Column(String(255))
    capacity = Column(Float)
    location = Column(JSON)  # GeoJSON format
    county = Column(String(100))
    
class WorkforceData(Base):
    __tablename__ = 'workforce_data'
    
    id = Column(Integer, primary_key=True)
    soc_code = Column(String(10))
    occupation_title = Column(String(255))
    employment_count = Column(Integer)
    median_wage = Column(Float)
    growth_rate = Column(Float)
    county = Column(String(100))
    measurement_year = Column(Integer)

# NEW MODEL: Analysis Results Storage
class AnalysisResult(Base):
    """Store complete analysis results for persistence and retrieval"""
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True)
    task_id = Column(String(36), unique=True, nullable=False, index=True)  # UUID
    
    # Analysis metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    status = Column(String(20), default='running')  # running, completed, failed
    error_message = Column(Text)
    
    # User parameters
    parameters = Column(JSON, nullable=False)
    
    # Analysis results
    results = Column(JSON)  # Complete results object
    visualizations = Column(JSON)  # Visualization data
    
    # Summary metrics for quick access
    total_clusters = Column(Integer)
    total_businesses = Column(Integer)
    projected_gdp_impact = Column(Float)
    projected_total_jobs = Column(Integer)
    meets_targets = Column(Boolean)
    analysis_mode = Column(String(20))  # quick or full
    
    # Export tracking
    export_id = Column(String(50), unique=True, index=True)  # For export URLs
    
    # Session tracking
    session_id = Column(String(100), index=True)  # For user session association
    user_ip = Column(String(45))  # For analytics
    
    # Relationships
    analysis_clusters = relationship("AnalysisCluster", back_populates="analysis_result", cascade="all, delete-orphan")

class AnalysisCluster(Base):
    """Store clusters associated with each analysis result"""
    __tablename__ = 'analysis_clusters'
    
    id = Column(Integer, primary_key=True)
    analysis_result_id = Column(Integer, ForeignKey('analysis_results.id'))
    
    # Cluster details
    name = Column(String(200))
    type = Column(String(50))
    business_count = Column(Integer)
    
    # Economic metrics
    projected_gdp_impact = Column(Float)
    projected_jobs = Column(Integer)
    confidence_score = Column(Float)
    
    # Store full cluster data
    cluster_data = Column(JSON)
    
    # Relationships
    analysis_result = relationship("AnalysisResult", back_populates="analysis_clusters")
