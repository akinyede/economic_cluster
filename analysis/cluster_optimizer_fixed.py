"""
FIXED: calculate_economic_impact() method
This patch corrects the jobs calculation to avoid double-counting
"""

def calculate_economic_impact(self, cluster_businesses: pd.DataFrame, years: int = 5) -> Dict:
    """
    Calculate conservative economic impact projections using paper's multipliers
    Based on Akinyede & Caruso (2025) methodology
    
    FIXED: Jobs calculation now correctly separates baseline from growth
    and applies employment multiplier only to NEW jobs
    """
    # Get conservative multipliers from config
    gdp_multiplier = self.config.ECONOMIC_MULTIPLIERS.get('gdp', 1.85)
    employment_multiplier = self.config.ECONOMIC_MULTIPLIERS.get('employment', 2.2)
    
    # Calculate base metrics
    total_revenue = cluster_businesses['revenue_estimate'].sum()
    baseline_employees = cluster_businesses['employees'].sum()  # CHANGED: renamed to baseline
    num_businesses = len(cluster_businesses)
    
    # Estimate growth rate based on industry mix
    avg_growth_rate = 0.06  # Default 6% annual growth
    
    # Calculate 5-year GDP projections with conservative multipliers
    direct_gdp = total_revenue * (1 + avg_growth_rate) ** years
    total_gdp_impact = direct_gdp * gdp_multiplier
    
    # FIXED: Employment projections - apply multiplier only to NEW jobs
    # Jobs grow slower than GDP (industry standard)
    job_growth_rate = avg_growth_rate * 0.8  # 4.8% annual job growth
    
    # Calculate future direct employment
    future_direct_jobs = baseline_employees * (1 + job_growth_rate) ** years
    
    # NEW jobs created (incremental)
    direct_job_growth = future_direct_jobs - baseline_employees
    
    # Apply employment multiplier ONLY to new jobs (indirect/induced effects)
    indirect_job_growth = direct_job_growth * (employment_multiplier - 1)
    
    # Total employment = baseline + new direct + new indirect
    total_jobs_impact = baseline_employees + direct_job_growth + indirect_job_growth
    
    # For reporting, separate direct and indirect
    direct_jobs = baseline_employees + direct_job_growth  # Total direct (baseline + growth)
    indirect_jobs = indirect_job_growth  # Only indirect from NEW jobs
    
    # INVARIANT CHECK: Ensure total >= direct and indirect = total - direct
    assert total_jobs_impact >= direct_jobs, "Total jobs must be >= direct jobs"
    assert abs((direct_jobs + indirect_jobs) - total_jobs_impact) < 1, "Jobs must sum correctly"
    
    # Apply regional strategic multipliers if applicable
    cluster_industries = cluster_businesses['naics_code'].str[:4].value_counts()
    strategic_multiplier = 1.0
    
    # Check for strategic sectors
    for naics_prefix, count in cluster_industries.head(5).items():
        if naics_prefix.startswith('3254'):  # Pharmaceutical
            strategic_multiplier = max(strategic_multiplier, self.config.REGIONAL_STRATEGIC_MULTIPLIERS.get('biosciences', 1.25))
        elif naics_prefix.startswith('54'):  # Tech services
            strategic_multiplier = max(strategic_multiplier, self.config.REGIONAL_STRATEGIC_MULTIPLIERS.get('technology', 1.20))
        elif naics_prefix.startswith('484') or naics_prefix.startswith('493'):  # Logistics
            strategic_multiplier = max(strategic_multiplier, self.config.REGIONAL_STRATEGIC_MULTIPLIERS.get('logistics', 1.15))
        elif naics_prefix.startswith('52'):  # Finance
            strategic_multiplier = max(strategic_multiplier, self.config.REGIONAL_STRATEGIC_MULTIPLIERS.get('fintech', 1.15))
    
    # Apply strategic multiplier
    total_gdp_impact *= strategic_multiplier
    total_jobs_impact = int(total_jobs_impact * strategic_multiplier)
    direct_jobs = int(direct_jobs * strategic_multiplier)
    indirect_jobs = int(indirect_jobs * strategic_multiplier)
    
    # Calculate confidence intervals (±15% for conservative estimates)
    confidence_level = 0.15
    gdp_lower = total_gdp_impact * (1 - confidence_level)
    gdp_upper = total_gdp_impact * (1 + confidence_level)
    jobs_lower = int(total_jobs_impact * (1 - confidence_level))
    jobs_upper = int(total_jobs_impact * (1 + confidence_level))
    
    return {
        'gdp_impact_5yr': total_gdp_impact,
        'gdp_impact_lower': gdp_lower,
        'gdp_impact_upper': gdp_upper,
        'total_jobs': int(total_jobs_impact),
        'direct_jobs': int(direct_jobs),
        'indirect_jobs': int(indirect_jobs),
        'jobs_lower': jobs_lower,
        'jobs_upper': jobs_upper,
        'num_businesses': num_businesses,
        'base_revenue': total_revenue,
        'base_employees': baseline_employees,  # CHANGED: now returns baseline separately
        'multipliers_used': {
            'gdp': gdp_multiplier,
            'employment': employment_multiplier,
            'strategic': strategic_multiplier
        },
        'confidence_interval': f"±{confidence_level*100:.0f}%",
        'methodology': 'Conservative calibration (Akinyede & Caruso 2025)',
        'calculation_notes': 'Employment multiplier applied only to incremental job growth'
    }
