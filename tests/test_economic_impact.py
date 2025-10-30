import pandas as pd
from analysis.cluster_optimizer import ClusterOptimizer


def test_jobs_calculation_consistency():
    optimizer = ClusterOptimizer()
    test_data = pd.DataFrame({
        'revenue_estimate': [1_000_000] * 10,
        'employees': [100] * 10,
        'naics_code': ['5415'] * 10
    })
    result = optimizer.calculate_economic_impact(test_data, years=5)

    assert result['total_jobs'] >= result['direct_jobs']
    assert (result['direct_jobs'] + result['indirect_jobs']) == result['total_jobs']


def test_no_negative_jobs():
    optimizer = ClusterOptimizer()
    test_data = pd.DataFrame({
        'revenue_estimate': [100_000] * 5,
        'employees': [10] * 5,
        'naics_code': ['5415'] * 5
    })
    result = optimizer.calculate_economic_impact(test_data, years=5)

    assert result['total_jobs'] > 0
    assert result['direct_jobs'] > 0
    assert result['indirect_jobs'] >= 0


def test_multiplier_applied_only_to_new_jobs():
    optimizer = ClusterOptimizer()
    baseline_employees = 1000
    test_data = pd.DataFrame({
        'revenue_estimate': [10_000_000],
        'employees': [baseline_employees],
        'naics_code': ['5415']
    })
    years = 5
    result = optimizer.calculate_economic_impact(test_data, years=years)

    # Expected using the corrected approach (before strategic rounding)
    employment_mult = optimizer.config.ECONOMIC_MULTIPLIERS.get('employment', 2.2)
    job_growth_rate = 0.06 * 0.8
    future_direct = baseline_employees * (1 + job_growth_rate) ** years
    direct_growth = future_direct - baseline_employees
    indirect_growth = direct_growth * (employment_mult - 1)
    expected_total = baseline_employees + direct_growth + indirect_growth

    # Include strategic multiplier for NAICS 54x (technology)
    tech_mult = optimizer.config.REGIONAL_STRATEGIC_MULTIPLIERS.get('technology', 1.0)
    expected_total *= tech_mult
    # Allow 2% tolerance due to rounding
    assert abs(result['total_jobs'] - expected_total) / expected_total < 0.02
