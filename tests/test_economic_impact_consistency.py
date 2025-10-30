import pandas as pd
from analysis.cluster_optimizer import ClusterOptimizer


def test_economic_impact_jobs_consistency():
    # Build a small synthetic cluster with varied revenue and employees
    df = pd.DataFrame({
        'revenue_estimate': [5_000_000, 12_000_000, 8_500_000, 3_250_000],
        'employees': [50, 120, 80, 32],
        'naics_code': ['5415', '4931', '3329', '3254'],
    })

    opt = ClusterOptimizer()
    impact = opt.calculate_economic_impact(df, years=5)

    # Basic invariants
    assert isinstance(impact['total_jobs'], int)
    assert isinstance(impact['direct_jobs'], int)
    assert isinstance(impact['indirect_jobs'], int)

    # total = direct + indirect; totals non-negative
    assert impact['total_jobs'] == impact['direct_jobs'] + impact['indirect_jobs']
    assert impact['total_jobs'] >= impact['direct_jobs'] >= 0
    assert impact['indirect_jobs'] >= 0

