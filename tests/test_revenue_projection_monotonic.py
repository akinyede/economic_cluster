from analysis.revenue_projector import RevenueProjector


def test_revenue_projection_monotonic_growth():
    rp = RevenueProjector()
    business = {
        'revenue_estimate': 2_000_000,
        'employees': 25,
        'years_in_business': 5,
        'naics_code': '5415',  # computer systems
        'composite_score': 65,
    }
    proj = rp.project_business_revenue(business, years=5)
    years = sorted(proj['yearly_projections'].keys())
    first = proj['yearly_projections'][years[0]]['revenue']
    last = proj['yearly_projections'][years[-1]]['revenue']
    assert last >= first  # non-decreasing over horizon

