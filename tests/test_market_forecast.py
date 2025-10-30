from analysis.market_analyzer import EnhancedMarketAnalyzer


def test_market_forecast_monotonic():
    ma = EnhancedMarketAnalyzer()
    fc = ma._forecast_demand('logistics', years=5)
    years = sorted(fc['forecast_years'].keys())
    vals = [fc['forecast_years'][y] for y in years]
    # should strictly increase given positive growth rate
    assert all(v2 >= v1 for v1, v2 in zip(vals, vals[1:]))

