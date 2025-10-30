import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings

from analysis.cluster_optimizer import ClusterOptimizer


@pytest.fixture
def optimizer():
    return ClusterOptimizer()


@pytest.fixture
def sample_businesses():
    rng = np.random.default_rng(42)
    n = 100
    return pd.DataFrame({
        'name': [f'Biz {i}' for i in range(n)],
        'employees': rng.integers(5, 500, size=n),
        'revenue_estimate': rng.uniform(1e5, 1e7, size=n),
        'naics_code': rng.choice(['484', '511', '3254', '541'], size=n),
        'county': rng.choice(['Jackson County, MO', 'Johnson County, KS'], size=n),
        'year_established': rng.integers(1990, 2022, size=n),
        'composite_score': rng.uniform(40, 90, size=n),
        'patent_count': rng.poisson(2, size=n),
        'sbir_awards': rng.poisson(0.5, size=n)
    })


def test_determine_optimal_clusters(optimizer, sample_businesses):
    res = optimizer.determine_optimal_clusters(
        sample_businesses, economic_targets={'gdp_target': 1e9, 'job_target': 1000}, min_k=2, max_k=5
    )
    assert 'optimal_k' in res and 2 <= res['optimal_k'] <= 5


@given(
    st.lists(
        st.fixed_dictionaries({
            'name': st.text(min_size=1, max_size=32),
            'employees': st.integers(1, 10000),
            'revenue_estimate': st.floats(allow_nan=False, allow_infinity=False, min_value=1e4, max_value=1e9),
            'naics_code': st.sampled_from(['484', '511', '3254', '541']),
            'county': st.sampled_from(['Jackson County, MO', 'Johnson County, KS']),
            'year_established': st.integers(1950, 2024),
            'composite_score': st.floats(min_value=0, max_value=100),
            'patent_count': st.integers(0, 100),
            'sbir_awards': st.integers(0, 20)
        }),
        min_size=30, max_size=120
    )
)
@settings(max_examples=5, deadline=None)
def test_optimize_clusters_properties(businesses):
    df = pd.DataFrame(businesses)
    opt = ClusterOptimizer()
    clusters = opt.optimize_clusters(df, num_clusters=3, optimization_focus='balanced')
    assert clusters and all(c.get('business_count', 0) > 0 for c in clusters)
    # No duplicates across clusters by name
    names = [b['name'] for c in clusters for b in c.get('businesses', [])]
    assert len(names) == len(set(names))
