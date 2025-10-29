from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SynGenConfig:
    n_scenarios: int = 1000
    seed: int = 42
    mgdp_multiplier: float = 1.85
    mjobs_multiplier: float = 2.2
    friction: float = 0.80


def gaussian_copula(marginals: Dict[str, Tuple[float, float]], corr: np.ndarray, n: int, seed: int = 42) -> pd.DataFrame:
    """Generate n samples with given marginals (mean, std) and correlation matrix via Gaussian copula."""
    rng = np.random.default_rng(seed)
    z = rng.multivariate_normal(mean=np.zeros(len(marginals)), cov=corr, size=n)
    # Standard normal CDF using scipy (numpy 2.0 removed np.erf)
    from scipy.stats import norm
    u = norm.cdf(z)
    cols = list(marginals.keys())
    data = {}
    for i, k in enumerate(cols):
        mean, std = marginals[k]
        # Invert Gaussian CDF to normal with given mean/std via percent point function
        from scipy.stats import norm as _norm
        x = mean + std * _norm.ppf(u[:, i])
        data[k] = x
    return pd.DataFrame(data)


def erfinv_clip(x: np.ndarray) -> np.ndarray:
    # Numerical stability helper for erfinv; clip to (-1+eps, 1-eps)
    x = np.clip(x, -0.999999, 0.999999)
    from scipy.special import erfinv  # optional heavy dep already in requirements
    return erfinv(x)


def generate_dataset(config: SynGenConfig) -> Tuple[pd.DataFrame, Dict]:
    """Generate synthetic cluster scenarios with audit log per paper SYN-GEN-5000 spirit."""
    rng = np.random.default_rng(config.seed)

    # Define marginals (mean, std) for base features
    marginals = {
        'business_count': (40, 20),
        'avg_employees': (120, 80),
        'avg_business_age': (12, 6),
        'strategic_score': (70, 12),
        'innovation_score': (65, 15),
    }
    # A simple correlation matrix among 5 features
    corr = np.array([
        [1.0, 0.3, -0.2, 0.2, 0.25],
        [0.3, 1.0, -0.1, 0.15, 0.2],
        [-0.2, -0.1, 1.0, -0.05, -0.05],
        [0.2, 0.15, -0.05, 1.0, 0.4],
        [0.25, 0.2, -0.05, 0.4, 1.0],
    ])

    base = gaussian_copula(marginals, corr, config.n_scenarios, seed=config.seed)
    base['business_count'] = np.clip(np.round(base['business_count']), 5, 200)
    base['avg_employees'] = np.clip(np.round(base['avg_employees']), 5, 5000)
    base['avg_business_age'] = np.clip(np.round(base['avg_business_age']), 1, 50)
    base['strategic_score'] = np.clip(base['strategic_score'], 30, 95)
    base['innovation_score'] = np.clip(base['innovation_score'], 20, 95)

    # Derive totals
    total_employees = base['business_count'] * base['avg_employees']
    # Revenue per employee by a rough industry mix factor: 150k..450k
    revenue_per_emp = 150_000 + (base['innovation_score'] - 20) / 75 * (450_000 - 150_000)
    total_revenue = total_employees * revenue_per_emp

    # Conservative GDP/Jobs using multipliers + friction
    gdp_impact = total_revenue * config.mgdp_multiplier * config.friction
    job_creation = (total_employees * config.mjobs_multiplier * config.friction).round().astype(int)

    df = pd.DataFrame({
        'business_count': base['business_count'],
        'total_employees': total_employees,
        'total_revenue': total_revenue,
        'avg_business_age': base['avg_business_age'],
        'strategic_score': base['strategic_score'],
        'innovation_score': base['innovation_score'],
        'gdp_impact': gdp_impact,
        'job_creation': job_creation,
    })

    audit = {
        'seed': config.seed,
        'n_scenarios': config.n_scenarios,
        'multipliers': {
            'mgdp': config.mgdp_multiplier,
            'mjobs': config.mjobs_multiplier,
            'friction': config.friction,
        },
        'marginals': marginals,
    }
    return df, audit


def save_dataset(df: pd.DataFrame, audit: Dict, out_csv: Union[str, Path], out_audit: Union[str, Path]) -> None:
    out_csv = Path(out_csv)
    out_audit = Path(out_audit)
    df.to_csv(out_csv, index=False)
    out_audit.write_text(json.dumps(audit, indent=2), encoding='utf-8')
