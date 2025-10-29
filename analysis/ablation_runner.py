from dataclasses import dataclass
from typing import Dict

from .synthetic_generator import SynGenConfig, generate_dataset


@dataclass
class AblationResult:
    baseline_gdp_mean: float
    baseline_jobs_mean: float
    no_calibration_gdp_mean: float
    no_calibration_jobs_mean: float
    delta_gdp_pct: float
    delta_jobs_pct: float


def run_ablation(n: int = 1000, seed: int = 7) -> Dict:
    """Simple ablation over synthetic data to approximate calibration impact.

    Baseline uses MGDP=1.85, MJOBS=2.2, friction=0.80.
    No-calibration uses MGDP=1.0, MJOBS=1.0, friction=1.0.
    Returns mean GDP/Jobs and percentage deltas.
    """
    base_cfg = SynGenConfig(n_scenarios=n, seed=seed, mgdp_multiplier=1.85, mjobs_multiplier=2.2, friction=0.80)
    df_base, _ = generate_dataset(base_cfg)
    nc_cfg = SynGenConfig(n_scenarios=n, seed=seed, mgdp_multiplier=1.0, mjobs_multiplier=1.0, friction=1.0)
    df_nc, _ = generate_dataset(nc_cfg)

    gdp_mean_base = float(df_base['gdp_impact'].mean())
    jobs_mean_base = float(df_base['job_creation'].mean())
    gdp_mean_nc = float(df_nc['gdp_impact'].mean())
    jobs_mean_nc = float(df_nc['job_creation'].mean())

    delta_gdp_pct = ((gdp_mean_base - gdp_mean_nc) / gdp_mean_nc) * 100 if gdp_mean_nc > 0 else 0.0
    delta_jobs_pct = ((jobs_mean_base - jobs_mean_nc) / jobs_mean_nc) * 100 if jobs_mean_nc > 0 else 0.0

    return AblationResult(
        baseline_gdp_mean=gdp_mean_base,
        baseline_jobs_mean=jobs_mean_base,
        no_calibration_gdp_mean=gdp_mean_nc,
        no_calibration_jobs_mean=jobs_mean_nc,
        delta_gdp_pct=delta_gdp_pct,
        delta_jobs_pct=delta_jobs_pct,
    ).__dict__

