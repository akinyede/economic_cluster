import json
from pathlib import Path
from typing import Dict, Union

import pandas as pd

from .backtest import compute_mape


def run_historical_backtest(csv_path: Union[Path, str] = None) -> Dict:
    """Run backtest on historical outcomes CSV and return MAPE metrics.

    Expects CSV columns: projected_gdp, actual_gdp, projected_jobs, actual_jobs.
    Also computes a simple calibrated projection using the average realization
    rate observed in the dataset.
    """
    if csv_path is None:
        csv_path = Path(__file__).resolve().parents[1] / 'data' / 'kc_outcomes_2010_2020.csv'
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # Pre-calibration MAPE
    gdp_mape_pre = compute_mape(df['actual_gdp'].tolist(), df['projected_gdp'].tolist())
    jobs_mape_pre = compute_mape(df['actual_jobs'].tolist(), df['projected_jobs'].tolist())

    # Simple calibration: scale projections by mean realization rate (actual / projected)
    # Avoid divide-by-zero
    rates_gdp = []
    rates_jobs = []
    for _, row in df.iterrows():
        pg = float(row['projected_gdp'])
        ag = float(row['actual_gdp'])
        pj = float(row['projected_jobs'])
        aj = float(row['actual_jobs'])
        if pg > 0:
            rates_gdp.append(ag / pg)
        if pj > 0:
            rates_jobs.append(aj / pj)

    avg_rate_gdp = sum(rates_gdp) / len(rates_gdp) if rates_gdp else 1.0
    avg_rate_jobs = sum(rates_jobs) / len(rates_jobs) if rates_jobs else 1.0

    df['calib_gdp'] = df['projected_gdp'] * avg_rate_gdp
    df['calib_jobs'] = df['projected_jobs'] * avg_rate_jobs

    gdp_mape_post = compute_mape(df['actual_gdp'].tolist(), df['calib_gdp'].tolist())
    jobs_mape_post = compute_mape(df['actual_jobs'].tolist(), df['calib_jobs'].tolist())

    return {
        'avg_rate_gdp': avg_rate_gdp,
        'avg_rate_jobs': avg_rate_jobs,
        'gdp_mape_pre': gdp_mape_pre,
        'jobs_mape_pre': jobs_mape_pre,
        'gdp_mape_post': gdp_mape_post,
        'jobs_mape_post': jobs_mape_post,
    }


def save_backtest_report(output_path: Union[Path, str], metrics: Dict) -> None:
    output_path = Path(output_path)
    output_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')

