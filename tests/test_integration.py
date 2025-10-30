import os
import json
import pytest

os.environ.setdefault('USE_SQLITE', 'true')

from app import create_app


@pytest.fixture
def app():
    app = create_app()
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def test_metrics_endpoint(client):
    resp = client.get('/metrics')
    assert resp.status_code == 200
    assert b'cluster_analyses_total' in resp.data


def test_run_analysis_async_validation(client):
    # Invalid params should be rejected by pydantic
    payload = {"economic_targets": {"gdp_growth": -1}}
    resp = client.post('/api/run_analysis_async', json=payload, headers={'X-CSRF-Token': 'test'})
    assert resp.status_code in (400, 503)

