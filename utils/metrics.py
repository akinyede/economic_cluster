"""Application metrics collection using Prometheus"""
import time
import functools
import logging
from prometheus_client import (
    Counter, Histogram, Gauge, CollectorRegistry, CONTENT_TYPE_LATEST, generate_latest
)
from flask import Response

logger = logging.getLogger(__name__)

REGISTRY = CollectorRegistry()

analysis_counter = Counter(
    'cluster_analyses_total',
    'Total number of cluster analyses performed',
    ['status', 'mode'],
    registry=REGISTRY
)

analysis_duration = Histogram(
    'analysis_duration_seconds',
    'Time spent performing cluster analysis',
    ['stage'],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800],
    registry=REGISTRY
)

active_analyses = Gauge(
    'active_analyses_count',
    'Number of currently running analyses',
    registry=REGISTRY
)

business_count_histogram = Histogram(
    'businesses_analyzed_count',
    'Number of businesses analyzed per request',
    buckets=[10, 50, 100, 500, 1000, 5000, 10000, 50000],
    registry=REGISTRY
)

cluster_count_histogram = Histogram(
    'clusters_identified_count',
    'Number of clusters identified',
    buckets=[1, 2, 3, 5, 7, 10, 15, 20],
    registry=REGISTRY
)

api_request_duration = Histogram(
    'external_api_duration_seconds',
    'Time spent calling external APIs',
    ['api_name', 'endpoint'],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
    registry=REGISTRY
)

api_request_counter = Counter(
    'external_api_requests_total',
    'Total external API requests',
    ['api_name', 'endpoint', 'status'],
    registry=REGISTRY
)

error_counter = Counter(
    'application_errors_total',
    'Total application errors',
    ['error_type', 'component'],
    registry=REGISTRY
)


class MetricsCollector:
    @staticmethod
    def track_analysis(mode: str = 'full'):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                active_analyses.inc()
                start = time.time()
                status = 'success'
                try:
                    result = func(*args, **kwargs)
                    try:
                        if isinstance(result, dict):
                            if 'total_businesses' in result:
                                business_count_histogram.observe(result['total_businesses'])
                            if 'total_clusters' in result:
                                cluster_count_histogram.observe(result['total_clusters'])
                    except Exception:
                        pass
                    return result
                except Exception as e:
                    status = 'failed'
                    error_counter.labels(error_type=type(e).__name__, component='analysis').inc()
                    raise
                finally:
                    dur = time.time() - start
                    analysis_duration.labels(stage='total').observe(dur)
                    analysis_counter.labels(status=status, mode=mode).inc()
                    active_analyses.dec()
                    logger.info(f"Analysis completed: status={status}, duration={dur:.2f}s, mode={mode}")
            return wrapper
        return decorator

    @staticmethod
    def track_stage(stage_name: str):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with analysis_duration.labels(stage=stage_name).time():
                    return func(*args, **kwargs)
            return wrapper
        return decorator


def get_metrics() -> Response:
    return Response(generate_latest(REGISTRY), mimetype=CONTENT_TYPE_LATEST)


metrics_collector = MetricsCollector()

