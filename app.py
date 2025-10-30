# -*- coding: utf-8 -*-
"""Flask web application for KC Cluster Prediction Tool"""
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore
import atexit

# Suppress all non-critical warnings at application startup
warnings.filterwarnings('ignore', message='.*urllib3 v2 only supports OpenSSL.*')
warnings.filterwarnings('ignore', message='.*If you are loading a serialized model.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='urllib3')
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

import secrets
# Workaround: force SelectSelector on macOS to avoid kqueue TypeError in werkzeug
try:
    import platform, selectors
    if platform.system() == 'Darwin' and hasattr(selectors, 'KqueueSelector'):
        selectors.DefaultSelector = selectors.SelectSelector
except Exception:
    pass


from flask import Flask, render_template, request, jsonify, session, Response, send_file
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
from typing import Dict, Any, Optional
import logging
from threading import Thread, Lock
import uuid
import numpy as np
import json
import csv
import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import re
from functools import wraps
import os

# Local imports
from config import Config
from main import ClusterPredictionTool
from utils.cache import cache  # Import the new cache instance
from utils.visualization_generator import VisualizationGenerator
from database import db
from models import AnalysisResult, AnalysisCluster
from utils.metrics import get_metrics, metrics_collector
from config_validator import validate_analysis_params as validate_params_pydantic

# Import enhanced ML components
try:
    from ml.ensemble_model import EnsemblePredictor
    from analysis.ml_cluster_enhancer_v2 import MLClusterEnhancerV2
    from kc_enhancement_pipeline import KCEnhancementPipeline
    ENHANCED_MODELS_AVAILABLE = True
except ImportError:
    ENHANCED_MODELS_AVAILABLE = False
    logger.warning("Enhanced ML models not available - using original models")

# --- App Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use a global variable for the async_mode
socketio = SocketIO(async_mode='threading')

# Resource controls for background analysis tasks
ANALYSIS_THREAD_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix='analysis_worker')
ACTIVE_TASKS = {}
TASK_SEMAPHORE = Semaphore(10)

def convert_numpy_types(obj):
    """Recursively convert NumPy types to Python native types"""
    import math
    
    try:
        # Handle numpy arrays first (before any scalar checks)
        if isinstance(obj, np.ndarray):
            return [convert_numpy_types(item) for item in obj.tolist()]
        # Handle numpy types
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # Convert NaN to None for JSON compatibility
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Handle Python float NaN
        elif isinstance(obj, float) and math.isnan(obj):
            return None
        # Handle pandas Series/DataFrame
        elif hasattr(obj, 'to_dict'):
            return convert_numpy_types(obj.to_dict())
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        # Try to handle pandas NA/NaN for scalar values
        else:
            try:
                import pandas as pd
                if pd.isna(obj):
                    return None
            except:
                pass
            return obj
    except Exception as e:
        # If any conversion fails, return the object as-is
        return str(obj) if hasattr(obj, '__str__') else obj


# Input validation decorators
def validate_uuid(param_name):
    """Decorator to validate UUID parameters"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            param_value = kwargs.get(param_name)
            if param_value:
                try:
                    uuid.UUID(param_value)
                except ValueError:
                    return jsonify({'error': f'Invalid {param_name} format'}), 400
            return f(*args, **kwargs)
        return wrapper
    return decorator

def validate_export_id(param_name):
    """Decorator to validate export_id format"""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            param_value = kwargs.get(param_name)
            if param_value:
                # Export ID format: kc_analysis_YYYYMMDD_HHMMSS_XXXXXXXX
                pattern = r'^kc_analysis_\d{8}_\d{6}_[a-fA-F0-9]{8}$'
                if not re.match(pattern, param_value):
                    return jsonify({'error': f'Invalid {param_name} format'}), 400
            return f(*args, **kwargs)
        return wrapper
    return decorator

def validate_analysis_params(params):
    """Validate analysis parameters"""
    errors = []
    
    # Validate economic targets
    if 'economic_targets' in params:
        targets = params['economic_targets']
        if 'gdp_growth' in targets:
            if not isinstance(targets['gdp_growth'], (int, float)) or targets['gdp_growth'] < 0:
                errors.append("GDP growth must be a positive number")
        if 'direct_jobs' in targets:
            if not isinstance(targets['direct_jobs'], int) or targets['direct_jobs'] < 0:
                errors.append("Direct jobs must be a positive integer")
    
    # Validate business filters
    if 'business_filters' in params:
        filters = params['business_filters']
        if 'min_employees' in filters and 'max_employees' in filters:
            if filters['min_employees'] > filters['max_employees']:
                errors.append("Min employees cannot be greater than max employees")
    
    return errors

def create_app(config_class=Config):
    """Create and configure the Flask app."""
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.secret_key = secrets.token_hex(16)
    
    # Force UTF-8 encoding for all responses
    app.config['JSON_AS_ASCII'] = False
    
    # Initialize database
    with app.app_context():
        db.initialize(use_sqlite=True)  # Using SQLite for now
    
    # Initialize extensions
    socketio.init_app(app)
    cache.init_app(app)  # Initialize the cache with the app instance
    
    # Analysis task tracking (temporary in-memory storage)
    analysis_tasks = {}
    task_lock = Lock()  # Thread lock for safe access to analysis_tasks
    
    # Initialize enhanced models if available
    ensemble_predictor = None
    ml_enhancer = None
    
    if ENHANCED_MODELS_AVAILABLE:
        try:
            ensemble_predictor = EnsemblePredictor()
            if ensemble_predictor.load_models('models'):
                logger.info("Enhanced ensemble models loaded successfully")
            else:
                logger.warning("Failed to load enhanced models")
                ensemble_predictor = None
            
            ml_enhancer = MLClusterEnhancerV2({'use_kc_features': True})
            logger.info("ML Cluster Enhancer V2 initialized with KC features")
        except Exception as e:
            logger.error(f"Error loading enhanced models: {e}")
            ensemble_predictor = None
            ml_enhancer = None

    # --- Optional CSRF helpers (enabled via ENFORCE_CSRF=true) ---
    def _get_or_set_csrf_token():
        token = session.get('csrf_token')
        if not token:
            token = secrets.token_hex(16)
            session['csrf_token'] = token
        return token

    def require_csrf(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if os.getenv('ENFORCE_CSRF', 'false').lower() == 'true':
                expected = session.get('csrf_token')
                provided = request.headers.get('X-CSRF-Token') or request.args.get('csrf_token')
                if not expected or not provided or provided != expected:
                    return jsonify({'error': 'CSRF validation failed'}), 403
            return f(*args, **kwargs)
        return wrapper

    def save_analysis_result(task_id: str, params: Dict, results: Dict, visualizations: Dict, status: str = 'completed', error: str = None, session_id: str = None, user_ip: str = None):
        """Save analysis results to database"""
        try:
            db_session = db.get_session()
            
            # Create export ID
            from datetime import timezone
            export_id = f"kc_analysis_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{task_id[:8]}"
            
            # Extract summary metrics
            economic_impact = results.get('economic_impact', {})
            clusters = results.get('steps', {}).get('cluster_optimization', {}).get('clusters', [])
            
            # Get total businesses from the correct location in results
            total_businesses = 0
            if 'steps' in results and 'business_scoring' in results['steps']:
                total_businesses = results['steps']['business_scoring'].get('total_businesses', 0)
            
            # Create analysis result record
            analysis_result = AnalysisResult(
                task_id=task_id,
                status=status,
                error_message=error,
                parameters=params,
                results=results,
                visualizations=visualizations,
                total_clusters=len(clusters),
                total_businesses=total_businesses,
                projected_gdp_impact=economic_impact.get('projected_gdp_impact', 0),
                projected_total_jobs=economic_impact.get('projected_total_jobs', 0),
                meets_targets=economic_impact.get('meets_targets', False),
                analysis_mode='quick' if params.get('quick_mode', False) else 'full',
                export_id=export_id,
                session_id=session_id,
                user_ip=user_ip,
                completed_at=datetime.utcnow() if status == 'completed' else None
            )
            
            # Use explicit transaction to ensure atomicity
            with db_session.begin():
                db_session.add(analysis_result)
                
                # Save associated clusters
                for cluster_data in clusters:
                    analysis_cluster = AnalysisCluster(
                        analysis_result=analysis_result,
                        name=cluster_data.get('name', 'Unknown'),
                        type=cluster_data.get('type', 'mixed'),
                        business_count=cluster_data.get('business_count', 0),
                        projected_gdp_impact=cluster_data.get('projected_gdp_impact', 0),
                        projected_jobs=cluster_data.get('projected_jobs', 0),
                        confidence_score=cluster_data.get('confidence_score', 0),
                        cluster_data=cluster_data
                    )
                    db_session.add(analysis_cluster)
                # Commit happens automatically when exiting the with block
            logger.info(f"Analysis result saved to database with task_id: {task_id}, export_id: {export_id}")
            return export_id
            
        except Exception as e:
            logger.error(f"Failed to save analysis result: {str(e)}", exc_info=True)
            if db_session:
                db_session.rollback()
            return None
        finally:
            if db_session:
                db_session.close()

    def run_analysis_task(task_id: str, params: Dict, session_id: str = None, user_ip: str = None):
        """Run the analysis in a background thread."""
        try:
            logger.info(f"Starting analysis for task {task_id} with params: {params}")
            # Prepare per-task log output
            try:
                output_dir = os.path.join(os.path.dirname(__file__), 'analysis_output')
                os.makedirs(output_dir, exist_ok=True)
                task_log_path = os.path.join(output_dir, f'task_{task_id}.log')
            except Exception:
                task_log_path = None
            
            # Create a new app context for the background thread
            with app.app_context():
                # Initialize the tool within the app context
                tool = ClusterPredictionTool()
                
                # Progress callback function
                def progress_callback(stage, progress, message, highlight=None):
                    socketio.emit('analysis_progress', {
                        'task_id': task_id,
                        'stage': stage,
                        'progress': progress,
                        'message': message,
                        'highlight': highlight
                    })
                    # Persist a snapshot for polling clients and resilience
                    try:
                        with task_lock:
                            task = analysis_tasks.get(task_id, {})
                            task['progress'] = progress
                            task['message'] = message
                            task['stage'] = stage
                            if highlight:
                                hl = task.get('highlights', []) or []
                                hl.append(highlight)
                                task['highlights'] = hl[-30:]
                            analysis_tasks[task_id] = task
                        # Cache a lightweight snapshot for durability
                        cache.set(
                            f"analysis_progress_{task_id}",
                            {
                                'task_id': task_id,
                                'stage': stage,
                                'progress': progress,
                                'message': message,
                                'timestamp': int(__import__('time').time())
                            },
                            timeout=900
                        )
                        # Append to per-task log file for easy tailing
                        if task_log_path:
                            try:
                                from datetime import datetime as _dt
                                with open(task_log_path, 'a', encoding='utf-8') as _lf:
                                    ts = _dt.utcnow().isoformat() + 'Z'
                                    line = f"{ts}\tstage={stage}\tprogress={progress}\tmessage={message}"
                                    if highlight:
                                        line += f"\thighlight={highlight}"
                                    _lf.write(line + "\n")
                            except Exception:
                                pass
                    except Exception:
                        pass
                
                results = tool.run_full_analysis(custom_params=params, progress_callback=progress_callback)
                # Convert NumPy types to native Python types
                results = convert_numpy_types(results)
                
                # Generate visualizations
                viz_generator = VisualizationGenerator()
                visualizations = viz_generator.generate_all_visualizations(results)
                
                # Save to database
                export_id = save_analysis_result(task_id, params, results, visualizations, 'completed', None, session_id, user_ip)
                
                # Add both export_id and task_id to results so frontend can use them for exports
                results['export_id'] = export_id
                results['task_id'] = task_id
                
                # Store both results and visualizations in memory and cache
                with task_lock:
                    analysis_tasks[task_id]['result'] = results
                    analysis_tasks[task_id]['visualizations'] = visualizations
                    analysis_tasks[task_id]['status'] = 'completed'
                    analysis_tasks[task_id]['export_id'] = export_id
                
                logger.info(f"Analysis task {task_id} completed with {len(visualizations)} visualizations.")
                
                # Cache final results with timestamp to avoid any potential collisions
                import time
                cache_timestamp = int(time.time() * 1000)  # millisecond precision
                cache.set(f"analysis_result_{task_id}_{cache_timestamp}", results, timeout=3600)
                cache.set(f"analysis_visualizations_{task_id}_{cache_timestamp}", visualizations, timeout=3600)
                # Also cache with just task_id for backward compatibility
                cache.set(f"analysis_result_{task_id}", results, timeout=3600)
                cache.set(f"analysis_visualizations_{task_id}", visualizations, timeout=3600)
                
                # Write final marker to per-task log
                try:
                    if task_log_path:
                        from datetime import datetime as _dt
                        with open(task_log_path, 'a', encoding='utf-8') as _lf:
                            _lf.write(f"{_dt.utcnow().isoformat()}Z\tstage=completed\tprogress=100\tmessage=Analysis finished\texport_id={export_id}\n")
                except Exception:
                    pass

        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Error in analysis task {task_id}: {error_msg}", exc_info=True)
            
            # Save error to database (only once)
            save_analysis_result(task_id, params, {}, {}, 'failed', error_msg, session_id, user_ip)
            
            with task_lock:
                analysis_tasks[task_id]['status'] = 'failed'
                analysis_tasks[task_id]['error'] = error_msg
            socketio.emit('analysis_error', {
                'task_id': task_id,
                'error': error_msg,
                'traceback': traceback.format_exc()
            })
            # Write error marker to per-task log
            try:
                if task_log_path:
                    from datetime import datetime as _dt
                    with open(task_log_path, 'a', encoding='utf-8') as _lf:
                        _lf.write(f"{_dt.utcnow().isoformat()}Z\tstage=failed\tprogress=0\tmessage={error_msg}\n")
            except Exception:
                pass

    # Wrapped version with metrics
    @metrics_collector.track_analysis(mode='full')
    def run_analysis_task_with_metrics(task_id: str, params: Dict, session_id: str = None, user_ip: str = None):
        return run_analysis_task(task_id, params, session_id, user_ip)

    # --- Routes ---
    @app.route('/')
    def index():
        """Render the main application page"""
        # Create session ID if not exists
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        # Ensure CSRF token present for POSTs
        _get_or_set_csrf_token()
        return render_template('index.html')

    @app.route('/favicon.ico')
    def favicon():
        """Serve a default favicon to avoid 404 noise"""
        try:
            static_path = os.path.join(app.root_path, 'static', 'favicon.ico')
            if os.path.exists(static_path):
                return send_file(static_path)
        except Exception:
            pass
        # 204 No Content if not present
        return ('', 204)

    @app.route('/api/backtest', methods=['GET'])
    def api_backtest():
        """Run historical backtest and return JSON metrics"""
        try:
            from analysis.backtests_runner import run_historical_backtest
            metrics = run_historical_backtest()
            return jsonify({'status': 'ok', 'metrics': metrics})
        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/api/predict/enhanced', methods=['POST'])
    @require_csrf
    def predict_enhanced():
        """
        Make predictions using enhanced ensemble model with KC features
        
        Expected JSON body:
        {
            "business_count": 50,
            "total_employees": 5000,
            "total_revenue": 1000000000,
            "avg_business_age": 12,
            "strategic_score": 80,
            "innovation_score": 75,
            "critical_mass": 70,
            "supply_chain_completeness": 0.75,
            "geographic_density": 0.8,
            "workforce_diversity": 0.7,
            "natural_community_score": 0.65,
            "cluster_synergy": 75,
            "market_position": 70,
            "kc_crime_safety": 80,  # Optional KC features
            "kc_development_activity": 75,
            "kc_demographic_strength": 82,
            "kc_infrastructure_score": 85,
            "kc_market_access": 80
        }
        """
        try:
            if not ENHANCED_MODELS_AVAILABLE or not ensemble_predictor:
                return jsonify({'error': 'Enhanced models not available'}), 503
            
            # Get features from request
            features = request.json
            
            # Validate required features
            required_features = [
                'business_count', 'total_employees', 'total_revenue',
                'avg_business_age', 'strategic_score', 'innovation_score',
                'critical_mass'
            ]
            
            missing = [f for f in required_features if f not in features]
            if missing:
                return jsonify({
                    'error': 'Missing required features',
                    'missing_features': missing
                }), 400
            
            # Set defaults for optional features
            optional_defaults = {
                'supply_chain_completeness': 0.65,
                'geographic_density': 0.7,
                'workforce_diversity': 0.6,
                'natural_community_score': 0.6,
                'cluster_synergy': 65,
                'market_position': 65
            }
            
            for key, default in optional_defaults.items():
                if key not in features:
                    features[key] = default
            
            # Make predictions
            start_time = __import__('time').time()
            results = ensemble_predictor.predict(features, target='all')
            prediction_time = __import__('time').time() - start_time
            
            # Format response
            response = {
                'predictions': {
                    'gdp_impact': {
                        'value': results['gdp_impact']['prediction'],
                        'value_billions': results['gdp_impact']['prediction'] / 1e9,
                        'confidence': results['gdp_impact']['confidence'],
                        'model_used': results['gdp_impact']['model_used']
                    },
                    'job_creation': {
                        'value': int(results['job_creation']['prediction']),
                        'confidence': results['job_creation']['confidence'],
                        'model_used': results['job_creation']['model_used']
                    },
                    'roi_percentage': {
                        'value': results['roi_percentage']['prediction'],
                        'confidence': results['roi_percentage']['confidence'],
                        'model_used': results['roi_percentage']['model_used']
                    }
                },
                'metadata': {
                    'kc_data_quality': results['ensemble_metadata']['kc_data_quality'],
                    'model_strategy': results['ensemble_metadata']['model_selection'],
                    'overall_confidence': results['ensemble_metadata']['confidence'],
                    'prediction_time_ms': round(prediction_time * 1000, 2),
                    'enhanced_models': True
                }
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Enhanced prediction error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/models/status', methods=['GET'])
    def get_models_status():
        """Get status of loaded models including KC enhancement"""
        try:
            status = {
                'enhanced_models_available': ENHANCED_MODELS_AVAILABLE,
                'ensemble_loaded': False,
                'ml_enhancer_loaded': False,
                'models': {
                    '13_feature': [],
                    '18_feature': []
                },
                'kc_features': [],
                'version': '2.0'
            }
            
            if ENHANCED_MODELS_AVAILABLE and ensemble_predictor:
                status['ensemble_loaded'] = ensemble_predictor.is_loaded
                if ensemble_predictor.is_loaded:
                    status['models']['13_feature'] = list(ensemble_predictor.models_13.keys())
                    status['models']['18_feature'] = list(ensemble_predictor.models_18.keys())
                    status['kc_features'] = ensemble_predictor.kc_features
            
            if ENHANCED_MODELS_AVAILABLE and ml_enhancer:
                status['ml_enhancer_loaded'] = True
                status['ml_enhancer_uses_kc'] = ml_enhancer.use_kc_features
            
            # Load ensemble config if available
            from pathlib import Path
            config_path = Path('models/ensemble_config.json')
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    status['ensemble_config'] = config
            
            return jsonify(status)
            
        except Exception as e:
            logger.error(f"Model status error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/diagnostics', methods=['GET'])
    def api_diagnostics():
        """Return diagnostics including backtest metrics and ablation summary (cached)."""
        try:
            # Try cache first
            cache_key = 'diagnostics_metrics_v1'
            cached = cache.get(cache_key)
            if cached:
                return jsonify({'status': 'ok', 'cached': True, 'data': cached})

            from analysis.backtests_runner import run_historical_backtest
            from analysis.ablation_runner import run_ablation
            import time

            backtest = run_historical_backtest()
            ablation = run_ablation(n=1000, seed=13)
            data = {
                'backtest': backtest,
                'ablation': ablation,
                'generated_at': int(time.time())
            }
            # Cache for 15 minutes
            cache.set(cache_key, data, timeout=900)
            return jsonify({'status': 'ok', 'cached': False, 'data': data})
        except Exception as e:
            logger.error(f"Diagnostics failed: {e}", exc_info=True)
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/api/run_analysis_async', methods=['POST'])
    @require_csrf
    def run_analysis_async():
        """Start a new analysis task via managed thread pool with validation."""
        # Capacity gate
        if not TASK_SEMAPHORE.acquire(blocking=False):
            return jsonify({'error': 'Server at capacity', 'message': 'Too many concurrent analyses. Please try again later.'}), 503

        task_id = str(uuid.uuid4())
        raw_params = request.json or {}

        # Validate and normalize
        try:
            params = validate_params_pydantic(raw_params)
        except Exception as e:
            TASK_SEMAPHORE.release()
            return jsonify({'error': 'Invalid parameters', 'details': str(e)}), 400

        # Request context
        session_id = session.get('session_id')
        user_ip = request.remote_addr

        import time as _time
        with task_lock:
            analysis_tasks[task_id] = {
                'status': 'running',
                'result': None,
                'error': None,
                'progress': 0,
                'message': 'Queued',
                'stage': 'queued',
                'highlights': [],
                'start_time': int(_time.time())
            }

        # Submit to thread pool
        def _task_wrapper():
            try:
                run_analysis_task_with_metrics(task_id, params, session_id, user_ip)
            finally:
                TASK_SEMAPHORE.release()
                with task_lock:
                    ACTIVE_TASKS.pop(task_id, None)

        future = ANALYSIS_THREAD_POOL.submit(_task_wrapper)
        with task_lock:
            ACTIVE_TASKS[task_id] = future

        return jsonify({'task_id': task_id, 'status': 'Analysis started'})

    @app.route('/api/analysis_status/<task_id>', methods=['GET'])
    def get_analysis_status(task_id: str):
        """Check the status of an analysis task."""
        # First check database
        db_session = db.get_session()
        try:
            analysis_result = db_session.query(AnalysisResult).filter_by(task_id=task_id).first()
            if analysis_result:
                if analysis_result.status == 'completed':
                    return jsonify({
                        'status': 'completed',
                        'result': convert_numpy_types(analysis_result.results),
                        'visualizations': analysis_result.visualizations,
                        'export_id': analysis_result.export_id,
                        'task_id': task_id
                    })
                elif analysis_result.status == 'failed':
                    return jsonify({
                        'status': 'failed',
                        'error': analysis_result.error_message
                    })
                else:
                    return jsonify({'status': 'running'})
        finally:
            db_session.close()
        
        # Check in-memory storage
        with task_lock:
            task = analysis_tasks.get(task_id)
        if not task:
            # Check cache as fallback
            cached_result = cache.get(f"analysis_result_{task_id}")
            cached_visualizations = cache.get(f"analysis_visualizations_{task_id}")
            if cached_result:
                result = convert_numpy_types(cached_result)
                visualizations = cached_visualizations or {}
                return jsonify({'status': 'completed', 'result': result, 'visualizations': visualizations})
            return jsonify({'status': 'not_found'}), 404
        
        if task['status'] == 'completed':
            result = convert_numpy_types(task['result'])
            visualizations = task.get('visualizations', {})
            export_id = task.get('export_id')
            payload = {'status': 'completed', 'result': result, 'visualizations': visualizations, 'export_id': export_id}
            try:
                import time as _time
                if 'start_time' in task:
                    payload['elapsed_seconds'] = int(_time.time() - task['start_time'])
            except Exception:
                pass
            return jsonify(payload)
        elif task['status'] == 'failed':
            payload = {'status': 'failed', 'error': task['error']}
            try:
                import time as _time
                if 'start_time' in task:
                    payload['elapsed_seconds'] = int(_time.time() - task['start_time'])
            except Exception:
                pass
            return jsonify(payload)
        else:
            payload = {
                'status': 'running',
                'task_id': task_id,
                'progress': task.get('progress'),
                'message': task.get('message'),
                'stage': task.get('stage'),
                'highlights': task.get('highlights', [])
            }
            try:
                import time as _time
                if 'start_time' in task:
                    payload['elapsed_seconds'] = int(_time.time() - task['start_time'])
            except Exception:
                pass
            return jsonify(payload)

    # --- Export Endpoints ---
    @app.route('/api/export/json/<export_id>', methods=['GET'])
    @validate_export_id('export_id')
    def export_json(export_id: str):
        """Export analysis results as JSON"""
        db_session = db.get_session()
        try:
            analysis_result = db_session.query(AnalysisResult).filter_by(export_id=export_id).first()
            if not analysis_result:
                return jsonify({'error': 'Analysis not found'}), 404
            
            # Prepare export data
            export_data = {
                'metadata': {
                    'export_id': export_id,
                    'created_at': analysis_result.created_at.isoformat(),
                    'analysis_mode': analysis_result.analysis_mode,
                    'parameters': analysis_result.parameters
                },
                'results': analysis_result.results,
                'summary': {
                    'total_clusters': analysis_result.total_clusters,
                    'total_businesses': analysis_result.total_businesses,
                    'projected_gdp_impact': analysis_result.projected_gdp_impact,
                    'projected_total_jobs': analysis_result.projected_total_jobs,
                    'meets_targets': analysis_result.meets_targets
                }
            }
            
            return Response(
                json.dumps(export_data, indent=2),
                mimetype='application/json',
                headers={'Content-Disposition': f'attachment; filename={export_id}.json'}
            )
        finally:
            db_session.close()

    # Metrics and health endpoints
    @app.route('/metrics')
    def metrics():
        return get_metrics()

    # health endpoints added later with /readyz

    def initialize_resources():
        logger.info("Initializing application resources")
    # Flask 3 removed before_first_request; invoke directly at startup
    initialize_resources()

    def _cleanup_resources():
        try:
            logger.info("Shutting down analysis thread pool")
            ANALYSIS_THREAD_POOL.shutdown(wait=True, cancel_futures=True)
            logger.info("Thread pool shutdown complete")
        except Exception:
            pass

    atexit.register(_cleanup_resources)

    @app.route('/api/export/csv/<export_id>', methods=['GET'])
    @validate_export_id('export_id')
    def export_csv(export_id: str):
        """Export analysis results as CSV"""
        db_session = db.get_session()
        try:
            analysis_result = db_session.query(AnalysisResult).filter_by(export_id=export_id).first()
            if not analysis_result:
                return jsonify({'error': 'Analysis not found'}), 404
            
            # Create CSV in memory
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write summary section
            writer.writerow(['KC CLUSTER ANALYSIS RESULTS'])
            writer.writerow(['Export ID', export_id])
            writer.writerow(['Analysis Date', analysis_result.created_at.strftime('%Y-%m-%d %H:%M:%S')])
            writer.writerow([])
            
            # Write economic impact
            writer.writerow(['ECONOMIC IMPACT SUMMARY'])
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Clusters', analysis_result.total_clusters])
            writer.writerow(['Total Businesses Analyzed', analysis_result.total_businesses])
            writer.writerow(['Projected GDP Impact', f'${analysis_result.projected_gdp_impact:,.0f}'])
            writer.writerow(['Projected Total Jobs', f'{analysis_result.projected_total_jobs:,}'])
            writer.writerow(['Meets Targets', 'Yes' if analysis_result.meets_targets else 'No'])
            writer.writerow([])
            
            # Write cluster details
            writer.writerow(['CLUSTER DETAILS'])
            writer.writerow(['Cluster Name', 'Type', 'Business Count', 'GDP Impact', 'Jobs', 'Confidence Score', 'Critical Mass'])
            
            for cluster in analysis_result.analysis_clusters:
                # Get critical mass from cluster_data JSON if available
                critical_mass = 'N/A'
                if cluster.cluster_data and 'critical_mass' in cluster.cluster_data:
                    critical_mass = f"{cluster.cluster_data['critical_mass']:.0f}/100"
                
                writer.writerow([
                    cluster.name,
                    cluster.type,
                    cluster.business_count,
                    f'${cluster.projected_gdp_impact:,.0f}',
                    f'{cluster.projected_jobs:,}',
                    f'{cluster.confidence_score:.1%}',
                    critical_mass
                ])
            
            # Reset pointer to beginning
            output.seek(0)
            
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename={export_id}.csv'}
            )
        finally:
            db_session.close()

    @app.route('/api/analysis_log/<task_id>', methods=['GET'])
    @validate_uuid('task_id')
    def get_analysis_log(task_id: str):
        """Return the tail of the per-task analysis log for visibility into progress."""
        try:
            tail = int(request.args.get('tail', '200'))
        except Exception:
            tail = 200
        output_dir = os.path.join(os.path.dirname(__file__), 'analysis_output')
        task_log_path = os.path.join(output_dir, f'task_{task_id}.log')
        if not os.path.exists(task_log_path):
            return jsonify({'error': 'Log not found'}), 404
        try:
            # Simple tail implementation
            with open(task_log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            lines = lines[-tail:] if tail > 0 else lines
            return Response(''.join(lines), mimetype='text/plain')
        except Exception as e:
            logger.error(f"Failed to read analysis log for {task_id}: {e}")
            return jsonify({'error': 'Failed to read log'}), 500

    @app.route('/api/export/pdf/<export_id>', methods=['GET'])
    @validate_export_id('export_id')
    def export_pdf(export_id: str):
        """Export analysis results as PDF"""
        db_session = db.get_session()
        try:
            analysis_result = db_session.query(AnalysisResult).filter_by(export_id=export_id).first()
            if not analysis_result:
                return jsonify({'error': 'Analysis not found'}), 404
            
            # Create PDF in memory
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            story.append(Paragraph("KC Cluster Analysis Report", styles['Title']))
            story.append(Spacer(1, 12))
            
            # Metadata
            story.append(Paragraph(f"Export ID: {export_id}", styles['Normal']))
            story.append(Paragraph(f"Analysis Date: {analysis_result.created_at.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Summary
            story.append(Paragraph("Executive Summary", styles['Heading1']))
            summary_data = [
                ['Metric', 'Value'],
                ['Total Clusters Identified', str(analysis_result.total_clusters)],
                ['Total Businesses Analyzed', f'{analysis_result.total_businesses:,}'],
                ['Projected GDP Impact', f'${analysis_result.projected_gdp_impact:,.0f}'],
                ['Projected Total Jobs', f'{analysis_result.projected_total_jobs:,}'],
                ['Meets Economic Targets', 'Yes' if analysis_result.meets_targets else 'No']
            ]
            
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 12))
            
            # Cluster details
            story.append(Paragraph("Cluster Details", styles['Heading1']))
            cluster_data = [['Cluster Name', 'Type', 'Businesses', 'GDP Impact', 'Jobs', 'Critical Mass']]
            
            for cluster in analysis_result.analysis_clusters:
                # Get critical mass from cluster_data JSON if available
                critical_mass = 'N/A'
                if cluster.cluster_data and 'critical_mass' in cluster.cluster_data:
                    critical_mass = f"{cluster.cluster_data['critical_mass']:.0f}/100"
                
                cluster_data.append([
                    cluster.name,
                    cluster.type,
                    str(cluster.business_count),
                    f'${cluster.projected_gdp_impact:,.0f}',
                    f'{cluster.projected_jobs:,}',
                    critical_mass
                ])
            
            cluster_table = Table(cluster_data)
            cluster_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(cluster_table)
            
            # Build PDF
            doc.build(story)
            buffer.seek(0)
            
            return send_file(
                buffer,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f'{export_id}.pdf'
            )
        finally:
            db_session.close()
    
    @app.route('/api/export/pdf/by_task/<task_id>', methods=['GET'])
    @validate_uuid('task_id')
    def export_pdf_by_task(task_id: str):
        """Export PDF by task_id (fallback for frontend compatibility)"""
        db_session = db.get_session()
        try:
            # Find analysis by task_id instead of export_id
            analysis_result = db_session.query(AnalysisResult).filter_by(task_id=task_id).first()
            if not analysis_result:
                return jsonify({'error': 'Analysis not found'}), 404
            
            # Get the export_id and use the main PDF export logic
            export_id = analysis_result.export_id
            
            # Create PDF in memory (same logic as export_pdf)
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Title
            story.append(Paragraph("KC Cluster Analysis Report", styles['Title']))
            story.append(Spacer(1, 12))
            
            # Metadata
            story.append(Paragraph(f"Export ID: {export_id}", styles['Normal']))
            story.append(Paragraph(f"Analysis Date: {analysis_result.created_at.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Summary
            story.append(Paragraph("Executive Summary", styles['Heading1']))
            summary_data = [
                ['Metric', 'Value'],
                ['Total Clusters Identified', str(analysis_result.total_clusters)],
                ['Total Businesses Analyzed', f'{analysis_result.total_businesses:,}'],
                ['Projected GDP Impact', f'${analysis_result.projected_gdp_impact:,.0f}'],
                ['Projected Total Jobs', f'{analysis_result.projected_total_jobs:,}'],
                ['Meets Economic Targets', 'Yes' if analysis_result.meets_targets else 'No']
            ]
            
            summary_table = Table(summary_data)
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 12))
            
            # Cluster details
            story.append(Paragraph("Cluster Details", styles['Heading1']))
            cluster_data = [['Cluster Name', 'Type', 'Businesses', 'GDP Impact', 'Jobs', 'Critical Mass']]
            
            for cluster in analysis_result.analysis_clusters:
                # Get critical mass from cluster_data JSON if available
                critical_mass = 'N/A'
                if cluster.cluster_data and 'critical_mass' in cluster.cluster_data:
                    critical_mass = f"{cluster.cluster_data['critical_mass']:.0f}/100"
                
                cluster_data.append([
                    cluster.name,
                    cluster.type,
                    str(cluster.business_count),
                    f'${cluster.projected_gdp_impact:,.0f}',
                    f'{cluster.projected_jobs:,}',
                    critical_mass
                ])
            
            cluster_table = Table(cluster_data)
            cluster_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(cluster_table)
            
            # Build PDF
            doc.build(story)
            buffer.seek(0)
            
            return send_file(
                buffer,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=f'{export_id}.pdf'
            )
        except Exception as e:
            logger.error(f"Error exporting PDF for task_id {task_id}: {str(e)}", exc_info=True)
            return jsonify({'error': f'Failed to generate PDF: {str(e)}'}), 500
        finally:
            db_session.close()
    
    @app.route('/api/export/json/by_task/<task_id>', methods=['GET'])
    @validate_uuid('task_id')
    def export_json_by_task(task_id: str):
        """Export JSON by task_id (fallback for frontend compatibility)"""
        db_session = db.get_session()
        try:
            analysis_result = db_session.query(AnalysisResult).filter_by(task_id=task_id).first()
            if not analysis_result:
                return jsonify({'error': 'Analysis not found'}), 404
            
            # Prepare export data (same as export_json)
            export_data = {
                'metadata': {
                    'export_id': analysis_result.export_id,
                    'task_id': task_id,
                    'created_at': analysis_result.created_at.isoformat(),
                    'analysis_mode': analysis_result.analysis_mode,
                    'parameters': analysis_result.parameters
                },
                'results': analysis_result.results,
                'summary': {
                    'total_clusters': analysis_result.total_clusters,
                    'total_businesses': analysis_result.total_businesses,
                    'projected_gdp_impact': analysis_result.projected_gdp_impact,
                    'projected_total_jobs': analysis_result.projected_total_jobs,
                    'meets_targets': analysis_result.meets_targets
                }
            }
            
            return Response(
                json.dumps(export_data, indent=2),
                mimetype='application/json',
                headers={'Content-Disposition': f'attachment; filename={analysis_result.export_id}.json'}
            )
        except Exception as e:
            logger.error(f"Error exporting JSON for task_id {task_id}: {str(e)}", exc_info=True)
            return jsonify({'error': f'Failed to generate JSON: {str(e)}'}), 500
        finally:
            db_session.close()
    
    @app.route('/api/export/csv/by_task/<task_id>', methods=['GET'])
    @validate_uuid('task_id')
    def export_csv_by_task(task_id: str):
        """Export CSV by task_id (fallback for frontend compatibility)"""
        db_session = db.get_session()
        try:
            analysis_result = db_session.query(AnalysisResult).filter_by(task_id=task_id).first()
            if not analysis_result:
                return jsonify({'error': 'Analysis not found'}), 404
            
            # Create CSV in memory (same logic as export_csv)
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write summary section
            writer.writerow(['KC CLUSTER ANALYSIS RESULTS'])
            writer.writerow(['Export ID', analysis_result.export_id])
            writer.writerow(['Analysis Date', analysis_result.created_at.strftime('%Y-%m-%d %H:%M:%S')])
            writer.writerow([])
            
            # Write economic impact
            writer.writerow(['ECONOMIC IMPACT SUMMARY'])
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Clusters', analysis_result.total_clusters])
            writer.writerow(['Total Businesses Analyzed', analysis_result.total_businesses])
            writer.writerow(['Projected GDP Impact', f'${analysis_result.projected_gdp_impact:,.0f}'])
            writer.writerow(['Projected Total Jobs', f'{analysis_result.projected_total_jobs:,}'])
            writer.writerow(['Meets Targets', 'Yes' if analysis_result.meets_targets else 'No'])
            writer.writerow([])
            
            # Write cluster details
            writer.writerow(['CLUSTER DETAILS'])
            writer.writerow(['Cluster Name', 'Type', 'Business Count', 'GDP Impact', 'Jobs', 'Confidence Score', 'Critical Mass'])
            
            for cluster in analysis_result.analysis_clusters:
                # Get critical mass from cluster_data JSON if available
                critical_mass = 'N/A'
                if cluster.cluster_data and 'critical_mass' in cluster.cluster_data:
                    critical_mass = f"{cluster.cluster_data['critical_mass']:.0f}/100"
                
                writer.writerow([
                    cluster.name,
                    cluster.type,
                    cluster.business_count,
                    f'${cluster.projected_gdp_impact:,.0f}',
                    f'{cluster.projected_jobs:,}',
                    f'{cluster.confidence_score:.1%}',
                    critical_mass
                ])
            
            # Reset pointer to beginning
            output.seek(0)
            
            return Response(
                output.getvalue(),
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename={analysis_result.export_id}.csv'}
            )
        except Exception as e:
            logger.error(f"Error exporting CSV for task_id {task_id}: {str(e)}", exc_info=True)
            return jsonify({'error': f'Failed to generate CSV: {str(e)}'}), 500
        finally:
            db_session.close()

    # --- Scenario Management Endpoints ---
    @app.route('/api/scenarios/save', methods=['POST'])
    def save_scenario():
        """Save a scenario for later comparison"""
        try:
            data = request.json
            if not data or 'name' not in data:
                return jsonify({'status': 'error', 'message': 'Scenario name is required'}), 400
            
            # Generate scenario ID
            from datetime import timezone
            scenario_id = f"scenario_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            
            # Get current session
            session_id = session.get('session_id')
            
            # Save to cache with expiration (7 days)
            scenario_data = {
                'id': scenario_id,
                'name': data['name'],
                'parameters': data.get('parameters', {}),
                'results': data.get('results', {}),
                'created_at': datetime.utcnow().isoformat(),
                'session_id': session_id
            }
            
            # Store in cache
            cache.set(f"scenario_{scenario_id}", scenario_data, timeout=604800)  # 7 days
            
            # Also maintain a list of scenarios per session
            scenarios_list = cache.get(f"scenarios_list_{session_id}") or []
            scenarios_list.append({
                'id': scenario_id,
                'name': data['name'],
                'created_at': scenario_data['created_at']
            })
            cache.set(f"scenarios_list_{session_id}", scenarios_list, timeout=604800)
            
            return jsonify({
                'status': 'success',
                'scenario_id': scenario_id,
                'message': 'Scenario saved successfully'
            })
        except Exception as e:
            logger.error(f"Error saving scenario: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/api/scenarios/list', methods=['GET'])
    def list_scenarios():
        """List all saved scenarios for the current session"""
        try:
            session_id = session.get('session_id')
            scenarios_list = cache.get(f"scenarios_list_{session_id}") or []
            
            # Enrich with full data if needed
            full_scenarios = []
            for scenario_meta in scenarios_list:
                scenario_data = cache.get(f"scenario_{scenario_meta['id']}")
                if scenario_data:
                    full_scenarios.append({
                        'id': scenario_meta['id'],
                        'name': scenario_meta['name'],
                        'created_at': scenario_meta['created_at'],
                        'has_results': bool(scenario_data.get('results'))
                    })
            
            return jsonify({
                'status': 'success',
                'scenarios': full_scenarios
            })
        except Exception as e:
            logger.error(f"Error listing scenarios: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    @app.route('/api/scenarios/compare', methods=['POST'])
    def compare_scenarios():
        """Compare multiple scenarios"""
        try:
            data = request.json
            scenario_ids = data.get('scenario_ids', [])
            
            if not scenario_ids or len(scenario_ids) < 2:
                return jsonify({'status': 'error', 'message': 'At least 2 scenarios required for comparison'}), 400
            
            # Fetch all scenarios
            scenarios = []
            for scenario_id in scenario_ids:
                scenario_data = cache.get(f"scenario_{scenario_id}")
                if scenario_data:
                    scenarios.append(scenario_data)
            
            if len(scenarios) < 2:
                return jsonify({'status': 'error', 'message': 'Some scenarios not found'}), 404
            
            # Compare key metrics
            comparison = {
                'scenarios': [],
                'metrics': {
                    'gdp_impact': [],
                    'total_jobs': [],
                    'total_clusters': [],
                    'total_businesses': [],
                    'success_probability': []
                }
            }
            
            for scenario in scenarios:
                results = scenario.get('results', {})
                economic_impact = results.get('economic_impact', {})
                
                scenario_summary = {
                    'id': scenario['id'],
                    'name': scenario['name'],
                    'created_at': scenario['created_at']
                }
                comparison['scenarios'].append(scenario_summary)
                
                # Extract metrics
                comparison['metrics']['gdp_impact'].append(economic_impact.get('projected_gdp_impact', 0))
                comparison['metrics']['total_jobs'].append(economic_impact.get('projected_total_jobs', 0))
                comparison['metrics']['total_clusters'].append(results.get('total_clusters', 0))
                comparison['metrics']['total_businesses'].append(results.get('total_businesses', 0))
                comparison['metrics']['success_probability'].append(results.get('success_probability', 0))
            
            # Calculate differences
            comparison['analysis'] = {
                'best_gdp_scenario': scenarios[comparison['metrics']['gdp_impact'].index(max(comparison['metrics']['gdp_impact']))]['name'],
                'best_jobs_scenario': scenarios[comparison['metrics']['total_jobs'].index(max(comparison['metrics']['total_jobs']))]['name'],
                'most_clusters': scenarios[comparison['metrics']['total_clusters'].index(max(comparison['metrics']['total_clusters']))]['name']
            }
            
            return jsonify({
                'status': 'success',
                'comparison': comparison
            })
        except Exception as e:
            logger.error(f"Error comparing scenarios: {str(e)}")
            return jsonify({'status': 'error', 'message': str(e)}), 500

    # --- SocketIO Events ---
    @socketio.on('connect')
    def handle_connect():
        """Handle new client connections."""
        logger.info(f"Client connected: {request.sid}")
        emit('connection_response', {'message': 'Successfully connected to server!'})

    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnections."""
        logger.info(f"Client disconnected: {request.sid}")
    
    @app.route('/api/map/<task_id>', methods=['GET'])
    def get_map(task_id: str):
        """Get the map visualization for a completed analysis."""
        try:
            # First check database
            db_session = db.get_session()
            try:
                analysis_result = db_session.query(AnalysisResult).filter_by(task_id=task_id).first()
                if analysis_result and analysis_result.status == 'completed':
                    cached_result = analysis_result.results
                else:
                    # Fallback to cache
                    cached_result = cache.get(f"analysis_result_{task_id}")
                    if not cached_result:
                        # Check in-memory storage
                        with task_lock:
                            task = analysis_tasks.get(task_id)
                        if task and task['status'] == 'completed':
                            cached_result = task['result']
                        else:
                            return jsonify({'status': 'error', 'message': 'Analysis results not found'}), 404
            finally:
                db_session.close()
            
            # Generate map visualization
            from utils.map_generator import MapGenerator
            map_gen = MapGenerator()
            
            # Check if visualization mode is specified via query param (fallback to improved)
            viz_mode = request.args.get('visualization_mode', 'improved')
            if viz_mode == 'legacy':
                # Force legacy mode
                map_gen.visualization_mode = 'legacy'
            
            # Log the data structure for debugging
            logger.info(f"Generating map for task {task_id}")
            logger.info(f"Cached result keys: {list(cached_result.keys()) if cached_result else 'None'}")
            if cached_result and 'steps' in cached_result:
                logger.info(f"Steps keys: {list(cached_result['steps'].keys())}")
                if 'cluster_optimization' in cached_result['steps']:
                    cluster_data = cached_result['steps']['cluster_optimization']
                    logger.info(f"Cluster optimization keys: {list(cluster_data.keys())}")
                    if 'clusters' in cluster_data:
                        logger.info(f"Number of clusters: {len(cluster_data['clusters'])}")
            
            # Generate the HTML map
            map_html = map_gen.create_cluster_map(cached_result)
            
            # Return map HTML
            return jsonify({
                'status': 'success',
                'map_html': map_html
            })
            
        except Exception as e:
            logger.error(f"Error generating map for task {task_id}: {str(e)}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': f"Failed to generate map: {str(e)}"
            }), 500

    @app.route('/api/map/by_export/<export_id>', methods=['GET'])
    @validate_export_id('export_id')
    def get_map_by_export(export_id: str):
        """Get the map visualization using analysis export_id (frontend friendly)."""
        try:
            # Look up by export_id
            db_session = db.get_session()
            try:
                analysis_result = db_session.query(AnalysisResult).filter_by(export_id=export_id).first()
                if not analysis_result or analysis_result.status != 'completed':
                    return jsonify({'status': 'error', 'message': 'Analysis results not found'}), 404
                cached_result = analysis_result.results
            finally:
                db_session.close()

            from utils.map_generator import MapGenerator
            map_gen = MapGenerator()
            viz_mode = request.args.get('visualization_mode', 'improved')
            if viz_mode == 'legacy':
                map_gen.visualization_mode = 'legacy'
            map_html = map_gen.create_cluster_map(cached_result)
            return jsonify({'status': 'success', 'map_html': map_html})
        except Exception as e:
            logger.error(f"Error generating map for export_id {export_id}: {str(e)}", exc_info=True)
            return jsonify({'status': 'error', 'message': f"Failed to generate map: {str(e)}"}), 500
    
    # --- Session History ---
    @app.route('/api/analysis_history', methods=['GET'])
    def get_analysis_history():
        """Get analysis history for current session"""
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'analyses': []})
        
        db_session = db.get_session()
        try:
            analyses = db_session.query(AnalysisResult).filter_by(
                session_id=session_id
            ).order_by(AnalysisResult.created_at.desc()).limit(10).all()
            
            history = []
            for analysis in analyses:
                history.append({
                    'task_id': analysis.task_id,
                    'export_id': analysis.export_id,
                    'created_at': analysis.created_at.isoformat(),
                    'status': analysis.status,
                    'analysis_mode': analysis.analysis_mode,
                    'total_clusters': analysis.total_clusters,
                    'projected_gdp_impact': analysis.projected_gdp_impact,
                    'meets_targets': analysis.meets_targets
                })
            
            return jsonify({'analyses': history})
        finally:
            db_session.close()

    
    # Add security headers and UTF-8 encoding to all responses
    @app.after_request
    def add_security_headers(response):
        """Add security headers and UTF-8 encoding to all responses"""
        # Ensure UTF-8 encoding for all text responses
        if response.content_type and 'text/' in response.content_type or 'application/json' in response.content_type:
            if 'charset' not in response.content_type:
                response.content_type = response.content_type + '; charset=utf-8'
        
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        # Only add HSTS in production
        if not app.debug:
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'

        # Allow relaxing CSP in non-debug via env flag
        relaxed = os.getenv('RELAXED_CSP', 'false').lower() == 'true'

        if app.debug or relaxed:
            # Development/relaxed: allow inline styles/scripts and source maps from CDNs used
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://code.jquery.com https://cdn.plot.ly https://cdn.socket.io https://unpkg.com https://cdnjs.cloudflare.com https://maxcdn.bootstrapcdn.com; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://netdna.bootstrapcdn.com https://unpkg.com https://maxcdn.bootstrapcdn.com; "
                "font-src 'self' https://cdnjs.cloudflare.com https://netdna.bootstrapcdn.com https://maxcdn.bootstrapcdn.com https://cdn.jsdelivr.net data:; "
                "img-src 'self' data: blob: https://*.tile.openstreetmap.org https://*.basemaps.cartocdn.com https://unpkg.com https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
                "frame-src 'self' blob: https://www.openstreetmap.org https://leafletjs.com; "
                "connect-src 'self' ws: wss: https://cdn.socket.io https://cdn.jsdelivr.net https://*.tile.openstreetmap.org https://*.basemaps.cartocdn.com;"
            )
        else:
            # Production: stricter policy
            csp = (
                "default-src 'self'; "
                "script-src 'self' https://cdn.jsdelivr.net https://code.jquery.com https://cdn.plot.ly https://cdn.socket.io https://unpkg.com https://cdnjs.cloudflare.com https://maxcdn.bootstrapcdn.com; "
                "style-src 'self' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://netdna.bootstrapcdn.com https://unpkg.com https://maxcdn.bootstrapcdn.com; "
                "font-src 'self' https://cdnjs.cloudflare.com https://netdna.bootstrapcdn.com https://maxcdn.bootstrapcdn.com https://cdn.jsdelivr.net data:; "
                "img-src 'self' data: blob: https://*.tile.openstreetmap.org https://*.basemaps.cartocdn.com https://unpkg.com https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
                "frame-src 'self' blob: https://www.openstreetmap.org https://leafletjs.com; "
                "connect-src 'self' ws: wss: https://*.tile.openstreetmap.org https://*.basemaps.cartocdn.com;"
            )

        response.headers['Content-Security-Policy'] = csp
        return response

    # Health and readiness endpoints
    @app.route('/healthz', methods=['GET'])
    def healthz():
        return jsonify({'status': 'ok'}), 200

    @app.route('/readyz', methods=['GET'])
    def readyz():
        # Basic readiness: DB session attempt and cache access
        try:
            s = db.get_session()
            s.close()
            cache.set('ready_check', 'ok', timeout=10)
            cache.get('ready_check')
            return jsonify({'status': 'ready'}), 200
        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            return jsonify({'status': 'not_ready', 'error': str(e)}), 503

    return app

if __name__ == '__main__':
    import os
    flask_app = create_app()
    # Use SocketIO's development server
    # Disable debug mode to avoid werkzeug selector errors on Python 3.9/macOS
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    socketio.run(flask_app, debug=debug_mode, port=5001, allow_unsafe_werkzeug=True, use_reloader=False)
# CSRF utilities (optional, enabled via ENFORCE_CSRF=true)
def _get_or_set_csrf_token():
    token = session.get('csrf_token')
    if not token:
        token = secrets.token_hex(16)
        session['csrf_token'] = token
    return token

def require_csrf(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if os.getenv('ENFORCE_CSRF', 'false').lower() == 'true':
            expected = session.get('csrf_token')
            provided = request.headers.get('X-CSRF-Token') or request.args.get('csrf_token')
            if not expected or not provided or provided != expected:
                return jsonify({'error': 'CSRF validation failed'}), 403
        return f(*args, **kwargs)
    return wrapper
