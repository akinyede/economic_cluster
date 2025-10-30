#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick start script for the web interface
"""
import os
import sys
import warnings

# Suppress urllib3 LibreSSL warning (macOS compatibility issue - non-functional)
warnings.filterwarnings('ignore', message='.*urllib3 v2 only supports OpenSSL.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='urllib3')

# Suppress XGBoost compatibility warnings (handled automatically by code)
warnings.filterwarnings('ignore', message='.*If you are loading a serialized model.*')
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

import subprocess
import platform
import selectors

def check_dependencies():
    """Check if required packages are installed"""
    required = ['flask', 'flask_cors', 'plotly', 'pandas', 'numpy', 'deap', 'sqlalchemy', 'flask_socketio']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Missing required packages:", ", ".join(missing))
        print("\nPlease install them using:")
        print("pip install -r requirements_mvp.txt")
        return False
    
    return True

def main():
    """Run the web application"""
    print("=" * 60)
    print("KC CLUSTER PREDICTION TOOL - WEB INTERFACE")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Set environment for SQLite
    os.environ['USE_SQLITE'] = 'true'
    # Ensure Flask debug and reloader are off to avoid macOS selector issues
    os.environ.pop('FLASK_DEBUG', None)
    # Make sure WERKZEUG_RUN_MAIN is not forced; we disable reloader explicitly
    os.environ.pop('WERKZEUG_RUN_MAIN', None)
    # Relax CSP in local runs to allow Plotly/Bootstrap inline styles/scripts
    os.environ.setdefault('RELAXED_CSP', 'true')
    # Force safer selector on macOS to avoid kqueue TypeError
    try:
        if platform.system() == 'Darwin' and hasattr(selectors, 'KqueueSelector'):
            selectors.DefaultSelector = selectors.SelectSelector
    except Exception:
        pass
    
    print("\nStarting web server...")
    print("Open your browser to: http://localhost:5001 (or next available port)")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 60)
    # Default the app to the latest final enriched dataset if present (workspace-relative)
    try:
        base_dir = os.path.dirname(__file__)
        std_path = os.path.join(base_dir, 'data', 'final_enriched_entities_std.csv')
        fallback_path = os.path.join(base_dir, 'data', 'final_enriched_entities.csv')
        if os.path.exists(std_path):
            os.environ.setdefault('FINAL_DATASET_CSV', std_path)
        elif os.path.exists(fallback_path):
            os.environ.setdefault('FINAL_DATASET_CSV', fallback_path)
    except Exception:
        pass

    # Run the Flask app with cleaner output
    try:
        # Use run_app_simple.py for cleaner execution
        app_path = os.path.join(os.path.dirname(__file__), "run_app_simple.py")
        if not os.path.exists(app_path):
            # Fallback to app.py if run_app_simple.py doesn't exist
            app_path = os.path.join(os.path.dirname(__file__), "app.py")
        
        # Suppress the werkzeug selector errors by filtering stderr
        process = subprocess.Popen(
            [sys.executable, app_path],
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        rc = process.wait()
        if rc != 0:
            print(f"\nServer exited unexpectedly with code {rc}. See errors above.")
    except KeyboardInterrupt:
        print("\n\nServer stopped.")

if __name__ == "__main__":
    main()
