#!/usr/bin/env python3
"""
Simple runner for the KC Cluster Prediction Tool
Bypasses werkzeug development server issues on Python 3.9/macOS
"""

import os
import sys

# Ensure Werkzeug reloader env isn't forced; we'll disable it explicitly
os.environ.pop('WERKZEUG_RUN_MAIN', None)

# Force safer selector on macOS to avoid kqueue TypeError
try:
    import platform, selectors
    if platform.system() == 'Darwin' and hasattr(selectors, 'KqueueSelector'):
        selectors.DefaultSelector = selectors.SelectSelector
except Exception:
    pass

# Ensure Flask debug is off
os.environ.pop('FLASK_DEBUG', None)
# Relax CSP for local development to avoid Plotly/Bootstrap inline blocks breaking
os.environ.setdefault('RELAXED_CSP', 'true')

# Import and run the app
from app import create_app, socketio

def choose_port(preferred: int = 5001, attempts: int = 20) -> int:
    """Return a free TCP port on localhost, preferring `preferred`."""
    import socket
    for offset in range(attempts):
        port = preferred + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return preferred

if __name__ == '__main__':
    print("=" * 60)
    print("KC CLUSTER PREDICTION TOOL - STARTING SERVER")
    print("=" * 60)
    
    flask_app = create_app()
    # Allow overriding preferred port via env var PORT or CLUSTER_PORT
    try:
        preferred = int(os.getenv('PORT') or os.getenv('CLUSTER_PORT') or '5001')
    except Exception:
        preferred = 5001
    port = choose_port(preferred)
    
    print(f"\n✅ Server starting on http://localhost:{port}")
    print("✅ Enhanced KC models are loaded and ready")
    print("\nTo use the enhanced features:")
    print("1. Open http://localhost:5001 in your browser")
    print('2. Select "Enhanced with KC Data (Best Accuracy)" from ML Enhancement dropdown')
    print("3. Fill in your parameters and run analysis")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Persist chosen port for convenience
    try:
        with open('current_port.txt', 'w') as f:
            f.write(str(port))
    except Exception:
        pass

    # Run without debug to avoid werkzeug issues
    socketio.run(flask_app, 
                 debug=False,  # Disable debug to avoid reloader issues
                 port=port, 
                 host='127.0.0.1',
                 allow_unsafe_werkzeug=True, use_reloader=False)
