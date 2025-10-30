#!/bin/bash
set -e
echo "🚀 Starting KC Cluster Tool Improvements Implementation"

echo "📦 Creating backup..."
tar -czf backup_$(date +%Y%m%d_%H%M%S).tar.gz .

echo "📁 Ensuring directories..."
mkdir -p tests utils monitoring/grafana/dashboards monitoring/grafana/datasources

echo "📦 Installing new dependencies..."
pip install -U prometheus-client pydantic hypothesis pytest-cov psutil

echo "✅ Done"
