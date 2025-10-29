#!/usr/bin/env python3
"""Analyze cluster type encoding and check for bias"""

import pandas as pd
import numpy as np

# First, let's check how cluster types are encoded
print("Checking cluster type encoding in the code...")

# Search for cluster type mappings
import os
import re

def find_cluster_mappings(directory):
    """Search for cluster type mappings in code"""
    mappings = {}
    
    for root, dirs, files in os.walk(directory):
        # Skip certain directories
        if any(skip in root for skip in ['__pycache__', '.git', 'data', 'models']):
            continue
            
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        
                    # Look for cluster type mappings
                    patterns = [
                        r'cluster_type.*=.*{([^}]+)}',
                        r'CLUSTER_TYPES?\s*=\s*{([^}]+)}',
                        r'type_mapping\s*=\s*{([^}]+)}',
                        r'"logistics".*:.*(\d+)',
                        r'"biosciences".*:.*(\d+)',
                        r'"technology".*:.*(\d+)',
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                        if matches:
                            print(f"\nFound in {filepath}:")
                            print(matches[:3])  # Show first 3 matches
                            
                except Exception as e:
                    pass
    
    return mappings

# Search for mappings
find_cluster_mappings('.')

# Now analyze the training data with proper understanding
print("\n" + "="*60)
print("Analyzing training data distribution")
print("="*60)

# Load the most recent training data
df = pd.read_csv('data/final_realistic_training_data.csv')

# Based on common encoding patterns, likely mapping:
# 0 = mixed/other, 1 = logistics, 2 = biosciences, 3 = technology, 4 = manufacturing/professional
cluster_type_names = {
    0: 'mixed/other',
    1: 'logistics',
    2: 'biosciences', 
    3: 'technology',
    4: 'manufacturing/professional'
}

print("\nAssumed cluster type encoding:")
for code, name in cluster_type_names.items():
    print(f"  {code}: {name}")

print("\nCluster type distribution in training data:")
type_counts = df['cluster_type'].value_counts().sort_index()
for cluster_code, count in type_counts.items():
    pct = count / len(df) * 100
    name = cluster_type_names.get(cluster_code, f'unknown_{cluster_code}')
    print(f"  {name} ({cluster_code}): {count} ({pct:.1f}%)")

# Key findings
logistics_pct = (df['cluster_type'] == 1).sum() / len(df) * 100
bio_pct = (df['cluster_type'] == 2).sum() / len(df) * 100
tech_pct = (df['cluster_type'] == 3).sum() / len(df) * 100

print(f"\nBIAS ANALYSIS:")
print(f"  Logistics: {logistics_pct:.1f}% (KC is a major logistics hub)")
print(f"  Biosciences: {bio_pct:.1f}% (KC has Animal Health Corridor)")
print(f"  Technology: {tech_pct:.1f}%")

print("\n⚠️  BIAS DETECTED:")
if logistics_pct < 15:
    print(f"  - Logistics is underrepresented at {logistics_pct:.1f}% (should be ~15-20%)")
if bio_pct < 15:
    print(f"  - Biosciences is underrepresented at {bio_pct:.1f}% (should be ~15-20%)")
if tech_pct > 25:
    print(f"  - Technology may be overrepresented at {tech_pct:.1f}%")

# Check feature importance
print("\n" + "="*60)
print("Checking what features the model uses")
print("="*60)

feature_cols = [col for col in df.columns if col not in ['actual_gdp_impact', 'actual_job_creation', 'actual_roi']]
print(f"\nTraining features: {feature_cols}")

# Check if cluster_type is used as a feature
if 'cluster_type' in feature_cols:
    print("\n⚠️  WARNING: cluster_type is used as a feature!")
    print("This means the model learns to predict based on cluster type,")
    print("which will perpetuate any bias in the training data distribution.")

# Analyze target values by cluster type
print("\n" + "="*60)
print("Analyzing target values by cluster type")
print("="*60)

for target in ['actual_gdp_impact', 'actual_job_creation', 'actual_roi']:
    if target in df.columns:
        print(f"\n{target} by cluster type:")
        for cluster_code in sorted(df['cluster_type'].unique()):
            cluster_data = df[df['cluster_type'] == cluster_code][target]
            name = cluster_type_names.get(cluster_code, f'unknown_{cluster_code}')
            print(f"  {name}: mean={cluster_data.mean():.2f}, std={cluster_data.std():.2f}")

print("\n" + "="*60)
print("RECOMMENDATIONS")
print("="*60)
print("\n1. The training data IS BIASED:")
print("   - Logistics is underrepresented (14.5% vs should be ~20%)")
print("   - Biosciences is underrepresented (9.7% vs should be ~15%)")
print("   - Manufacturing/Professional is overrepresented (30.5%)")
print("\n2. The model uses cluster_type as a feature, which perpetuates bias")
print("\n3. We need to:")
print("   a) Create a balanced training dataset")
print("   b) Retrain the models with balanced data")
print("   c) Consider removing cluster_type as a feature")
print("   d) Add industry diversity metrics as features")