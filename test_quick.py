#!/usr/bin/env python3
"""
Quick test script to verify models work
"""
import sys
import os

print("=" * 80)
print("Quick Test - Retail ML Models")
print("=" * 80)

# Test imports
print("\n[1/5] Testing imports...")
try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    print("✓ Core packages imported")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test data loading
print("\n[2/5] Testing data loading...")
try:
    from utils.dataset_acquisition import RetailDatasetAcquisition
    acquirer = RetailDatasetAcquisition(data_dir='data/raw')
    datasets = acquirer.load_dataset('synthetic')
    print(f"✓ Loaded {len(datasets)} datasets")
    for name, df in datasets.items():
        print(f"  - {name}: {len(df)} rows")
except Exception as e:
    print(f"✗ Data loading error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test model initialization
print("\n[3/5] Testing model initialization...")
try:
    from models.demand_forecasting import DemandForecastingModel
    from models.clv_retention import CLVModel
    from models.markdown_optimization import MarkdownOptimizationModel
    print("✓ All models can be imported")
except Exception as e:
    print(f"✗ Model import error: {e}")
    sys.exit(1)

# Test quick model training
print("\n[4/5] Testing quick model training (sample data)...")
try:
    transactions = datasets['transactions']
    products = datasets['products']
    customers = datasets['customers']
    
    # Use more data for proper feature engineering (lag features need history)
    sample_transactions = transactions.head(50000)
    
    print("  Training demand forecasting model...")
    demand_model = DemandForecastingModel()
    # Use store level for faster training
    demand_model.train(sample_transactions, products, None, hierarchy_level='store')
    print("  ✓ Demand forecasting model trained")
    
    print("  Training CLV model...")
    clv_model = CLVModel()
    clv_model.train(sample_transactions, customers)
    print("  ✓ CLV model trained")
    
except Exception as e:
    print(f"✗ Model training error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n[5/5] Summary...")
print("=" * 80)
print("✓ All tests passed!")
print("✓ Ready to run full pipeline with: python3 main.py")
print("=" * 80)

