#!/usr/bin/env python3
"""
Comprehensive test of all 8 retail ML models
Trains and tests each model with sample data
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 80)
print("Testing All 8 Retail ML Models")
print("=" * 80)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Load data
print("[Loading Data]")
from utils.dataset_acquisition import RetailDatasetAcquisition

acquirer = RetailDatasetAcquisition(data_dir='data/raw')
datasets = acquirer.load_dataset('synthetic')

transactions = datasets['transactions']
products = datasets['products']
customers = datasets['customers']
returns = datasets.get('returns')
inventory = datasets.get('inventory')

print(f"✓ Loaded {len(transactions):,} transactions, {len(products):,} products, {len(customers):,} customers")
if returns is not None:
    print(f"✓ Loaded {len(returns):,} returns")
if inventory is not None:
    print(f"✓ Loaded {len(inventory):,} inventory records")

# Use sample data for faster testing
print("\n[Using sample data for faster testing]")
sample_transactions = transactions.head(50000)  # 50K transactions
sample_products = products.head(500)  # 500 products
sample_customers = customers.head(1000)  # 1K customers
if returns is not None:
    sample_returns = returns.head(5000)
else:
    sample_returns = None

results = {}

# Model 1: Demand Forecasting
print("\n" + "=" * 80)
print("MODEL 1: Demand Forecasting & Inventory Optimization")
print("=" * 80)
try:
    from models.demand_forecasting import DemandForecastingModel
    
    print("Training model...")
    demand_model = DemandForecastingModel()
    # Use more data and category level for better feature engineering
    larger_sample = sample_transactions.head(20000).copy()
    # Ensure date is datetime
    if 'date' in larger_sample.columns:
        larger_sample['date'] = pd.to_datetime(larger_sample['date'])
    try:
        demand_model.train(larger_sample, sample_products, inventory, hierarchy_level='category')
        print("✓ Model trained")
    except Exception as e:
        print(f"⚠ Training failed (likely insufficient data after feature engineering): {e}")
        print("  This is expected with small samples due to lag features requiring history")
        raise
    
    print("Generating forecasts...")
    forecasts = demand_model.predict(sample_transactions, sample_products, horizon=7, hierarchy_level='category')
    print(f"✓ Generated {len(forecasts)} forecasts")
    print(f"  Sample forecast: {forecasts.head(3).to_string()}")
    
    results['Demand Forecasting'] = {'status': 'SUCCESS', 'forecasts': len(forecasts)}
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    results['Demand Forecasting'] = {'status': 'FAILED', 'error': str(e)}

# Model 2: Markdown Optimization
print("\n" + "=" * 80)
print("MODEL 2: Markdown Optimization & Pricing Strategy")
print("=" * 80)
try:
    from models.markdown_optimization import MarkdownOptimizationModel
    
    print("Training model...")
    markdown_model = MarkdownOptimizationModel()
    # Ensure transactions have date column for month extraction
    txn_with_date = sample_transactions.copy()
    if 'date' in txn_with_date.columns:
        txn_with_date['date'] = pd.to_datetime(txn_with_date['date'])
        txn_with_date['month'] = txn_with_date['date'].dt.month
    # Ensure category exists
    if 'category' not in txn_with_date.columns and 'category' in sample_products.columns:
        txn_with_date = txn_with_date.merge(
            sample_products[['product_id', 'category']], 
            on='product_id', 
            how='left'
        )
    markdown_model.train(txn_with_date, sample_products)
    print("✓ Model trained")
    
    print("Generating markdown recommendations...")
    # Create sample inventory
    sample_inventory = pd.DataFrame({
        'product_id': sample_products['product_id'].head(50),
        'stock_level': np.random.randint(20, 100, size=50)
    })
    
    recommendations = markdown_model.predict_optimal_markdown(
        sample_products.head(50), sample_inventory, days_until_end_of_season=30
    )
    print(f"✓ Generated {len(recommendations)} recommendations")
    if len(recommendations) > 0:
        print(f"  Sample recommendation:\n{recommendations.head(3).to_string()}")
    
    results['Markdown Optimization'] = {'status': 'SUCCESS', 'recommendations': len(recommendations)}
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    results['Markdown Optimization'] = {'status': 'FAILED', 'error': str(e)}

# Model 3: CLV & Retention
print("\n" + "=" * 80)
print("MODEL 3: Customer Lifetime Value & Retention")
print("=" * 80)
try:
    from models.clv_retention import CLVModel
    
    print("Training model...")
    clv_model = CLVModel()
    clv_model.train(sample_transactions, sample_customers)
    print("✓ Model trained")
    
    print("Calculating CLV scores...")
    clv_scores = clv_model.predict_customer_value(sample_customers, sample_transactions)
    print(f"✓ Calculated CLV for {len(clv_scores)} customers")
    if 'total_clv' in clv_scores.columns:
        print(f"  CLV stats: Mean=${clv_scores['total_clv'].mean():.2f}, "
              f"Median=${clv_scores['total_clv'].median():.2f}")
    elif 'predicted_future_value' in clv_scores.columns:
        print(f"  CLV stats: Mean=${clv_scores['predicted_future_value'].mean():.2f}, "
              f"Median=${clv_scores['predicted_future_value'].median():.2f}")
    
    print("Identifying at-risk customers...")
    try:
        at_risk = clv_model.identify_at_risk_customers(threshold_probability=0.5)
        print(f"✓ Identified {len(at_risk)} at-risk customers")
    except Exception as e:
        # Fallback: identify from CLV scores directly
        if 'churn_probability' in clv_scores.columns:
            at_risk = clv_scores[clv_scores['churn_probability'] > 0.5]
            print(f"✓ Identified {len(at_risk)} at-risk customers (using CLV scores)")
        else:
            print(f"⚠ Could not identify at-risk customers: {e}")
    
    results['CLV & Retention'] = {'status': 'SUCCESS', 'clv_scores': len(clv_scores), 'at_risk': len(at_risk)}
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    results['CLV & Retention'] = {'status': 'FAILED', 'error': str(e)}

# Model 4: Recommendations
print("\n" + "=" * 80)
print("MODEL 4: Personalization & Recommendation System")
print("=" * 80)
try:
    from models.recommendations import RecommendationSystem
    
    print("Training model...")
    rec_system = RecommendationSystem(n_factors=20)  # Smaller for faster training
    rec_system.train(sample_transactions, sample_products)
    print("✓ Model trained")
    
    print("Generating recommendations...")
    test_customers = sample_customers['customer_id'].head(3).tolist()
    total_recs = 0
    for customer_id in test_customers:
        recs = rec_system.recommend_hybrid(
            customer_id, sample_transactions, sample_products, n_recommendations=5
        )
        if recs is not None and len(recs) > 0:
            total_recs += len(recs)
            print(f"  Customer {customer_id}: {len(recs)} recommendations")
    
    print(f"✓ Generated recommendations for {len(test_customers)} customers")
    results['Recommendations'] = {'status': 'SUCCESS', 'recommendations': total_recs}
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    results['Recommendations'] = {'status': 'FAILED', 'error': str(e)}

# Model 5: Returns Prediction
print("\n" + "=" * 80)
print("MODEL 5: Returns Prediction & Reduction")
print("=" * 80)
try:
    from models.returns_prediction import ReturnsPredictionModel
    
    if sample_returns is not None and len(sample_returns) > 0:
        print("Training model...")
        returns_model = ReturnsPredictionModel()
        # Merge returns info into transactions
        txn_with_returns = sample_transactions.copy()
        returned_txn_ids = set(sample_returns['transaction_id'].values)
        txn_with_returns['is_returned'] = txn_with_returns['transaction_id'].isin(returned_txn_ids).astype(int)
        returns_model.train(txn_with_returns, sample_returns, sample_products, sample_customers)
        print("✓ Model trained")
        
        print("Predicting return probabilities...")
        test_transactions = sample_transactions.head(100)
        return_predictions = returns_model.predict_return_probability(
            test_transactions, sample_products, sample_customers
        )
        print(f"✓ Predicted return risk for {len(return_predictions)} transactions")
        
        if 'return_risk' in return_predictions.columns:
            high_risk = return_predictions[return_predictions['return_risk'].isin(['High', 'Very High'])]
            print(f"  High-risk transactions: {len(high_risk)}")
        
        results['Returns Prediction'] = {'status': 'SUCCESS', 'predictions': len(return_predictions)}
    else:
        print("⚠ No returns data available, skipping")
        results['Returns Prediction'] = {'status': 'SKIPPED', 'reason': 'No returns data'}
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    results['Returns Prediction'] = {'status': 'FAILED', 'error': str(e)}

# Model 6: Fraud Detection
print("\n" + "=" * 80)
print("MODEL 6: Fraud & Abuse Detection")
print("=" * 80)
try:
    from models.fraud_detection import FraudDetectionModel
    
    if sample_returns is not None and len(sample_returns) > 0:
        print("Training model...")
        fraud_model = FraudDetectionModel()
        # Merge product price into transactions
        txn_with_price = sample_transactions.merge(
            sample_products[['product_id', 'price']], 
            on='product_id', 
            how='left'
        )
        fraud_model.train(txn_with_price, sample_returns, sample_customers)
        print("✓ Model trained")
        
        print("Detecting fraud patterns...")
        test_transactions = sample_transactions.head(100).copy()
        # Merge price for fraud detection - ensure price column exists
        if 'price' not in test_transactions.columns:
            test_transactions = test_transactions.merge(
                sample_products[['product_id', 'price']], 
                on='product_id', 
                how='left'
            )
            # Fill missing prices with unit_price
            if 'unit_price' in test_transactions.columns:
                test_transactions['price'] = test_transactions['price'].fillna(test_transactions['unit_price'])
        
        fraud_predictions = fraud_model.predict_fraud_risk(
            test_transactions, sample_returns, sample_customers
        )
        print(f"✓ Analyzed {len(fraud_predictions)} transactions for fraud")
        
        if 'fraud_risk_level' in fraud_predictions.columns:
            high_risk = fraud_predictions[fraud_predictions['fraud_risk_level'].isin(['High', 'Very High'])]
            print(f"  High-risk transactions: {len(high_risk)}")
        
        print("Identifying fraud patterns...")
        fraud_patterns = fraud_model.identify_fraud_patterns(sample_transactions, sample_returns)
        print(f"✓ Identified {len(fraud_patterns)} fraud patterns")
        
        results['Fraud Detection'] = {'status': 'SUCCESS', 'predictions': len(fraud_predictions)}
    else:
        print("⚠ No returns data available, skipping")
        results['Fraud Detection'] = {'status': 'SKIPPED', 'reason': 'No returns data'}
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    results['Fraud Detection'] = {'status': 'FAILED', 'error': str(e)}

# Model 7: Omnichannel Fulfillment
print("\n" + "=" * 80)
print("MODEL 7: Omnichannel Fulfillment & Allocation")
print("=" * 80)
try:
    from models.omnichannel import OmnichannelFulfillmentModel
    
    print("Initializing model...")
    stores = sample_transactions['store_id'].unique().tolist()[:5]  # Sample stores
    omnichannel_model = OmnichannelFulfillmentModel()
    omnichannel_model.train(sample_transactions, inventory, stores)
    print("✓ Model trained")
    
    print("Optimizing fulfillment...")
    sample_orders = sample_transactions.head(5).copy()
    sample_orders['order_id'] = sample_orders['transaction_id']
    sample_orders['customer_location'] = np.random.choice(
        ['Local', 'Regional', 'National'], size=len(sample_orders)
    )
    
    allocations = omnichannel_model.optimize_allocation(
        sample_orders, inventory, stores
    )
    print(f"✓ Optimized {len(allocations)} orders")
    if len(allocations) > 0:
        print(f"  Sample allocation:\n{allocations.head(3).to_string()}")
    
    results['Omnichannel Fulfillment'] = {'status': 'SUCCESS', 'allocations': len(allocations)}
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    results['Omnichannel Fulfillment'] = {'status': 'FAILED', 'error': str(e)}

# Model 8: Merchandising Forecasting
print("\n" + "=" * 80)
print("MODEL 8: Merchandising & Buying Decision Forecasting")
print("=" * 80)
try:
    from models.merchandising import MerchandisingForecastModel
    
    print("Training model...")
    merchandising_model = MerchandisingForecastModel()
    # Use more data for trend detection
    larger_sample = sample_transactions.head(30000)
    try:
        merchandising_model.train(larger_sample, sample_products)
        print("✓ Model trained")
    except Exception as e:
        print(f"⚠ Training failed (likely insufficient data after aggregation): {e}")
        print("  This can happen with small samples - model needs sufficient time series data")
        raise
    
    print("Detecting trends...")
    trends = merchandising_model.detect_trends(sample_transactions, sample_products)
    print(f"✓ Trends detected for {len(trends)} categories")
    if len(trends) > 0:
        print(f"  Sample trends:\n{trends.head(3).to_string()}")
    
    print("Generating buying horizon forecasts...")
    categories = sample_products['category'].unique()[:2]  # Sample categories
    forecasts_generated = 0
    for category in categories:
        forecast = merchandising_model.forecast_buying_horizon(
            sample_transactions, sample_products, forecast_months=3, category=category
        )
        if forecast is not None:
            forecasts_generated += 1
    
    print(f"✓ Generated {forecasts_generated} category forecasts")
    
    results['Merchandising Forecasting'] = {'status': 'SUCCESS', 'trends': len(trends), 'forecasts': forecasts_generated}
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    results['Merchandising Forecasting'] = {'status': 'FAILED', 'error': str(e)}

# Final Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

success_count = 0
failed_count = 0
skipped_count = 0

for model_name, result in results.items():
    status = result['status']
    if status == 'SUCCESS':
        success_count += 1
        print(f"✓ {model_name}: SUCCESS")
        for key, value in result.items():
            if key != 'status':
                print(f"    - {key}: {value}")
    elif status == 'SKIPPED':
        skipped_count += 1
        print(f"⚠ {model_name}: SKIPPED - {result.get('reason', 'N/A')}")
    else:
        failed_count += 1
        print(f"✗ {model_name}: FAILED")
        if 'error' in result:
            error_msg = result['error'][:100]  # Truncate long errors
            print(f"    Error: {error_msg}...")

print("\n" + "=" * 80)
print(f"Results: {success_count} SUCCESS, {failed_count} FAILED, {skipped_count} SKIPPED")
print("=" * 80)

if failed_count == 0:
    print("\n🎉 All models tested successfully!")
    sys.exit(0)
else:
    print(f"\n⚠ {failed_count} model(s) failed. Check errors above.")
    sys.exit(1)

