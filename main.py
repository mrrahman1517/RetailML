"""
Main script to demonstrate all retail ML models

Run this script to:
1. Generate synthetic retail data
2. Train all models
3. Generate predictions and insights
"""

import pandas as pd
import numpy as np
from utils.data_generator import RetailDataGenerator
from models.demand_forecasting import DemandForecastingModel
from models.markdown_optimization import MarkdownOptimizationModel
from models.clv_retention import CLVModel
from models.recommendations import RecommendationSystem
from models.returns_prediction import ReturnsPredictionModel
from models.fraud_detection import FraudDetectionModel
from models.omnichannel import OmnichannelFulfillmentModel
from models.merchandising import MerchandisingForecastModel


def main():
    print("=" * 80)
    print("Retail Data Science & ML Models - Demonstration")
    print("=" * 80)
    
    # Step 1: Generate synthetic data
    print("\n[Step 1] Generating synthetic retail data...")
    generator = RetailDataGenerator(seed=42)
    datasets = generator.generate_all_datasets(output_dir='data/raw')
    
    products_df = datasets['products']
    customers_df = datasets['customers']
    transactions_df = datasets['transactions']
    returns_df = datasets['returns']
    inventory_df = datasets['inventory']
    
    print(f"Generated {len(products_df)} products, {len(customers_df)} customers, "
          f"{len(transactions_df)} transactions, {len(returns_df)} returns")
    
    # Step 2: Demand Forecasting
    print("\n[Step 2] Training Demand Forecasting Model...")
    demand_model = DemandForecastingModel()
    demand_model.train(transactions_df, products_df, inventory_df, hierarchy_level='sku_store')
    
    print("Generating demand forecasts...")
    forecasts = demand_model.predict(transactions_df, products_df, horizon=30)
    print(f"Generated {len(forecasts)} forecasts")
    print(f"Sample forecast: {forecasts.head()}")
    
    # Step 3: Markdown Optimization
    print("\n[Step 3] Training Markdown Optimization Model...")
    markdown_model = MarkdownOptimizationModel()
    markdown_model.train(transactions_df, products_df)
    
    print("Generating markdown recommendations...")
    # Sample inventory for recommendations
    sample_inventory = inventory_df.groupby('product_id')['stock_level'].last().reset_index()
    markdown_recs = markdown_model.predict_optimal_markdown(
        products_df, sample_inventory, days_until_end_of_season=30
    )
    print(f"Generated {len(markdown_recs)} markdown recommendations")
    print(f"Top recommendations:\n{markdown_recs.head()}")
    
    # Step 4: CLV & Retention
    print("\n[Step 4] Training CLV & Retention Model...")
    clv_model = CLVModel()
    clv_model.train(transactions_df, customers_df)
    
    print("Calculating customer lifetime values...")
    clv_scores = clv_model.predict_customer_value(customers_df, transactions_df)
    print(f"Calculated CLV for {len(clv_scores)} customers")
    print(f"CLV Statistics:\n{clv_scores.describe()}")
    
    at_risk = clv_model.identify_at_risk_customers(threshold_probability=0.5)
    print(f"Identified {len(at_risk)} at-risk customers")
    
    # Step 5: Recommendations
    print("\n[Step 5] Training Recommendation System...")
    rec_system = RecommendationSystem(n_factors=50)
    rec_system.train(transactions_df, products_df)
    
    print("Generating recommendations for sample customers...")
    sample_customers = customers_df['customer_id'].head(5).tolist()
    for customer_id in sample_customers:
        recs = rec_system.recommend_hybrid(
            customer_id, transactions_df, products_df, n_recommendations=5
        )
        print(f"Customer {customer_id}: {len(recs)} recommendations")
    
    # Step 6: Returns Prediction
    print("\n[Step 6] Training Returns Prediction Model...")
    returns_model = ReturnsPredictionModel()
    returns_model.train(transactions_df, returns_df, products_df, customers_df)
    
    print("Predicting return probabilities...")
    sample_transactions = transactions_df.head(1000)
    return_predictions = returns_model.predict_return_probability(
        sample_transactions, products_df, customers_df
    )
    print(f"Predicted return risk for {len(return_predictions)} transactions")
    print(f"High-risk transactions: {len(return_predictions[return_predictions['return_risk'].isin(['High', 'Very High'])])}")
    
    # Step 7: Fraud Detection
    print("\n[Step 7] Training Fraud Detection Model...")
    fraud_model = FraudDetectionModel()
    fraud_model.train(transactions_df, returns_df, customers_df)
    
    print("Detecting fraud patterns...")
    fraud_predictions = fraud_model.predict_fraud_risk(
        sample_transactions, returns_df, customers_df
    )
    print(f"Analyzed {len(fraud_predictions)} transactions for fraud")
    print(f"High-risk transactions: {len(fraud_predictions[fraud_predictions['fraud_risk_level'].isin(['High', 'Very High'])])}")
    
    fraud_patterns = fraud_model.identify_fraud_patterns(transactions_df, returns_df)
    print(f"Fraud patterns identified: {list(fraud_patterns.keys())}")
    
    # Step 8: Omnichannel Fulfillment
    print("\n[Step 8] Initializing Omnichannel Fulfillment Model...")
    stores = transactions_df['store_id'].unique().tolist()[:10]  # Sample stores
    omnichannel_model = OmnichannelFulfillmentModel()
    omnichannel_model.train(transactions_df, inventory_df, stores)
    
    print("Optimizing fulfillment for sample orders...")
    sample_orders = transactions_df.head(10).copy()
    sample_orders['order_id'] = sample_orders['transaction_id']
    sample_orders['customer_location'] = np.random.choice(
        ['Local', 'Regional', 'National'], size=len(sample_orders)
    )
    
    allocations = omnichannel_model.optimize_allocation(
        sample_orders, inventory_df, stores
    )
    print(f"Optimized {len(allocations)} orders")
    
    metrics = omnichannel_model.evaluate_fulfillment_strategy(allocations, sample_orders)
    print(f"Fulfillment metrics: {metrics}")
    
    # Step 9: Merchandising Forecasting
    print("\n[Step 9] Training Merchandising Forecast Model...")
    merchandising_model = MerchandisingForecastModel()
    merchandising_model.train(transactions_df, products_df)
    
    print("Detecting trends...")
    trends = merchandising_model.detect_trends(transactions_df, products_df)
    print(f"Trends detected for {len(trends)} categories")
    print(f"Trend summary:\n{trends}")
    
    print("Generating buying horizon forecasts...")
    category_forecasts = {}
    for category in products_df['category'].unique()[:3]:  # Sample categories
        forecast = merchandising_model.forecast_buying_horizon(
            transactions_df, products_df, forecast_months=6, category=category
        )
        if forecast is not None:
            category_forecasts[category] = forecast
            risk_assessment = merchandising_model.assess_risk(forecast)
            print(f"{category}: Forecast generated with risk assessment")
    
    print("\n" + "=" * 80)
    print("All models trained and demonstrated successfully!")
    print("=" * 80)
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(f"- Products: {len(products_df)}")
    print(f"- Customers: {len(customers_df)}")
    print(f"- Transactions: {len(transactions_df)}")
    print(f"- Returns: {len(returns_df)}")
    print(f"- Return Rate: {len(returns_df) / len(transactions_df) * 100:.2f}%")
    print(f"- Average Order Value: ${transactions_df['total_amount'].mean():.2f}")
    print(f"- Total Revenue: ${transactions_df['total_amount'].sum():,.2f}")


if __name__ == "__main__":
    main()

