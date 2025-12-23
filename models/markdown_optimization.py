"""
Markdown Optimization & Pricing Strategy Model

Implements causal ML for optimal markdown timing and pricing decisions.
Uses price elasticity modeling, uplift modeling, and counterfactual analysis.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class MarkdownOptimizationModel:
    """
    Causal ML model for markdown optimization.
    
    Key features:
    - Price elasticity estimation
    - Uplift modeling (treatment effect estimation)
    - Counterfactual analysis
    - Optimal markdown timing
    """
    
    def __init__(self):
        self.price_elasticity_model = None
        self.uplift_model = None
        self.scaler = StandardScaler()
        
    def calculate_price_elasticity(self, transactions_df, products_df):
        """Estimate price elasticity of demand"""
        # Merge transactions with products
        data = transactions_df.merge(
            products_df[['product_id', 'price']].rename(columns={'price': 'original_price'}),
            on='product_id', how='left'
        )
        
        # Calculate effective price (with discounts)
        data['effective_price'] = data['unit_price']
        data['discount_pct'] = (data['original_price'] - data['effective_price']) / data['original_price']
        
        # Aggregate by product and discount level
        elasticity_data = []
        
        for product_id in data['product_id'].unique():
            product_data = data[data['product_id'] == product_id]
            
            # Group by discount buckets
            product_data['discount_bucket'] = pd.cut(
                product_data['discount_pct'],
                bins=[-0.01, 0.01, 0.15, 0.30, 0.50, 1.0],
                labels=['No Discount', 'Small (0-15%)', 'Medium (15-30%)', 
                       'Large (30-50%)', 'Very Large (50%+)']
            )
            
            for bucket in product_data['discount_bucket'].cat.categories:
                bucket_data = product_data[product_data['discount_bucket'] == bucket]
                if len(bucket_data) > 0:
                    avg_price = bucket_data['effective_price'].mean()
                    avg_quantity = bucket_data['quantity'].mean()
                    total_quantity = bucket_data['quantity'].sum()
                    n_transactions = len(bucket_data)
                    
                    elasticity_data.append({
                        'product_id': product_id,
                        'discount_bucket': bucket,
                        'avg_price': avg_price,
                        'avg_quantity': avg_quantity,
                        'total_quantity': total_quantity,
                        'n_transactions': n_transactions
                    })
        
        elasticity_df = pd.DataFrame(elasticity_data)
        
        # Calculate elasticity for each product
        product_elasticities = []
        
        for product_id in elasticity_df['product_id'].unique():
            product_el = elasticity_df[elasticity_df['product_id'] == product_id].sort_values('avg_price')
            
            if len(product_el) >= 2:
                # Simple elasticity calculation: % change in quantity / % change in price
                prices = product_el['avg_price'].values
                quantities = product_el['total_quantity'].values
                
                # Calculate elasticity
                price_changes = np.diff(prices) / prices[:-1]
                quantity_changes = np.diff(quantities) / (quantities[:-1] + 1)
                
                # Average elasticity (weighted by transaction volume)
                elasticities = quantity_changes / (price_changes + 1e-6)
                weights = quantities[:-1]
                
                if len(elasticities) > 0 and np.sum(weights) > 0:
                    weighted_elasticity = np.average(elasticities, weights=weights)
                    product_elasticities.append({
                        'product_id': product_id,
                        'price_elasticity': weighted_elasticity,
                        'avg_price': product_el['avg_price'].mean()
                    })
        
        return pd.DataFrame(product_elasticities)
    
    def train_price_elasticity_model(self, transactions_df, products_df):
        """Train model to predict demand as function of price"""
        # Prepare data
        data = transactions_df.merge(products_df, on='product_id', how='left')
        
        # Ensure category exists
        if 'category' not in data.columns:
            if 'category' in products_df.columns:
                data = data.merge(products_df[['product_id', 'category']], on='product_id', how='left', suffixes=('', '_prod'))
                if 'category_prod' in data.columns:
                    data['category'] = data['category_prod']
                    data = data.drop(columns=['category_prod'])
            else:
                data['category'] = 'General'
        
        # Ensure month exists
        if 'month' not in data.columns:
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'], errors='coerce')
                data['month'] = data['date'].dt.month.fillna(1)
            else:
                data['month'] = 1
        
        data['discount_pct'] = (data['price'] - data['unit_price']) / (data['price'] + 1e-6)
        data['log_price'] = np.log(data['unit_price'] + 1)
        data['log_quantity'] = np.log(data['quantity'] + 1)
        
        # Features - only include columns that exist
        base_features = ['log_price', 'discount_pct']
        categorical_features = []
        
        if 'category' in data.columns:
            categorical_features.append('category')
        if 'month' in data.columns:
            categorical_features.append('month')
        
        feature_cols = base_features + categorical_features
        X = data[base_features].copy()
        
        # Add dummy variables for categorical features
        if categorical_features:
            dummies = pd.get_dummies(data[categorical_features], drop_first=True)
            X = pd.concat([X, dummies], axis=1)
        
        # Fill any NaN values before training
        X = X.fillna(0)
        
        y = data['quantity'].fillna(0)
        
        # Train model
        self.price_elasticity_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.price_elasticity_model.fit(X_train, y_train)
        
        # Store feature columns for prediction
        self.feature_columns = X.columns.tolist()
        
        return self
    
    def estimate_uplift(self, transactions_df, products_df):
        """
        Estimate uplift (treatment effect) of markdowns.
        Uses difference-in-differences approach.
        """
        # Prepare data
        data = transactions_df.merge(products_df, on='product_id', how='left')
        data['has_markdown'] = (data['unit_price'] < data['price'] * 0.95).astype(int)
        data['discount_pct'] = (data['price'] - data['unit_price']) / data['price']
        
        # Group by product and time period
        data['week'] = pd.to_datetime(data['date']).dt.to_period('W')
        
        uplift_data = []
        
        for product_id in data['product_id'].unique():
            product_data = data[data['product_id'] == product_id].copy()
            
            # Compare periods with and without markdowns
            markdown_periods = product_data[product_data['has_markdown'] == 1]['week'].unique()
            normal_periods = product_data[product_data['has_markdown'] == 0]['week'].unique()
            
            if len(markdown_periods) > 0 and len(normal_periods) > 0:
                # Average sales in markdown vs normal periods
                markdown_sales = product_data[product_data['has_markdown'] == 1]['quantity'].mean()
                normal_sales = product_data[product_data['has_markdown'] == 0]['quantity'].mean()
                
                # Uplift
                uplift = markdown_sales - normal_sales
                uplift_pct = (uplift / (normal_sales + 1)) * 100
                
                # Average discount in markdown periods
                avg_discount = product_data[product_data['has_markdown'] == 1]['discount_pct'].mean()
                
                uplift_data.append({
                    'product_id': product_id,
                    'uplift': uplift,
                    'uplift_pct': uplift_pct,
                    'avg_discount': avg_discount,
                    'normal_sales': normal_sales,
                    'markdown_sales': markdown_sales
                })
        
        return pd.DataFrame(uplift_data)
    
    def train(self, transactions_df, products_df):
        """Train markdown optimization models"""
        print("Training price elasticity model...")
        self.train_price_elasticity_model(transactions_df, products_df)
        
        print("Calculating price elasticities...")
        self.elasticities = self.calculate_price_elasticity(transactions_df, products_df)
        
        print("Estimating uplift effects...")
        self.uplift_effects = self.estimate_uplift(transactions_df, products_df)
        
        print("Training complete!")
        return self
    
    def predict_optimal_markdown(self, products_df, current_inventory, 
                                 days_until_end_of_season=30, 
                                 holding_cost_rate=0.001):
        """
        Predict optimal markdown strategy for products.
        
        Considers:
        - Price elasticity
        - Uplift effects
        - Inventory levels
        - Time until season end
        - Holding costs
        """
        recommendations = []
        
        # Merge with elasticity and uplift data
        product_data = products_df.merge(
            self.elasticities, on='product_id', how='left'
        ).merge(
            self.uplift_effects, on='product_id', how='left'
        )
        
        # Merge with inventory
        if current_inventory is not None:
            product_data = product_data.merge(
                current_inventory[['product_id', 'stock_level']],
                on='product_id', how='left'
            )
            product_data['stock_level'] = product_data['stock_level'].fillna(0)
        else:
            product_data['stock_level'] = 0
        
        for _, product in product_data.iterrows():
            current_price = product['price']
            elasticity = product.get('price_elasticity', -1.5)  # Default elasticity
            uplift_pct = product.get('uplift_pct', 0)
            stock_level = product.get('stock_level', 0)
            
            # Calculate expected revenue for different markdown levels
            markdown_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            best_markdown = 0.0
            best_expected_revenue = 0
            
            for markdown_pct in markdown_options:
                new_price = current_price * (1 - markdown_pct)
                
                # Estimate demand increase (using elasticity)
                if elasticity < 0:  # Normal goods have negative elasticity
                    price_change_pct = (new_price - current_price) / current_price
                    demand_change_pct = elasticity * price_change_pct
                else:
                    # Use uplift if available
                    demand_change_pct = uplift_pct / 100 if markdown_pct > 0 else 0
                
                # Base demand (simplified - in production would use forecast)
                base_demand = 10  # Placeholder
                expected_demand = base_demand * (1 + demand_change_pct)
                
                # Expected revenue
                expected_revenue = new_price * min(expected_demand, stock_level)
                
                # Holding cost (if inventory remains)
                remaining_inventory = max(0, stock_level - expected_demand)
                holding_cost = remaining_inventory * current_price * holding_cost_rate * days_until_end_of_season
                
                # Net expected value
                net_value = expected_revenue - holding_cost
                
                if net_value > best_expected_revenue:
                    best_expected_revenue = net_value
                    best_markdown = markdown_pct
            
            # Recommendation logic
            if stock_level > 50 and days_until_end_of_season < 30:
                urgency = 'High'
            elif stock_level > 20:
                urgency = 'Medium'
            else:
                urgency = 'Low'
            
            recommendations.append({
                'product_id': product['product_id'],
                'category': product['category'],
                'current_price': current_price,
                'recommended_markdown': best_markdown,
                'recommended_price': current_price * (1 - best_markdown),
                'expected_revenue': best_expected_revenue,
                'stock_level': stock_level,
                'urgency': urgency,
                'price_elasticity': elasticity,
                'uplift_pct': uplift_pct
            })
        
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df = recommendations_df.sort_values('expected_revenue', ascending=False)
        
        return recommendations_df
    
    def counterfactual_analysis(self, product_id, transactions_df, products_df, 
                                markdown_scenarios=[0.1, 0.2, 0.3, 0.4, 0.5]):
        """
        Counterfactual analysis: What would happen under different markdown scenarios?
        """
        product = products_df[products_df['product_id'] == product_id].iloc[0]
        current_price = product['price']
        
        # Get historical data
        product_transactions = transactions_df[transactions_df['product_id'] == product_id]
        avg_quantity = product_transactions['quantity'].mean()
        
        # Get elasticity
        elasticity = self.elasticities[
            self.elasticities['product_id'] == product_id
        ]['price_elasticity'].values
        
        if len(elasticity) > 0:
            elasticity = elasticity[0]
        else:
            elasticity = -1.5  # Default
        
        scenarios = []
        
        for markdown_pct in markdown_scenarios:
            new_price = current_price * (1 - markdown_pct)
            price_change_pct = (new_price - current_price) / current_price
            
            # Estimate demand change
            if elasticity < 0:
                demand_change_pct = elasticity * price_change_pct
            else:
                demand_change_pct = 0
            
            expected_quantity = avg_quantity * (1 + demand_change_pct)
            expected_revenue = new_price * expected_quantity
            revenue_change = expected_revenue - (current_price * avg_quantity)
            
            scenarios.append({
                'markdown_pct': markdown_pct,
                'new_price': new_price,
                'expected_quantity': expected_quantity,
                'expected_revenue': expected_revenue,
                'revenue_change': revenue_change,
                'revenue_change_pct': (revenue_change / (current_price * avg_quantity + 1)) * 100
            })
        
        return pd.DataFrame(scenarios)

