"""
Returns Prediction & Reduction Model

Predicts return probability and identifies factors leading to returns.
Focuses on size/fit modeling and customer behavior patterns.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class ReturnsPredictionModel:
    """
    Returns prediction model for retail.
    
    Key features:
    - Return probability prediction
    - Size/fit modeling
    - Customer behavior features
    - Return reason classification
    """
    
    def __init__(self):
        self.return_model = None
        self.size_fit_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def prepare_features(self, transactions_df, returns_df, products_df, customers_df):
        """Prepare features for return prediction"""
        # Merge all data
        data = transactions_df.merge(
            products_df[['product_id', 'category', 'price']],
            on='product_id', how='left'
        )
        
        # Add return indicator
        returned_transactions = set(returns_df['transaction_id'].unique())
        data['is_returned'] = data['transaction_id'].isin(returned_transactions).astype(int)
        
        # Merge return details if available
        if len(returns_df) > 0:
            return_details = returns_df[['transaction_id', 'return_reason', 'is_fraud']].drop_duplicates()
            data = data.merge(return_details, on='transaction_id', how='left')
            data['return_reason'] = data['return_reason'].fillna('Not Returned')
            data['is_fraud'] = data['is_fraud'].fillna(0)
        else:
            data['return_reason'] = 'Not Returned'
            data['is_fraud'] = 0
        
        # Customer features
        customer_features = transactions_df.groupby('customer_id').agg({
            'transaction_id': 'count',
            'total_amount': ['sum', 'mean'],
            'is_returned': 'sum' if 'is_returned' in transactions_df.columns else lambda x: 0
        }).reset_index()
        
        customer_features.columns = ['customer_id', 'total_transactions', 
                                    'total_spend', 'avg_order_value', 'historical_returns']
        
        # Calculate return rate
        customer_features['historical_return_rate'] = (
            customer_features['historical_returns'] / 
            (customer_features['total_transactions'] + 1)
        )
        
        data = data.merge(customer_features, on='customer_id', how='left')
        
        # Product features
        product_features = transactions_df.groupby('product_id').agg({
            'is_returned': lambda x: x.sum() / len(x) if 'is_returned' in transactions_df.columns else 0,
            'quantity': 'mean'
        }).reset_index()
        
        product_features.columns = ['product_id', 'product_return_rate', 'avg_quantity']
        data = data.merge(product_features, on='product_id', how='left')
        
        # Transaction features
        data['discount_pct'] = (data['price'] - data['unit_price']) / (data['price'] + 1e-6)
        data['is_discounted'] = (data['discount_pct'] > 0.05).astype(int)
        
        # Time features
        data['month'] = pd.to_datetime(data['date']).dt.month
        data['day_of_week'] = pd.to_datetime(data['date']).dt.dayofweek
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        data['is_holiday_season'] = data['month'].isin([11, 12]).astype(int)
        
        # Channel
        data['is_online'] = (data['channel'] == 'Online').astype(int)
        
        # Size-related features (if available in data)
        # In real data, would have actual size information
        data['size_mismatch_risk'] = np.random.random(len(data))  # Placeholder
        
        return data
    
    def train(self, transactions_df, returns_df, products_df, customers_df):
        """Train return prediction model"""
        print("Preparing features...")
        data = self.prepare_features(transactions_df, returns_df, products_df, customers_df)
        
        # Feature columns
        feature_cols = [
            'quantity', 'unit_price', 'discount_pct', 'is_discounted',
            'total_transactions', 'total_spend', 'avg_order_value',
            'historical_return_rate', 'product_return_rate',
            'month', 'day_of_week', 'is_weekend', 'is_holiday_season',
            'is_online', 'size_mismatch_risk'
        ]
        
        # Encode categorical
        if 'category' in data.columns:
            le_category = LabelEncoder()
            data['category_encoded'] = le_category.fit_transform(data['category'].fillna('Unknown'))
            feature_cols.append('category_encoded')
            self.label_encoders['category'] = le_category
        
        # Prepare X and y
        X = data[feature_cols].fillna(0)
        y = data['is_returned']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training return prediction model...")
        self.return_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        self.return_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.return_model.predict(X_test_scaled)
        y_pred_proba = self.return_model.predict_proba(X_test_scaled)[:, 1]
        
        print("\nModel Performance:")
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        self.feature_columns = feature_cols
        self.feature_importance = dict(zip(
            feature_cols,
            self.return_model.feature_importances_
        ))
        
        print("Training complete!")
        return self
    
    def predict_return_probability(self, transactions_df, products_df, customers_df):
        """Predict return probability for transactions"""
        if self.return_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features (without return data)
        data = transactions_df.merge(
            products_df[['product_id', 'category', 'price']],
            on='product_id', how='left'
        )
        
        # Customer features
        customer_features = transactions_df.groupby('customer_id').agg({
            'transaction_id': 'count',
            'total_amount': ['sum', 'mean']
        }).reset_index()
        
        customer_features.columns = ['customer_id', 'total_transactions', 
                                    'total_spend', 'avg_order_value']
        customer_features['historical_return_rate'] = 0  # Would use actual data
        
        data = data.merge(customer_features, on='customer_id', how='left')
        
        # Product features
        product_features = transactions_df.groupby('product_id').agg({
            'quantity': 'mean'
        }).reset_index()
        product_features.columns = ['product_id', 'avg_quantity']
        product_features['product_return_rate'] = 0  # Would use actual data
        
        data = data.merge(product_features, on='product_id', how='left')
        
        # Transaction features
        data['discount_pct'] = (data['price'] - data['unit_price']) / (data['price'] + 1e-6)
        data['is_discounted'] = (data['discount_pct'] > 0.05).astype(int)
        
        # Time features
        data['month'] = pd.to_datetime(data['date']).dt.month
        data['day_of_week'] = pd.to_datetime(data['date']).dt.dayofweek
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        data['is_holiday_season'] = data['month'].isin([11, 12]).astype(int)
        data['is_online'] = (data['channel'] == 'Online').astype(int)
        data['size_mismatch_risk'] = np.random.random(len(data))  # Placeholder
        
        # Encode category
        if 'category' in data.columns and 'category' in self.label_encoders:
            data['category_encoded'] = self.label_encoders['category'].transform(
                data['category'].fillna('Unknown')
            )
        else:
            data['category_encoded'] = 0
        
        # Prepare features
        X = data[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        return_proba = self.return_model.predict_proba(X_scaled)[:, 1]
        
        results = data[['transaction_id', 'customer_id', 'product_id', 'date']].copy()
        results['return_probability'] = return_proba
        results['return_risk'] = pd.cut(
            return_proba,
            bins=[0, 0.1, 0.3, 0.5, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        return results
    
    def identify_high_risk_transactions(self, transactions_df, products_df, customers_df, 
                                       threshold=0.3):
        """Identify transactions with high return risk"""
        predictions = self.predict_return_probability(transactions_df, products_df, customers_df)
        
        high_risk = predictions[predictions['return_probability'] > threshold].copy()
        high_risk = high_risk.sort_values('return_probability', ascending=False)
        
        return high_risk
    
    def get_feature_importance(self):
        """Get feature importance for interpretability"""
        return self.feature_importance
    
    def analyze_return_factors(self, transactions_df, returns_df, products_df):
        """Analyze factors contributing to returns"""
        data = self.prepare_features(transactions_df, returns_df, products_df, None)
        
        returned = data[data['is_returned'] == 1]
        not_returned = data[data['is_returned'] == 0]
        
        analysis = {}
        
        # Compare means
        numeric_cols = ['quantity', 'unit_price', 'discount_pct', 
                       'total_transactions', 'historical_return_rate']
        
        for col in numeric_cols:
            if col in data.columns:
                analysis[col] = {
                    'returned_mean': returned[col].mean(),
                    'not_returned_mean': not_returned[col].mean(),
                    'difference': returned[col].mean() - not_returned[col].mean()
                }
        
        # Category analysis
        if 'category' in data.columns:
            category_return_rates = data.groupby('category')['is_returned'].mean().sort_values(ascending=False)
            analysis['category_return_rates'] = category_return_rates.to_dict()
        
        # Return reasons
        if 'return_reason' in data.columns:
            reason_dist = returned['return_reason'].value_counts(normalize=True)
            analysis['return_reasons'] = reason_dist.to_dict()
        
        return analysis

