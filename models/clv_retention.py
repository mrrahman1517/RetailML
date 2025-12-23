"""
Customer Lifetime Value (CLV) & Retention Model

Implements survival analysis and sequence modeling for CLV prediction.
Focuses on expected future value and retention intervention targeting.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    print("Lifelines not available. Using alternative methods.")


class CLVModel:
    """
    Customer Lifetime Value and Retention Model.
    
    Key features:
    - Survival analysis for churn prediction
    - Sequence modeling for purchase patterns
    - Segment-specific CLV calculation
    - Expected future value estimation
    """
    
    def __init__(self):
        self.clv_model = None
        self.churn_model = None
        self.scaler = StandardScaler()
        self.kmf = None
        
    def prepare_customer_features(self, transactions_df, customers_df):
        """Prepare customer-level features for CLV modeling"""
        # Calculate RFM metrics
        customer_metrics = transactions_df.groupby('customer_id').agg({
            'date': ['max', 'min', 'count'],
            'total_amount': ['sum', 'mean'],
            'quantity': 'sum'
        }).reset_index()
        
        customer_metrics.columns = ['customer_id', 'last_purchase', 'first_purchase', 
                                   'frequency', 'monetary_total', 'monetary_avg', 'quantity_total']
        
        # Calculate recency (days since last purchase)
        max_date = transactions_df['date'].max()
        customer_metrics['recency'] = (max_date - customer_metrics['last_purchase']).dt.days
        
        # Calculate tenure (days since first purchase)
        customer_metrics['tenure'] = (customer_metrics['last_purchase'] - 
                                      customer_metrics['first_purchase']).dt.days
        
        # Average days between purchases
        customer_metrics['avg_days_between'] = customer_metrics['tenure'] / (customer_metrics['frequency'] + 1)
        
        # Category diversity
        category_diversity = transactions_df.groupby('customer_id')['category'].nunique().reset_index()
        category_diversity.columns = ['customer_id', 'category_diversity']
        customer_metrics = customer_metrics.merge(category_diversity, on='customer_id', how='left')
        
        # Channel preference
        channel_pref = transactions_df.groupby('customer_id')['channel'].apply(
            lambda x: (x == 'Online').sum() / len(x)
        ).reset_index()
        channel_pref.columns = ['customer_id', 'online_ratio']
        customer_metrics = customer_metrics.merge(channel_pref, on='customer_id', how='left')
        
        # Merge with customer segment
        customer_metrics = customer_metrics.merge(
            customers_df[['customer_id', 'segment']], 
            on='customer_id', how='left'
        )
        
        return customer_metrics
    
    def calculate_clv_components(self, customer_metrics, transactions_df):
        """Calculate CLV components: historical value, predicted future value"""
        # Historical CLV (already realized)
        customer_metrics['historical_clv'] = customer_metrics['monetary_total']
        
        # Predict future value based on patterns
        # Simple approach: extrapolate based on frequency and average order value
        # In production, would use more sophisticated models
        
        # Estimate remaining lifetime (simplified)
        # High-value customers likely to continue longer
        customer_metrics['estimated_months_remaining'] = customer_metrics.apply(
            lambda row: 24 if row['segment'] == 'High-Value' 
                       else 12 if row['segment'] == 'Medium' 
                       else 6, axis=1
        )
        
        # Predicted future value
        customer_metrics['predicted_future_value'] = (
            customer_metrics['frequency'] * 
            customer_metrics['monetary_avg'] * 
            customer_metrics['estimated_months_remaining'] / 
            (customer_metrics['avg_days_between'] / 30 + 1)
        )
        
        # Total CLV
        customer_metrics['total_clv'] = (
            customer_metrics['historical_clv'] + 
            customer_metrics['predicted_future_value']
        )
        
        return customer_metrics
    
    def train_churn_model(self, customer_metrics, transactions_df, churn_threshold_days=90):
        """Train model to predict churn probability"""
        # Define churn: no purchase in last N days
        max_date = transactions_df['date'].max()
        customer_metrics['is_churned'] = (
            customer_metrics['recency'] > churn_threshold_days
        ).astype(int)
        
        # Features for churn prediction
        feature_cols = ['recency', 'frequency', 'monetary_avg', 'tenure',
                       'avg_days_between', 'category_diversity', 'online_ratio']
        
        # One-hot encode segment
        segments_encoded = pd.get_dummies(customer_metrics['segment'], prefix='segment')
        X = pd.concat([
            customer_metrics[feature_cols].fillna(0),
            segments_encoded
        ], axis=1)
        
        y = customer_metrics['is_churned']
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.churn_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        self.churn_model.fit(X_train, y_train)
        self.churn_features = X.columns.tolist()
        
        return self
    
    def train_survival_model(self, customer_metrics, transactions_df):
        """Train survival analysis model for time-to-churn"""
        if not LIFELINES_AVAILABLE:
            return self
        
        # Prepare survival data
        max_date = transactions_df['date'].max()
        customer_metrics['T'] = customer_metrics['recency']  # Time observed
        customer_metrics['E'] = (customer_metrics['recency'] > 90).astype(int)  # Event (churn)
        
        # Fit Kaplan-Meier
        self.kmf = KaplanMeierFitter()
        self.kmf.fit(customer_metrics['T'], customer_metrics['E'])
        
        # Fit Cox proportional hazards if enough data
        if len(customer_metrics) > 100:
            try:
                cox_data = customer_metrics[['T', 'E', 'frequency', 'monetary_avg', 
                                            'recency', 'tenure']].copy()
                cox_data = cox_data.fillna(0)
                
                self.cox_model = CoxPHFitter()
                self.cox_model.fit(cox_data, duration_col='T', event_col='E')
            except:
                pass
        
        return self
    
    def train(self, transactions_df, customers_df):
        """Train CLV and retention models"""
        print("Preparing customer features...")
        customer_metrics = self.prepare_customer_features(transactions_df, customers_df)
        
        print("Calculating CLV components...")
        customer_metrics = self.calculate_clv_components(customer_metrics, transactions_df)
        
        print("Training churn model...")
        self.train_churn_model(customer_metrics, transactions_df)
        
        print("Training survival model...")
        self.train_survival_model(customer_metrics, transactions_df)
        
        self.customer_metrics = customer_metrics
        
        print("Training complete!")
        return self
    
    def predict_customer_value(self, customers_df=None, transactions_df=None):
        """Predict CLV for customers"""
        if customers_df is not None and transactions_df is not None:
            # Recalculate metrics
            customer_metrics = self.prepare_customer_features(transactions_df, customers_df)
            customer_metrics = self.calculate_clv_components(customer_metrics, transactions_df)
        else:
            # Use stored metrics
            customer_metrics = self.customer_metrics.copy()
        
        # Predict churn probability
        if self.churn_model is not None:
            feature_cols = ['recency', 'frequency', 'monetary_avg', 'tenure',
                           'avg_days_between', 'category_diversity', 'online_ratio']
            
            segments_encoded = pd.get_dummies(customer_metrics['segment'], prefix='segment')
            X = pd.concat([
                customer_metrics[feature_cols].fillna(0),
                segments_encoded
            ], axis=1)
            
            # Ensure all features match training
            for col in self.churn_features:
                if col not in X.columns:
                    X[col] = 0
            
            X = X[self.churn_features]
            
            customer_metrics['churn_probability'] = self.churn_model.predict(X)
        else:
            customer_metrics['churn_probability'] = 0.5
        
        # Calculate risk-adjusted CLV
        customer_metrics['risk_adjusted_clv'] = (
            customer_metrics['predicted_future_value'] * 
            (1 - customer_metrics['churn_probability'])
        )
        
        # Segment customers
        customer_metrics['clv_segment'] = pd.cut(
            customer_metrics['total_clv'],
            bins=[0, 100, 500, 2000, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        return customer_metrics[['customer_id', 'segment', 'historical_clv', 
                                'predicted_future_value', 'total_clv',
                                'risk_adjusted_clv', 'churn_probability', 'clv_segment']]
    
    def identify_at_risk_customers(self, threshold_probability=0.5):
        """Identify customers at risk of churning"""
        if self.customer_metrics is None:
            raise ValueError("Model not trained. Call train() first.")
        
        at_risk = self.customer_metrics[
            (self.customer_metrics['churn_probability'] > threshold_probability) &
            (self.customer_metrics['total_clv'] > 100)  # Focus on valuable customers
        ].copy()
        
        at_risk = at_risk.sort_values('total_clv', ascending=False)
        
        return at_risk[['customer_id', 'segment', 'total_clv', 'churn_probability', 
                        'recency', 'frequency', 'monetary_avg']]
    
    def estimate_intervention_impact(self, customer_id, intervention_type='retention_campaign'):
        """
        Estimate impact of retention intervention on customer value.
        """
        if self.customer_metrics is None:
            raise ValueError("Model not trained. Call train() first.")
        
        customer = self.customer_metrics[
            self.customer_metrics['customer_id'] == customer_id
        ]
        
        if len(customer) == 0:
            return None
        
        customer = customer.iloc[0]
        current_clv = customer['total_clv']
        current_churn_prob = customer['churn_probability']
        
        # Estimate intervention impact
        if intervention_type == 'retention_campaign':
            # Reduce churn probability by 20-40%
            churn_reduction = np.random.uniform(0.2, 0.4)
            new_churn_prob = current_churn_prob * (1 - churn_reduction)
            
            # Increase future value by extending lifetime
            lifetime_extension = 1.2  # 20% longer lifetime
            new_future_value = customer['predicted_future_value'] * lifetime_extension
            
            new_clv = customer['historical_clv'] + (
                new_future_value * (1 - new_churn_prob)
            )
            
            impact = new_clv - current_clv
            
            return {
                'customer_id': customer_id,
                'current_clv': current_clv,
                'new_clv': new_clv,
                'impact': impact,
                'impact_pct': (impact / current_clv) * 100,
                'current_churn_prob': current_churn_prob,
                'new_churn_prob': new_churn_prob,
                'intervention_type': intervention_type
            }
        
        return None

