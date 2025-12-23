"""
Fraud & Abuse Detection Model

Detects wardrobing, return fraud, and promotion abuse.
Uses anomaly detection and behavior-based modeling with careful thresholding.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class FraudDetectionModel:
    """
    Fraud and abuse detection for retail.
    
    Key features:
    - Anomaly detection for unusual patterns
    - Behavior-based models
    - Cost-sensitive classification
    - Careful thresholding to avoid false positives
    """
    
    def __init__(self):
        self.anomaly_model = None
        self.fraud_classifier = None
        self.scaler = StandardScaler()
        self.fraud_threshold = 0.5
        
    def prepare_fraud_features(self, transactions_df, returns_df, customers_df):
        """Prepare features for fraud detection"""
        # Merge data
        data = transactions_df.copy()
        
        # Add return information
        if returns_df is not None and len(returns_df) > 0:
            return_info = returns_df.groupby('transaction_id').agg({
                'is_fraud': 'max',
                'return_date': 'min'
            }).reset_index()
            
            data = data.merge(return_info, on='transaction_id', how='left')
            data['is_fraud'] = data['is_fraud'].fillna(0)
            data['has_return'] = data['return_date'].notna().astype(int)
        else:
            data['is_fraud'] = 0
            data['has_return'] = 0
        
        # Customer behavior features
        # Ensure date is datetime
        txn_df = transactions_df.copy()
        if 'date' in txn_df.columns:
            txn_df['date'] = pd.to_datetime(txn_df['date'], errors='coerce')
        
        customer_stats = txn_df.groupby('customer_id').agg({
            'transaction_id': 'count',
            'total_amount': ['sum', 'mean', 'std'],
            'quantity': 'sum',
            'date': ['min', 'max']
        }).reset_index()
        
        customer_stats.columns = ['customer_id', 'transaction_count', 'total_spend',
                                 'avg_order_value', 'order_value_std', 'total_quantity',
                                 'first_purchase', 'last_purchase']
        
        # Ensure dates are datetime for subtraction
        customer_stats['first_purchase'] = pd.to_datetime(customer_stats['first_purchase'], errors='coerce')
        customer_stats['last_purchase'] = pd.to_datetime(customer_stats['last_purchase'], errors='coerce')
        
        customer_stats['customer_tenure'] = (
            customer_stats['last_purchase'] - customer_stats['first_purchase']
        ).dt.days.fillna(0)
        
        # Return rate
        if returns_df is not None and len(returns_df) > 0:
            customer_returns = returns_df.groupby('customer_id').agg({
                'return_id': 'count',
                'is_fraud': 'sum'
            }).reset_index()
            customer_returns.columns = ['customer_id', 'return_count', 'fraud_count']
            
            customer_stats = customer_stats.merge(customer_returns, on='customer_id', how='left')
            customer_stats['return_count'] = customer_stats['return_count'].fillna(0)
            customer_stats['fraud_count'] = customer_stats['fraud_count'].fillna(0)
            customer_stats['return_rate'] = (
                customer_stats['return_count'] / 
                (customer_stats['transaction_count'] + 1)
            )
        else:
            customer_stats['return_count'] = 0
            customer_stats['fraud_count'] = 0
            customer_stats['return_rate'] = 0
        
        data = data.merge(customer_stats, on='customer_id', how='left')
        
        # Transaction-level fraud indicators
        data['high_value_order'] = (data['total_amount'] > data['avg_order_value'] * 2).astype(int)
        data['unusual_quantity'] = (data['quantity'] > 5).astype(int)
        
        # Time-based features
        data['hour'] = pd.to_datetime(data['date']).dt.hour if 'date' in data.columns else 0
        data['is_late_night'] = ((data['hour'] >= 22) | (data['hour'] <= 4)).astype(int)
        
        # Discount abuse indicators
        if 'price' in data.columns and 'unit_price' in data.columns:
            data['high_discount'] = ((data['price'] - data['unit_price']) / (data['price'] + 1e-6) > 0.5).astype(int)
        else:
            # Calculate discount from discount_applied if available
            if 'discount_applied' in data.columns:
                data['high_discount'] = (data['discount_applied'] > 50).astype(int)
            else:
                data['high_discount'] = 0
        
        # Wardrobing indicators (purchase and return pattern)
        data['wardrobing_risk'] = (
            (data['has_return'] == 1) & 
            (data['return_rate'] > 0.3) &
            (data['transaction_count'] < 5)
        ).astype(int)
        
        # Rapid purchase pattern
        data['rapid_purchases'] = (data['transaction_count'] > 10).astype(int)
        
        return data
    
    def train_anomaly_detection(self, data):
        """Train isolation forest for anomaly detection"""
        # Features for anomaly detection
        feature_cols = [
            'total_amount', 'quantity', 'transaction_count',
            'return_rate', 'avg_order_value', 'order_value_std',
            'high_value_order', 'unusual_quantity', 'high_discount',
            'wardrobing_risk'
        ]
        
        # Filter to available columns
        available_cols = [col for col in feature_cols if col in data.columns]
        X = data[available_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train isolation forest
        self.anomaly_model = IsolationForest(
            contamination=0.05,  # Expect 5% anomalies
            random_state=42,
            n_estimators=100
        )
        
        self.anomaly_model.fit(X_scaled)
        self.anomaly_features = available_cols
        
        return self
    
    def train_fraud_classifier(self, data):
        """Train supervised fraud classifier"""
        # Features for classification
        feature_cols = [
            'total_amount', 'quantity', 'transaction_count',
            'return_rate', 'avg_order_value', 'order_value_std',
            'high_value_order', 'unusual_quantity', 'high_discount',
            'wardrobing_risk', 'rapid_purchases', 'is_late_night',
            'return_count', 'fraud_count'
        ]
        
        # Filter to available columns
        available_cols = [col for col in feature_cols if col in data.columns]
        X = data[available_cols].fillna(0)
        
        # Ensure is_fraud exists and is discrete (0/1)
        if 'is_fraud' not in data.columns:
            data['is_fraud'] = 0
        
        y = data['is_fraud']
        # Ensure y is discrete (0 or 1) - convert any continuous values to binary
        y = (y > 0).astype(int)
        
        # Only train if we have fraud examples
        if y.sum() == 0:
            print("No fraud examples in training data. Creating synthetic labels for demonstration.")
            # Create synthetic labels: mark top 5% of transactions as fraud based on risk indicators
            risk_components = []
            if 'wardrobing_risk' in data.columns:
                risk_components.append(data['wardrobing_risk'] * 3)
            if 'return_rate' in data.columns:
                risk_components.append(data['return_rate'] * 2)
            if 'high_discount' in data.columns:
                risk_components.append(data['high_discount'] * 1)
            if 'transaction_count' in data.columns:
                risk_components.append((data['transaction_count'] < 3).astype(int) * 1)
            
            if risk_components:
                risk_score = sum(risk_components)
                threshold = risk_score.quantile(0.95) if hasattr(risk_score, 'quantile') else np.percentile(risk_score, 95)
                y = (risk_score > threshold).astype(int)
            else:
                # Fallback: random sampling
                y = (np.random.random(len(data)) > 0.95).astype(int)
            data['is_fraud'] = y
        
        # Ensure y is still discrete after any modifications
        y = (y > 0).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.sum() > 0 and len(y.unique()) > 1 else None
        )
        
        # Scale features - use a separate scaler for fraud classifier
        from sklearn.preprocessing import StandardScaler
        self.fraud_scaler = StandardScaler()
        X_train_scaled = self.fraud_scaler.fit_transform(X_train)
        X_test_scaled = self.fraud_scaler.transform(X_test)
        
        # Train model (cost-sensitive: false positives are costly)
        self.fraud_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',  # Handle class imbalance
            random_state=42
        )
        
        self.fraud_classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.fraud_classifier.predict(X_test_scaled)
        y_pred_proba = self.fraud_classifier.predict_proba(X_test_scaled)[:, 1]
        
        print("\nFraud Classifier Performance:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Store feature names in the exact order used for training
        self.fraud_features = available_cols.copy()
        
        return self
    
    def train(self, transactions_df, returns_df=None, customers_df=None):
        """Train fraud detection models"""
        print("Preparing fraud features...")
        data = self.prepare_fraud_features(transactions_df, returns_df, customers_df)
        
        print("Training anomaly detection model...")
        self.train_anomaly_detection(data)
        
        print("Training fraud classifier...")
        self.train_fraud_classifier(data)
        
        print("Training complete!")
        return self
    
    def predict_fraud_risk(self, transactions_df, returns_df=None, customers_df=None):
        """Predict fraud risk for transactions"""
        if self.anomaly_model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        data = self.prepare_fraud_features(transactions_df, returns_df, customers_df)
        
        # Anomaly scores
        if hasattr(self, 'anomaly_features'):
            X_anomaly = data[self.anomaly_features].fillna(0)
            X_anomaly_scaled = self.scaler.transform(X_anomaly)
            anomaly_scores = self.anomaly_model.decision_function(X_anomaly_scaled)
            anomaly_scores_normalized = 1 / (1 + np.exp(-anomaly_scores))  # Sigmoid to [0,1]
        else:
            anomaly_scores_normalized = np.zeros(len(data))
        
        # Fraud probability (if classifier available)
        if self.fraud_classifier is not None and hasattr(self, 'fraud_features'):
            # Ensure all required features exist and add missing ones with default values
            for feature in self.fraud_features:
                if feature not in data.columns:
                    data[feature] = 0
            
            # Select features in the exact same order as training
            X_fraud = data[self.fraud_features].fillna(0)
            
            # Use the fraud scaler that was fit during training
            if hasattr(self, 'fraud_scaler'):
                X_fraud_scaled = self.fraud_scaler.transform(X_fraud)
            else:
                # Fallback to main scaler if fraud_scaler doesn't exist
                X_fraud_scaled = self.scaler.transform(X_fraud)
            
            fraud_proba = self.fraud_classifier.predict_proba(X_fraud_scaled)[:, 1]
        else:
            fraud_proba = np.zeros(len(data))
        
        # Combine scores (weighted average)
        combined_score = 0.6 * fraud_proba + 0.4 * anomaly_scores_normalized
        
        results = data[['transaction_id', 'customer_id', 'product_id', 'date', 'total_amount']].copy()
        results['fraud_risk_score'] = combined_score
        results['anomaly_score'] = anomaly_scores_normalized
        results['fraud_probability'] = fraud_proba
        results['is_fraud_predicted'] = (combined_score > self.fraud_threshold).astype(int)
        results['fraud_risk_level'] = pd.cut(
            combined_score,
            bins=[0, 0.2, 0.5, 0.8, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        return results
    
    def identify_fraud_patterns(self, transactions_df, returns_df=None):
        """Identify common fraud patterns"""
        data = self.prepare_fraud_features(transactions_df, returns_df, None)
        
        patterns = {}
        
        # Wardrobing pattern: high return rate, low transaction count
        wardrobing = data[
            (data['return_rate'] > 0.5) & 
            (data['transaction_count'] < 5)
        ]
        patterns['wardrobing'] = {
            'count': len(wardrobing),
            'avg_return_rate': wardrobing['return_rate'].mean() if len(wardrobing) > 0 else 0,
            'avg_transaction_count': wardrobing['transaction_count'].mean() if len(wardrobing) > 0 else 0
        }
        
        # Promotion abuse: high discount usage
        promo_abuse = data[data['high_discount'] == 1]
        if len(promo_abuse) > 0:
            if 'price' in promo_abuse.columns and 'unit_price' in promo_abuse.columns:
                avg_discount = ((promo_abuse['price'] - promo_abuse['unit_price']) / 
                               (promo_abuse['price'] + 1e-6)).mean()
            elif 'discount_applied' in promo_abuse.columns:
                avg_discount = promo_abuse['discount_applied'].mean()
            else:
                avg_discount = 0
        else:
            avg_discount = 0
        
        patterns['promotion_abuse'] = {
            'count': len(promo_abuse),
            'avg_discount': avg_discount
        }
        
        # Rapid purchases: many transactions in short time
        rapid = data[data['rapid_purchases'] == 1]
        patterns['rapid_purchases'] = {
            'count': len(rapid),
            'avg_transaction_count': rapid['transaction_count'].mean() if len(rapid) > 0 else 0
        }
        
        return patterns
    
    def optimize_threshold(self, transactions_df, returns_df=None, 
                          false_positive_cost=10, false_negative_cost=100):
        """
        Optimize fraud detection threshold based on cost.
        False positives damage customer trust (high cost).
        """
        predictions = self.predict_fraud_risk(transactions_df, returns_df)
        
        # Try different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        costs = []
        
        for threshold in thresholds:
            predicted_fraud = predictions['fraud_risk_score'] > threshold
            
            # In production, would use actual fraud labels
            # For now, use anomaly score as proxy
            actual_fraud = predictions['anomaly_score'] > 0.7
            
            # Calculate costs
            false_positives = (predicted_fraud & ~actual_fraud).sum()
            false_negatives = (~predicted_fraud & actual_fraud).sum()
            
            total_cost = (false_positives * false_positive_cost + 
                         false_negatives * false_negative_cost)
            
            costs.append({
                'threshold': threshold,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'total_cost': total_cost
            })
        
        costs_df = pd.DataFrame(costs)
        optimal_threshold = costs_df.loc[costs_df['total_cost'].idxmin(), 'threshold']
        
        self.fraud_threshold = optimal_threshold
        
        return costs_df, optimal_threshold

