"""
Demand Forecasting & Inventory Optimization Model

Implements hierarchical probabilistic forecasting for retail demand.
Addresses long-tail SKUs, seasonality, and uncertainty quantification.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Prophet not available. Using alternative methods.")


class DemandForecastingModel:
    """
    Hierarchical demand forecasting with probabilistic outputs.
    
    Key features:
    - SKU-level, category-level, and store-level forecasts
    - Probabilistic outputs (confidence intervals)
    - Handles intermittent/sparse demand
    - Seasonal pattern detection
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def prepare_features(self, transactions_df, products_df, inventory_df=None):
        """Prepare features for demand forecasting"""
        # Aggregate daily sales by product and store
        daily_sales = transactions_df.groupby(['date', 'product_id', 'store_id']).agg({
            'quantity': 'sum',
            'total_amount': 'sum'
        }).reset_index()
        
        # Merge with product features
        daily_sales = daily_sales.merge(products_df[['product_id', 'category', 'price']], 
                                       on='product_id', how='left')
        
        # Time features
        daily_sales['year'] = pd.to_datetime(daily_sales['date']).dt.year
        daily_sales['month'] = pd.to_datetime(daily_sales['date']).dt.month
        daily_sales['day_of_week'] = pd.to_datetime(daily_sales['date']).dt.dayofweek
        daily_sales['day_of_year'] = pd.to_datetime(daily_sales['date']).dt.dayofyear
        daily_sales['is_weekend'] = (daily_sales['day_of_week'] >= 5).astype(int)
        
        # Seasonal indicators
        daily_sales['is_holiday_season'] = daily_sales['month'].isin([11, 12]).astype(int)
        daily_sales['is_spring'] = daily_sales['month'].isin([3, 4, 5]).astype(int)
        daily_sales['is_summer'] = daily_sales['month'].isin([6, 7, 8]).astype(int)
        daily_sales['is_fall'] = daily_sales['month'].isin([9, 10, 11]).astype(int)
        
        # Lag features
        daily_sales = daily_sales.sort_values(['product_id', 'store_id', 'date'])
        for lag in [1, 7, 14, 30]:
            daily_sales[f'lag_{lag}'] = daily_sales.groupby(['product_id', 'store_id'])['quantity'].shift(lag)
            # Fill NaN with 0 for early periods (no history available)
            daily_sales[f'lag_{lag}'] = daily_sales[f'lag_{lag}'].fillna(0)
        
        # Rolling statistics
        daily_sales['rolling_mean_7'] = daily_sales.groupby(['product_id', 'store_id'])['quantity'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        daily_sales['rolling_std_7'] = daily_sales.groupby(['product_id', 'store_id'])['quantity'].transform(
            lambda x: x.rolling(7, min_periods=1).std().fillna(0)
        )
        
        # Fill any remaining NaN values with 0 (instead of dropping rows)
        daily_sales = daily_sales.fillna(0)
        
        # Only drop rows if critical columns are missing
        required_cols = ['quantity', 'date']
        daily_sales = daily_sales.dropna(subset=required_cols)
        
        return daily_sales
    
    def train_prophet_model(self, ts_data, product_id, store_id):
        """Train Prophet model for a specific product-store combination"""
        if not PROPHET_AVAILABLE:
            return None
            
        try:
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(ts_data['date']),
                'y': ts_data['quantity'].values
            })
            
            model.fit(prophet_df)
            return model
        except:
            return None
    
    def train_ensemble_model(self, X_train, y_train):
        """Train ensemble model (Random Forest) for demand forecasting"""
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        return model
    
    def train(self, transactions_df, products_df, inventory_df=None, 
              hierarchy_level='sku_store'):
        """
        Train forecasting models
        
        hierarchy_level: 'sku_store', 'category', or 'store'
        """
        print("Preparing features...")
        data = self.prepare_features(transactions_df, products_df, inventory_df)
        
        # Select hierarchy level
        if hierarchy_level == 'sku_store':
            group_cols = ['product_id', 'store_id']
        elif hierarchy_level == 'category':
            group_cols = ['category', 'store_id']
        elif hierarchy_level == 'store':
            group_cols = ['store_id']
        else:
            group_cols = ['product_id', 'store_id']
        
        print(f"Training models at {hierarchy_level} level...")
        
        # Feature columns
        feature_cols = ['month', 'day_of_week', 'day_of_year', 'is_weekend',
                       'is_holiday_season', 'is_spring', 'is_summer', 'is_fall',
                       'lag_1', 'lag_7', 'lag_14', 'lag_30',
                       'rolling_mean_7', 'rolling_std_7', 'price']
        
        # Train model for each group (or aggregate)
        if hierarchy_level == 'sku_store':
            # Train on aggregated data for efficiency
            print("Training aggregated model...")
            X = data[feature_cols].fillna(0)
            y = data['quantity']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = self.train_ensemble_model(X_scaled, y)
            
            self.models[hierarchy_level] = model
            self.scalers[hierarchy_level] = scaler
            self.feature_importance[hierarchy_level] = dict(zip(
                feature_cols, model.feature_importances_
            ))
        else:
            # Aggregate to hierarchy level
            agg_data = data.groupby(group_cols + ['date']).agg({
                'quantity': 'sum',
                **{col: 'mean' if col != 'category' else 'first' 
                   for col in feature_cols if col in data.columns}
            }).reset_index()
            
            X = agg_data[[col for col in feature_cols if col in agg_data.columns]].fillna(0)
            y = agg_data['quantity']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = self.train_ensemble_model(X_scaled, y)
            
            self.models[hierarchy_level] = model
            self.scalers[hierarchy_level] = scaler
        
        print("Training complete!")
        return self
    
    def predict(self, transactions_df, products_df, horizon=30, 
                hierarchy_level='sku_store', return_uncertainty=True):
        """
        Generate probabilistic forecasts
        
        Returns:
        - Point forecasts
        - Lower and upper bounds (confidence intervals)
        """
        print("Preparing prediction data...")
        data = self.prepare_features(transactions_df, products_df)
        
        # Get last date
        last_date = data['date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                    periods=horizon, freq='D')
        
        # Feature columns
        feature_cols = ['month', 'day_of_week', 'day_of_year', 'is_weekend',
                       'is_holiday_season', 'is_spring', 'is_summer', 'is_fall',
                       'lag_1', 'lag_7', 'lag_14', 'lag_30',
                       'rolling_mean_7', 'rolling_std_7', 'price']
        
        # Get unique combinations for prediction
        if hierarchy_level == 'sku_store':
            groups = data[['product_id', 'store_id', 'price']].drop_duplicates()
        elif hierarchy_level == 'category':
            groups = data[['category', 'store_id']].drop_duplicates()
        else:
            groups = data[['store_id']].drop_duplicates()
        
        forecasts = []
        
        model = self.models.get(hierarchy_level)
        scaler = self.scalers.get(hierarchy_level)
        
        if model is None or scaler is None:
            raise ValueError(f"Model not trained for {hierarchy_level}")
        
        print(f"Generating forecasts for {len(groups)} groups...")
        
        for _, group in groups.iterrows():
            # Get recent history for this group
            if hierarchy_level == 'sku_store':
                group_data = data[(data['product_id'] == group['product_id']) & 
                                 (data['store_id'] == group['store_id'])]
            elif hierarchy_level == 'category':
                group_data = data[(data['category'] == group['category']) & 
                                 (data['store_id'] == group['store_id'])]
            else:
                group_data = data[data['store_id'] == group['store_id']]
            
            if len(group_data) == 0:
                continue
            
            # Generate forecasts for each future date
            for date in future_dates:
                # Create features for this date
                features = {
                    'month': date.month,
                    'day_of_week': date.dayofweek,
                    'day_of_year': date.timetuple().tm_yday,
                    'is_weekend': 1 if date.dayofweek >= 5 else 0,
                    'is_holiday_season': 1 if date.month in [11, 12] else 0,
                    'is_spring': 1 if date.month in [3, 4, 5] else 0,
                    'is_summer': 1 if date.month in [6, 7, 8] else 0,
                    'is_fall': 1 if date.month in [9, 10, 11] else 0,
                    'price': group.get('price', group_data['price'].mean())
                }
                
                # Use recent lags (simplified - in production, would use actual recent values)
                recent_data = group_data.tail(30)
                features['lag_1'] = recent_data['quantity'].iloc[-1] if len(recent_data) > 0 else 0
                features['lag_7'] = recent_data['quantity'].tail(7).mean() if len(recent_data) >= 7 else 0
                features['lag_14'] = recent_data['quantity'].tail(14).mean() if len(recent_data) >= 14 else 0
                features['lag_30'] = recent_data['quantity'].mean() if len(recent_data) > 0 else 0
                features['rolling_mean_7'] = recent_data['quantity'].tail(7).mean() if len(recent_data) >= 7 else 0
                features['rolling_std_7'] = recent_data['quantity'].tail(7).std() if len(recent_data) >= 7 else 0
                
                # Prepare feature vector
                X_pred = np.array([[features.get(col, 0) for col in feature_cols]])
                X_pred_scaled = scaler.transform(X_pred)
                
                # Predict
                pred = model.predict(X_pred_scaled)[0]
                pred = max(0, pred)  # Demand can't be negative
                
                # Uncertainty estimation (using prediction intervals)
                # In production, would use quantile regression or bootstrap
                std_estimate = features['rolling_std_7'] if features['rolling_std_7'] > 0 else pred * 0.3
                lower_bound = max(0, pred - 1.96 * std_estimate)
                upper_bound = pred + 1.96 * std_estimate
                
                forecast_row = {
                    'date': date,
                    'forecast': pred,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
                
                # Add group identifiers
                for col in groups.columns:
                    if col != 'price':
                        forecast_row[col] = group[col]
                
                forecasts.append(forecast_row)
        
        forecasts_df = pd.DataFrame(forecasts)
        print(f"Generated {len(forecasts_df)} forecasts")
        
        return forecasts_df
    
    def get_feature_importance(self, hierarchy_level='sku_store'):
        """Get feature importance for interpretability"""
        return self.feature_importance.get(hierarchy_level, {})

