"""
Merchandising & Buying Decision Forecasting

Forecasting for long-horizon buying decisions with trend detection
and scenario planning. Focuses on risk management over precision.
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


class MerchandisingForecastModel:
    """
    Merchandising and buying decision forecasting model.
    
    Key features:
    - Trend detection
    - Scenario forecasting
    - External signal integration
    - Risk assessment
    """
    
    def __init__(self):
        self.trend_model = None
        self.scenario_model = None
        self.scaler = StandardScaler()
        
    def detect_trends(self, transactions_df, products_df, lookback_months=12):
        """Detect trends in product categories and styles"""
        # Aggregate sales by category and month
        transactions_df['year_month'] = pd.to_datetime(transactions_df['date']).dt.to_period('M')
        
        category_trends = transactions_df.groupby(['category', 'year_month']).agg({
            'quantity': 'sum',
            'total_amount': 'sum'
        }).reset_index()
        
        # Calculate growth rates
        trends = []
        
        for category in category_trends['category'].unique():
            cat_data = category_trends[category_trends['category'] == category].sort_values('year_month')
            
            if len(cat_data) >= 2:
                # Recent vs previous period growth
                recent_qty = cat_data['quantity'].tail(3).mean()
                previous_qty = cat_data['quantity'].tail(6).head(3).mean()
                
                growth_rate = ((recent_qty - previous_qty) / (previous_qty + 1)) * 100
                
                # Trend direction
                if growth_rate > 10:
                    trend_direction = 'Rising'
                elif growth_rate < -10:
                    trend_direction = 'Declining'
                else:
                    trend_direction = 'Stable'
                
                trends.append({
                    'category': category,
                    'growth_rate': growth_rate,
                    'trend_direction': trend_direction,
                    'recent_volume': recent_qty,
                    'momentum': growth_rate
                })
        
        return pd.DataFrame(trends)
    
    def prepare_forecast_features(self, transactions_df, products_df, 
                                  external_signals=None):
        """Prepare features for long-horizon forecasting"""
        # Aggregate to monthly level
        transactions_df['year_month'] = pd.to_datetime(transactions_df['date']).dt.to_period('M')
        
        monthly_sales = transactions_df.groupby(['category', 'year_month']).agg({
            'quantity': 'sum',
            'total_amount': 'sum',
            'transaction_id': 'count'
        }).reset_index()
        
        monthly_sales.columns = ['category', 'year_month', 'quantity', 
                                'revenue', 'transaction_count']
        
        # Time features
        monthly_sales['month'] = monthly_sales['year_month'].astype(str).str[5:7].astype(int)
        monthly_sales['year'] = monthly_sales['year_month'].astype(str).str[0:4].astype(int)
        monthly_sales['quarter'] = ((monthly_sales['month'] - 1) // 3) + 1
        
        # Lag features
        monthly_sales = monthly_sales.sort_values(['category', 'year_month'])
        for lag in [1, 3, 6, 12]:
            monthly_sales[f'lag_{lag}'] = monthly_sales.groupby('category')['quantity'].shift(lag)
            # Fill NaN with 0 for early periods
            monthly_sales[f'lag_{lag}'] = monthly_sales[f'lag_{lag}'].fillna(0)
        
        # Rolling statistics
        monthly_sales['rolling_mean_3'] = monthly_sales.groupby('category')['quantity'].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        monthly_sales['rolling_mean_6'] = monthly_sales.groupby('category')['quantity'].transform(
            lambda x: x.rolling(6, min_periods=1).mean()
        )
        
        # Year-over-year growth
        monthly_sales['yoy_growth'] = monthly_sales.groupby('category')['quantity'].pct_change(12).fillna(0)
        
        # External signals (if provided)
        if external_signals is not None:
            monthly_sales = monthly_sales.merge(external_signals, on='year_month', how='left')
        
        # Fill any remaining NaN values with 0
        monthly_sales = monthly_sales.fillna(0)
        
        # Only drop rows if critical columns are missing
        required_cols = ['quantity', 'category', 'year_month']
        monthly_sales = monthly_sales.dropna(subset=required_cols)
        
        return monthly_sales
    
    def train_trend_model(self, monthly_data):
        """Train model for trend forecasting"""
        feature_cols = ['month', 'quarter', 'lag_1', 'lag_3', 'lag_6', 'lag_12',
                       'rolling_mean_3', 'rolling_mean_6', 'yoy_growth']
        
        # Filter to available columns
        available_cols = [col for col in feature_cols if col in monthly_data.columns]
        
        X = monthly_data[available_cols].fillna(0)
        y = monthly_data['quantity']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.trend_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.trend_model.fit(X_scaled, y)
        self.trend_features = available_cols
        
        return self
    
    def generate_scenarios(self, base_forecast, n_scenarios=100):
        """
        Generate scenario forecasts using Monte Carlo simulation.
        Accounts for uncertainty in long-horizon forecasts.
        """
        scenarios = []
        
        # Assume forecast uncertainty (coefficient of variation)
        cv = 0.3  # 30% coefficient of variation
        
        for i in range(n_scenarios):
            # Sample from distribution
            scenario_value = np.random.normal(
                base_forecast,
                base_forecast * cv
            )
            scenario_value = max(0, scenario_value)  # Can't be negative
            
            scenarios.append(scenario_value)
        
        scenarios = np.array(scenarios)
        
        # Calculate statistics
        scenario_stats = {
            'mean': scenarios.mean(),
            'median': np.median(scenarios),
            'p10': np.percentile(scenarios, 10),
            'p25': np.percentile(scenarios, 25),
            'p75': np.percentile(scenarios, 75),
            'p90': np.percentile(scenarios, 90),
            'std': scenarios.std()
        }
        
        return scenario_stats
    
    def forecast_buying_horizon(self, transactions_df, products_df, 
                                forecast_months=6, category=None,
                                external_signals=None):
        """
        Generate forecasts for buying horizon (months in advance).
        Returns scenario-based forecasts.
        """
        # Prepare data
        monthly_data = self.prepare_forecast_features(
            transactions_df, products_df, external_signals
        )
        
        # Filter by category if specified
        if category:
            monthly_data = monthly_data[monthly_data['category'] == category]
        
        if len(monthly_data) == 0:
            return None
        
        # Train model if not already trained
        if self.trend_model is None:
            self.train_trend_model(monthly_data)
        
        # Get latest data point
        latest = monthly_data.sort_values('year_month').iloc[-1]
        
        # Generate forecasts for each future month
        forecasts = []
        last_values = {
            'quantity': latest['quantity'],
            'lag_1': latest.get('lag_1', latest['quantity']),
            'lag_3': latest.get('lag_3', latest['quantity']),
            'lag_6': latest.get('lag_6', latest['quantity']),
            'lag_12': latest.get('lag_12', latest['quantity']),
            'rolling_mean_3': latest.get('rolling_mean_3', latest['quantity']),
            'rolling_mean_6': latest.get('rolling_mean_6', latest['quantity'])
        }
        
        # Get last year_month
        last_period = pd.Period(latest['year_month'])
        
        for i in range(1, forecast_months + 1):
            future_period = last_period + i
            future_month = future_period.month
            future_quarter = ((future_month - 1) // 3) + 1
            
            # Prepare features
            features = {
                'month': future_month,
                'quarter': future_quarter,
                'lag_1': last_values['lag_1'],
                'lag_3': last_values['lag_3'],
                'lag_6': last_values['lag_6'],
                'lag_12': last_values['lag_12'],
                'rolling_mean_3': last_values['rolling_mean_3'],
                'rolling_mean_6': last_values['rolling_mean_6'],
                'yoy_growth': latest.get('yoy_growth', 0)
            }
            
            # Prepare feature vector
            X_pred = np.array([[features.get(col, 0) for col in self.trend_features]])
            X_pred_scaled = self.scaler.transform(X_pred)
            
            # Predict
            base_forecast = self.trend_model.predict(X_pred_scaled)[0]
            base_forecast = max(0, base_forecast)
            
            # Generate scenarios
            scenario_stats = self.generate_scenarios(base_forecast)
            
            forecasts.append({
                'year_month': str(future_period),
                'month': future_month,
                'base_forecast': base_forecast,
                **scenario_stats
            })
            
            # Update lag values for next iteration
            last_values['lag_12'] = last_values['lag_6']
            last_values['lag_6'] = last_values['lag_3']
            last_values['lag_3'] = last_values['lag_1']
            last_values['lag_1'] = base_forecast
            last_values['rolling_mean_3'] = (
                (last_values['rolling_mean_3'] * 2 + base_forecast) / 3
            )
            last_values['rolling_mean_6'] = (
                (last_values['rolling_mean_6'] * 5 + base_forecast) / 6
            )
        
        return pd.DataFrame(forecasts)
    
    def assess_risk(self, forecasts_df):
        """Assess risk in buying decisions based on forecast uncertainty"""
        if forecasts_df is None or len(forecasts_df) == 0:
            return None
        
        # Calculate risk metrics
        forecasts_df['uncertainty_range'] = forecasts_df['p90'] - forecasts_df['p10']
        forecasts_df['uncertainty_pct'] = (
            forecasts_df['uncertainty_range'] / (forecasts_df['mean'] + 1)
        ) * 100
        
        # Risk levels
        forecasts_df['risk_level'] = pd.cut(
            forecasts_df['uncertainty_pct'],
            bins=[0, 20, 40, 60, 100],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Downside risk (p10 vs mean)
        forecasts_df['downside_risk'] = (
            (forecasts_df['mean'] - forecasts_df['p10']) / (forecasts_df['mean'] + 1)
        ) * 100
        
        return forecasts_df
    
    def train(self, transactions_df, products_df, external_signals=None):
        """Train merchandising forecast model"""
        print("Preparing forecast features...")
        monthly_data = self.prepare_forecast_features(
            transactions_df, products_df, external_signals
        )
        
        print("Training trend model...")
        self.train_trend_model(monthly_data)
        
        print("Training complete!")
        return self

