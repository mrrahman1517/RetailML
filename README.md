# Retail Data Science & ML Models

This repository contains machine learning models for key retail problems relevant to fashion retailers like Nordstrom. Each model addresses a specific business challenge with both technical rigor and business impact focus.

## Problems Addressed

1. **Demand Forecasting & Inventory Optimization** - Probabilistic forecasting for long-tail SKUs with seasonal patterns
2. **Markdown Optimization & Pricing Strategy** - Causal ML for optimal markdown timing
3. **Customer Lifetime Value (CLV) & Retention** - Survival analysis and sequence modeling
4. **Personalization & Recommendation Systems** - Collaborative filtering with cold-start handling
5. **Returns Prediction & Reduction** - Classification models for return probability
6. **Fraud & Abuse Detection** - Anomaly detection for wardrobing and return fraud
7. **Omnichannel Fulfillment** - Optimization under uncertainty
8. **Merchandising Forecasting** - Trend detection and scenario planning

## Dataset Sources

The project uses synthetic retail datasets that simulate real-world retail scenarios. Key characteristics:

- **Sales Transactions**: Multi-store, multi-SKU time series with seasonality
- **Customer Data**: Purchase history, demographics, and behavior patterns
- **Product Data**: Categories, prices, sizes, and attributes
- **Returns Data**: Return patterns with reasons and timing
- **Inventory Data**: Stock levels, replenishment, and stockout events

### Real-World Dataset References

For production use, consider these publicly available datasets:

- **FreshRetailNet-50K**: 50,000 store-product time series with hourly sales data
- **Amazon-M2**: Multilingual shopping session dataset for recommendations
- **Omnichannel Retail Datasets**: Product categorization and customer arrival patterns
- **UCI Machine Learning Repository**: Various retail transaction datasets
- **Kaggle**: Retail sales competitions and datasets

## Project Structure

```
RetailDS/
├── data/
│   ├── raw/              # Raw synthetic datasets
│   └── processed/        # Processed features
├── models/
│   ├── demand_forecasting.py
│   ├── markdown_optimization.py
│   ├── clv_retention.py
│   ├── recommendations.py
│   ├── returns_prediction.py
│   ├── fraud_detection.py
│   ├── omnichannel.py
│   └── merchandising.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── utils/
│   ├── data_generator.py
│   └── evaluation.py
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Generate Synthetic Data

```python
from utils.data_generator import RetailDataGenerator

generator = RetailDataGenerator()
generator.generate_all_datasets()
```

### Run Models

Each model can be run independently:

```python
# Demand Forecasting
from models.demand_forecasting import DemandForecastingModel
model = DemandForecastingModel()
model.train()
forecasts = model.predict(horizon=30)

# CLV & Retention
from models.clv_retention import CLVModel
clv_model = CLVModel()
clv_model.train()
clv_scores = clv_model.predict_customer_value()
```

## Model Details

### 1. Demand Forecasting & Inventory Optimization

**Approach**: Hierarchical probabilistic forecasting using:
- Prophet for trend and seasonality
- XGBoost for feature engineering
- Bayesian methods for uncertainty quantification

**Key Features**:
- SKU-level, category-level, and store-level forecasts
- Probabilistic outputs (confidence intervals)
- Intermittent demand handling

### 2. Markdown Optimization & Pricing Strategy

**Approach**: Causal inference and price elasticity modeling:
- Uplift modeling to estimate treatment effects
- Price elasticity regression
- Counterfactual analysis

**Key Features**:
- Optimal markdown timing
- Price sensitivity by segment
- Revenue impact estimation

### 3. Customer Lifetime Value & Retention

**Approach**: Survival analysis and sequence modeling:
- Cox proportional hazards model
- LSTM for purchase sequences
- Segment-specific CLV calculation

**Key Features**:
- Churn probability prediction
- Expected future value
- Retention intervention targeting

### 4. Personalization & Recommendation Systems

**Approach**: Hybrid collaborative filtering:
- Matrix factorization (implicit feedback)
- Content-based filtering
- Cold-start handling

**Key Features**:
- User-item recommendations
- New product recommendations
- Seasonal trend incorporation

### 5. Returns Prediction & Reduction

**Approach**: Classification with feature engineering:
- Gradient boosting for return probability
- Size/fit modeling
- Customer behavior features

**Key Features**:
- Return risk scoring
- Size recommendation improvement
- Intervention targeting

### 6. Fraud & Abuse Detection

**Approach**: Anomaly detection and behavior modeling:
- Isolation Forest for anomalies
- Behavioral clustering
- Cost-sensitive classification

**Key Features**:
- Fraud score calculation
- Pattern detection
- Threshold optimization

### 7. Omnichannel Fulfillment

**Approach**: Optimization under uncertainty:
- Demand prediction integration
- Cost optimization
- Constraint satisfaction

**Key Features**:
- Fulfillment routing
- Inventory allocation
- Cost-time tradeoffs

### 8. Merchandising Forecasting

**Approach**: Trend detection and scenario planning:
- Time series decomposition
- External signal integration
- Monte Carlo scenario generation

**Key Features**:
- Trend identification
- Scenario forecasting
- Risk assessment

## License

This project is for educational and demonstration purposes.

## Contributing

This is a demonstration project. For production use, ensure proper data governance, model validation, and compliance with relevant regulations.

