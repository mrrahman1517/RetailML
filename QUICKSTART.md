# Quick Start Guide - Retail Data Science & ML Models

This guide will help you quickly get started with the retail ML models for Nordstrom-relevant problems.

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd RetailDS
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Note: Some packages like Prophet may require additional system dependencies. If you encounter issues:
   - **Prophet**: May require `pystan` and system compilers
   - **TensorFlow/PyTorch**: Install based on your system (CPU/GPU)

## Quick Start Options

### Option 1: Run the Complete Pipeline (Recommended)

Run all models with synthetic data:

```bash
python main.py
```

This will:
1. Generate synthetic retail datasets
2. Train all 8 ML models
3. Generate predictions and insights
4. Display summary statistics

### Option 2: Acquire Real Datasets

Download and prepare real retail datasets:

```bash
python -m utils.dataset_acquisition
```

This will attempt to download the UCI Online Retail dataset. If that fails, it will generate synthetic data as a fallback.

### Option 3: Interactive Exploration (Jupyter Notebook)

For detailed exploration and visualization:

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

## Dataset Sources

### Real Datasets (Recommended for Production)

1. **UCI Online Retail Dataset**
   - Automatically downloaded by `dataset_acquisition.py`
   - Real UK-based online retail transactions
   - URL: https://archive.ics.uci.edu/ml/datasets/Online+Retail

2. **Kaggle Datasets** (Manual download)
   - Walmart Sales Forecasting
   - Superstore Sales
   - Requires Kaggle API setup

3. **Synthetic Data** (Default)
   - Generated automatically if real data unavailable
   - Simulates fashion retail scenarios
   - Includes seasonality, long-tail SKUs, customer segments

### Using Your Own Data

To use your own retail data, ensure your datasets have these columns:

**Transactions:**
- `date` (datetime)
- `customer_id` (string)
- `product_id` (string)
- `quantity` (numeric)
- `total_amount` or `unit_price` (numeric)
- `store_id` (string, optional)
- `category` (string, optional)

**Products:**
- `product_id` (string)
- `category` (string)
- `price` (numeric)

**Customers:**
- `customer_id` (string)
- Additional features (segment, demographics, etc.)

## Running Individual Models

### 1. Demand Forecasting

```python
from models.demand_forecasting import DemandForecastingModel
from utils.dataset_acquisition import RetailDatasetAcquisition

# Load data
acquirer = RetailDatasetAcquisition()
datasets = acquirer.load_dataset('synthetic')  # or 'online_retail'

# Train model
model = DemandForecastingModel()
model.train(datasets['transactions'], datasets['products'], 
            datasets.get('inventory'), hierarchy_level='sku_store')

# Generate forecasts
forecasts = model.predict(datasets['transactions'], datasets['products'], 
                         horizon=30)
print(forecasts.head())
```

### 2. Customer Lifetime Value

```python
from models.clv_retention import CLVModel

model = CLVModel()
model.train(datasets['transactions'], datasets['customers'])

clv_scores = model.predict_customer_value(datasets['customers'], 
                                          datasets['transactions'])
at_risk = model.identify_at_risk_customers(threshold_probability=0.5)
```

### 3. Markdown Optimization

```python
from models.markdown_optimization import MarkdownOptimizationModel

model = MarkdownOptimizationModel()
model.train(datasets['transactions'], datasets['products'])

# Get inventory levels
inventory = datasets['inventory'].groupby('product_id')['stock_level'].last().reset_index()

# Generate recommendations
recommendations = model.predict_optimal_markdown(
    datasets['products'], inventory, days_until_end_of_season=30
)
```

### 4. Recommendation System

```python
from models.recommendations import RecommendationSystem

rec_system = RecommendationSystem(n_factors=50)
rec_system.train(datasets['transactions'], datasets['products'])

# Get recommendations for a customer
recommendations = rec_system.recommend_hybrid(
    customer_id='CUST_000001',
    transactions_df=datasets['transactions'],
    products_df=datasets['products'],
    n_recommendations=10
)
```

## Model Outputs

### Demand Forecasting
- Point forecasts with confidence intervals
- Hierarchical forecasts (SKU → Category → Store)
- Feature importance for interpretability

### CLV & Retention
- Predicted customer lifetime value
- Churn probability scores
- At-risk customer identification

### Markdown Optimization
- Optimal discount recommendations
- Timing recommendations
- Expected revenue impact

### Recommendations
- Personalized product recommendations
- Cold-start handling for new products
- Hybrid collaborative + content-based filtering

## Troubleshooting

### Common Issues

1. **Prophet Installation Issues**
   ```bash
   # Try installing pystan first
   pip install pystan
   pip install prophet
   ```

2. **Memory Issues with Large Datasets**
   - Use `hierarchy_level='category'` instead of `'sku_store'` for demand forecasting
   - Sample data for initial testing
   - Process data in chunks

3. **Missing Dependencies**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

4. **Date Format Issues**
   - Ensure dates are in pandas datetime format
   - Use `pd.to_datetime()` to convert if needed

## Next Steps

1. **Explore the Notebook**: Open `notebooks/exploratory_analysis.ipynb` for detailed analysis
2. **Customize Models**: Modify model parameters in individual model files
3. **Add Your Data**: Integrate your own retail datasets
4. **Production Deployment**: Adapt models for production use with proper validation and monitoring

## Key Interview Talking Points

When discussing these models in interviews, emphasize:

1. **Probabilistic Thinking**: "Point forecasts are insufficient; we need uncertainty quantification for risk-aware decisions."

2. **Causal ML**: "Markdown decisions are fundamentally causal problems—what happens if we change price today vs. next week?"

3. **Business Alignment**: "Rather than predicting churn in isolation, we frame around expected future value and how interventions change that trajectory."

4. **Systems Thinking**: "The challenge isn't just predicting demand, but making allocation decisions under uncertainty while balancing customer experience and cost."

5. **Rigor Matters**: "These are ML problems where rigor actually matters—uncertainty, decision costs, and customer impact are first-class concerns."

## Support

For questions or issues:
- Check the README.md for detailed model documentation
- Review model source code in the `models/` directory
- Explore example usage in `main.py`
