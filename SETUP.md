# Setup Instructions

Complete setup guide for the Retail Data Science & ML Models project.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- 2GB+ disk space for datasets

## Step-by-Step Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Note on Optional Dependencies:**

Some packages may require additional setup:

- **Prophet**: May require `pystan` and C++ compiler
  ```bash
  pip install pystan
  pip install prophet
  ```

- **TensorFlow/PyTorch**: Install based on your system
  - CPU only: Already in requirements.txt
  - GPU support: See [TensorFlow](https://www.tensorflow.org/install) or [PyTorch](https://pytorch.org/get-started/locally/) documentation

- **Kaggle API** (optional, for downloading Kaggle datasets):
  ```bash
  pip install kaggle
  # Place kaggle.json in ~/.kaggle/ directory
  ```

### 2. Verify Installation

```bash
python -c "import pandas, numpy, sklearn; print('Core packages installed successfully')"
```

### 3. Generate or Download Data

**Option A: Use Synthetic Data (Quick Start)**
```bash
python main.py
```
This will automatically generate synthetic data and run all models.

**Option B: Download Real Dataset**
```python
from utils.dataset_acquisition import RetailDatasetAcquisition

acquirer = RetailDatasetAcquisition()
online_retail_df = acquirer.prepare_online_retail_dataset()
```

**Option C: Use Your Own Data**
1. Place your data files in `data/raw/`
2. Ensure column names match the schema in `DATASETS.md`
3. Update `main.py` to load your data

### 4. Run the Complete Pipeline

```bash
python main.py
```

Expected output:
- Data generation/loading
- Model training for all 8 problems
- Predictions and insights
- Summary statistics

### 5. Explore in Jupyter Notebook

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

## Project Structure

```
RetailDS/
├── data/
│   ├── raw/              # Raw datasets (CSV files)
│   └── processed/        # Processed features (if needed)
├── models/               # ML model implementations
│   ├── demand_forecasting.py
│   ├── markdown_optimization.py
│   ├── clv_retention.py
│   ├── recommendations.py
│   ├── returns_prediction.py
│   ├── fraud_detection.py
│   ├── omnichannel.py
│   └── merchandising.py
├── notebooks/            # Jupyter notebooks
│   └── exploratory_analysis.ipynb
├── utils/                # Utility functions
│   ├── data_generator.py
│   ├── dataset_acquisition.py
│   └── evaluation.py
├── main.py              # Main execution script
├── requirements.txt     # Python dependencies
├── README.md           # Project overview
├── QUICKSTART.md       # Quick start guide
├── DATASETS.md         # Dataset information
└── SETUP.md            # This file
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Prophet Installation Fails**
   ```bash
   # Try installing dependencies separately
   pip install pystan
   pip install prophet
   # Or skip Prophet - models have fallback methods
   ```

3. **Memory Errors**
   - Reduce dataset size: `df.sample(n=100000)`
   - Use category-level forecasting instead of SKU-level
   - Process data in chunks

4. **Date Parsing Errors**
   ```python
   df['date'] = pd.to_datetime(df['date'], errors='coerce')
   ```

5. **Import Errors**
   - Ensure you're in the project root directory
   - Check Python path: `import sys; print(sys.path)`
   - Use: `python -m utils.dataset_acquisition`

### Platform-Specific Notes

**macOS:**
- May need Xcode Command Line Tools: `xcode-select --install`
- Prophet may require: `brew install gcc`

**Linux:**
- May need: `sudo apt-get install build-essential python3-dev`
- Prophet may require: `sudo apt-get install gcc g++`

**Windows:**
- Use Anaconda/Miniconda recommended
- Prophet may require Visual C++ Build Tools

## Verification

After setup, verify everything works:

```python
# Test imports
from models.demand_forecasting import DemandForecastingModel
from models.clv_retention import CLVModel
from utils.data_generator import RetailDataGenerator
from utils.dataset_acquisition import RetailDatasetAcquisition

print("✓ All imports successful")

# Test data generation
generator = RetailDataGenerator(seed=42)
datasets = generator.generate_all_datasets()
print(f"✓ Generated {len(datasets)} datasets")

# Test model initialization
model = DemandForecastingModel()
print("✓ Models initialize successfully")
```

## Next Steps

1. **Read QUICKSTART.md** for usage examples
2. **Explore DATASETS.md** for dataset information
3. **Run main.py** to see all models in action
4. **Open the Jupyter notebook** for detailed analysis
5. **Customize models** for your specific use case

## Getting Help

- Check the README.md for detailed model documentation
- Review model source code in `models/` directory
- Explore example usage in `main.py`
- See QUICKSTART.md for common use cases

## Production Considerations

For production deployment:

1. **Model Versioning**: Use MLflow or similar
2. **Data Validation**: Add schema validation
3. **Monitoring**: Implement model performance monitoring
4. **Retraining**: Set up automated retraining pipelines
5. **API**: Wrap models in REST API (Flask/FastAPI)
6. **Testing**: Add unit and integration tests
7. **Documentation**: Document model assumptions and limitations

## License & Usage

This project is for educational and demonstration purposes. When using real retail data:
- Ensure compliance with data privacy regulations (GDPR, CCPA, etc.)
- Anonymize customer data
- Respect data usage agreements
- Follow ethical ML practices

