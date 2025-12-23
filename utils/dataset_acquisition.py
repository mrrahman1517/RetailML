"""
Dataset Acquisition Utilities

Provides functions to download and prepare real retail datasets for ML modeling.
Includes support for:
- Kaggle datasets
- UCI Machine Learning Repository
- Public retail datasets
- Synthetic data generation (fallback)
"""

import os
import pandas as pd
import numpy as np
import requests
import zipfile
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class RetailDatasetAcquisition:
    """Acquire and prepare retail datasets for ML modeling"""
    
    def __init__(self, data_dir='data/raw'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_kaggle_dataset(self, dataset_name: str, kaggle_username: str = None, 
                                kaggle_key: str = None) -> bool:
        """
        Download dataset from Kaggle
        
        Requires kaggle package and API credentials.
        Install: pip install kaggle
        Setup: Place kaggle.json in ~/.kaggle/
        """
        try:
            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            api = KaggleApi()
            api.authenticate()
            
            print(f"Downloading {dataset_name} from Kaggle...")
            api.dataset_download_files(dataset_name, path=str(self.data_dir), unzip=True)
            print(f"Dataset downloaded successfully to {self.data_dir}")
            return True
        except ImportError:
            print("Kaggle package not installed. Install with: pip install kaggle")
            return False
        except Exception as e:
            print(f"Error downloading Kaggle dataset: {e}")
            return False
    
    def download_uci_dataset(self, dataset_url: str, filename: str) -> Optional[pd.DataFrame]:
        """Download dataset from UCI Machine Learning Repository"""
        try:
            filepath = self.data_dir / filename
            if filepath.exists():
                print(f"File already exists: {filepath}")
                return pd.read_csv(filepath)
            
            print(f"Downloading from {dataset_url}...")
            response = requests.get(dataset_url, timeout=30)
            response.raise_for_status()
            
            filepath.write_bytes(response.content)
            print(f"Downloaded to {filepath}")
            
            # Try to read as CSV
            try:
                return pd.read_csv(filepath)
            except:
                # Try other formats
                if filename.endswith('.data'):
                    return pd.read_csv(filepath, sep='\s+', header=None)
                return None
        except Exception as e:
            print(f"Error downloading UCI dataset: {e}")
            return None
    
    def get_retail_datasets_info(self) -> Dict[str, Dict]:
        """
        Returns information about available retail datasets
        
        Returns dictionary with dataset names and metadata
        """
        datasets_info = {
            'online_retail': {
                'name': 'Online Retail Dataset',
                'source': 'UCI',
                'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx',
                'description': 'Online retail transactions from UK-based retailer',
                'type': 'transactions',
                'format': 'xlsx'
            },
            'superstore_sales': {
                'name': 'Superstore Sales',
                'source': 'Kaggle',
                'dataset': 'rohanrao/air-quality-data-in-india',
                'description': 'Retail sales data with product categories',
                'type': 'sales',
                'format': 'csv'
            },
            'walmart_sales': {
                'name': 'Walmart Sales Forecasting',
                'source': 'Kaggle',
                'dataset': 'c/walmart-recruiting-store-sales-forecasting',
                'description': 'Historical sales data for Walmart stores',
                'type': 'forecasting',
                'format': 'csv'
            }
        }
        return datasets_info
    
    def prepare_online_retail_dataset(self) -> Optional[pd.DataFrame]:
        """
        Download and prepare the UCI Online Retail dataset
        
        This is a real-world retail transaction dataset
        """
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx'
        filepath = self.data_dir / 'online_retail.xlsx'
        
        try:
            if not filepath.exists():
                print("Downloading Online Retail dataset from UCI...")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                filepath.write_bytes(response.content)
                print(f"Downloaded to {filepath}")
            
            # Read Excel file
            df = pd.read_excel(filepath)
            
            # Clean and standardize column names
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            
            # Convert InvoiceDate to datetime
            if 'invoicedate' in df.columns:
                df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')
                df = df.rename(columns={'invoicedate': 'date'})
            
            # Remove cancelled orders (typically start with 'C')
            if 'invoiceno' in df.columns:
                df = df[~df['invoiceno'].astype(str).str.startswith('C')]
            
            # Remove negative quantities
            if 'quantity' in df.columns:
                df = df[df['quantity'] > 0]
            
            # Standardize column names for our models
            column_mapping = {
                'stockcode': 'product_id',
                'description': 'product_name',
                'quantity': 'quantity',
                'unitprice': 'unit_price',
                'customerid': 'customer_id',
                'country': 'country'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            # Calculate total amount
            if 'quantity' in df.columns and 'unit_price' in df.columns:
                df['total_amount'] = df['quantity'] * df['unit_price']
            
            # Add category (simplified - would need product descriptions in real scenario)
            df['category'] = 'General'
            
            # Add store_id (online retail, so single store)
            df['store_id'] = 'Online'
            df['channel'] = 'Online'
            
            print(f"Prepared dataset: {len(df)} transactions")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"Unique products: {df['product_id'].nunique()}")
            print(f"Unique customers: {df['customer_id'].nunique()}")
            
            # Save processed version
            output_path = self.data_dir / 'online_retail_processed.csv'
            df.to_csv(output_path, index=False)
            print(f"Saved processed dataset to {output_path}")
            
            return df
            
        except Exception as e:
            print(f"Error preparing Online Retail dataset: {e}")
            return None
    
    def create_sample_retail_dataset(self, n_transactions=100000) -> Dict[str, pd.DataFrame]:
        """
        Create a comprehensive sample retail dataset if real data is unavailable
        
        This uses the existing data generator as a fallback
        """
        from utils.data_generator import RetailDataGenerator
        
        print("Generating synthetic retail dataset...")
        generator = RetailDataGenerator(seed=42)
        datasets = generator.generate_all_datasets(output_dir=str(self.data_dir))
        
        return datasets
    
    def get_available_datasets(self) -> Dict[str, bool]:
        """Check which datasets are available locally"""
        available = {}
        
        # Check for online retail dataset
        online_retail_path = self.data_dir / 'online_retail_processed.csv'
        available['online_retail'] = online_retail_path.exists()
        
        # Check for synthetic datasets
        synthetic_files = ['products.csv', 'customers.csv', 'transactions.csv', 
                          'returns.csv', 'inventory.csv']
        available['synthetic'] = all(
            (self.data_dir / f).exists() for f in synthetic_files
        )
        
        return available
    
    def load_dataset(self, dataset_name: str = 'synthetic') -> Dict[str, pd.DataFrame]:
        """
        Load available datasets
        
        Args:
            dataset_name: 'synthetic' or 'online_retail'
        """
        datasets = {}
        
        if dataset_name == 'online_retail':
            df = pd.read_csv(self.data_dir / 'online_retail_processed.csv')
            datasets['transactions'] = df
            
            # Create derived datasets
            datasets['products'] = df.groupby('product_id').agg({
                'product_name': 'first',
                'unit_price': 'mean',
                'category': 'first'
            }).reset_index()
            datasets['products']['product_id'] = datasets['products']['product_id'].astype(str)
            
            datasets['customers'] = df.groupby('customer_id').agg({
                'date': ['min', 'max'],
                'total_amount': 'sum'
            }).reset_index()
            datasets['customers'].columns = ['customer_id', 'first_purchase_date', 
                                           'last_purchase_date', 'total_spent']
            
        elif dataset_name == 'synthetic':
            datasets['products'] = pd.read_csv(self.data_dir / 'products.csv')
            datasets['customers'] = pd.read_csv(self.data_dir / 'customers.csv')
            datasets['transactions'] = pd.read_csv(self.data_dir / 'transactions.csv')
            datasets['returns'] = pd.read_csv(self.data_dir / 'returns.csv')
            # Inventory is optional - generate minimal if missing
            inventory_path = self.data_dir / 'inventory.csv'
            if inventory_path.exists():
                datasets['inventory'] = pd.read_csv(inventory_path)
            else:
                # Generate minimal inventory
                print("Generating minimal inventory dataset...")
                products = datasets['products']
                stores = datasets['transactions']['store_id'].unique()[:10]  # Sample stores
                inventory_records = []
                # Just one snapshot per product-store
                for store in stores:
                    for _, product in products.head(100).iterrows():  # Sample products
                        inventory_records.append({
                            'date': pd.Timestamp('2024-01-01'),
                            'store_id': store,
                            'product_id': product['product_id'],
                            'stock_level': np.random.randint(10, 100),
                            'is_stockout': 0
                        })
                datasets['inventory'] = pd.DataFrame(inventory_records)
                print(f"Generated minimal inventory: {len(datasets['inventory'])} records")
            
            # Convert date columns
            for df_name in ['transactions', 'returns', 'inventory']:
                if df_name in datasets and 'date' in datasets[df_name].columns:
                    datasets[df_name]['date'] = pd.to_datetime(datasets[df_name]['date'])
        
        return datasets


def main():
    """Main function to acquire and prepare datasets"""
    print("=" * 80)
    print("Retail Dataset Acquisition")
    print("=" * 80)
    
    acquirer = RetailDatasetAcquisition()
    
    # Show available datasets info
    print("\nAvailable Retail Datasets:")
    datasets_info = acquirer.get_retail_datasets_info()
    for name, info in datasets_info.items():
        print(f"\n{info['name']}:")
        print(f"  Source: {info['source']}")
        print(f"  Description: {info['description']}")
        print(f"  Type: {info['type']}")
    
    # Try to download Online Retail dataset
    print("\n" + "=" * 80)
    print("Attempting to download Online Retail dataset...")
    online_retail_df = acquirer.prepare_online_retail_dataset()
    
    if online_retail_df is not None:
        print("\n✓ Online Retail dataset ready!")
    else:
        print("\n✗ Could not download Online Retail dataset")
        print("Falling back to synthetic data generation...")
        datasets = acquirer.create_sample_retail_dataset()
        print("✓ Synthetic datasets generated!")
    
    # Check what's available
    print("\n" + "=" * 80)
    print("Available Datasets:")
    available = acquirer.get_available_datasets()
    for name, is_available in available.items():
        status = "✓" if is_available else "✗"
        print(f"  {status} {name}")
    
    print("\n" + "=" * 80)
    print("Dataset acquisition complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

