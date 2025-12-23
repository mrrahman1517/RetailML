"""
Synthetic Retail Data Generator

Generates realistic retail datasets for training ML models.
Simulates fashion retail scenarios with seasonality, long-tail SKUs, and customer behavior.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random


class RetailDataGenerator:
    """Generate synthetic retail datasets"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Product categories (fashion-focused)
        self.categories = ['Dresses', 'Tops', 'Bottoms', 'Outerwear', 'Shoes', 
                          'Accessories', 'Handbags', 'Jewelry']
        
        # Store locations
        self.stores = [f'Store_{i:03d}' for i in range(1, 51)]  # 50 stores
        
        # Size options
        self.sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
        
    def generate_products(self, n_products=5000):
        """Generate product catalog with long-tail distribution"""
        products = []
        
        # Long-tail: 20% of products get 80% of sales
        popular_products = int(n_products * 0.2)
        
        for i in range(n_products):
            is_popular = i < popular_products
            category = np.random.choice(self.categories)
            
            # Base price varies by category
            base_prices = {'Dresses': 150, 'Tops': 60, 'Bottoms': 80, 
                          'Outerwear': 200, 'Shoes': 120, 'Accessories': 40,
                          'Handbags': 300, 'Jewelry': 250}
            
            base_price = base_prices[category]
            price = base_price * np.random.uniform(0.7, 1.5)
            
            # Popular products have higher base demand
            base_demand = np.random.gamma(2, 10) if is_popular else np.random.gamma(1, 3)
            
            products.append({
                'product_id': f'PROD_{i:05d}',
                'category': category,
                'price': round(price, 2),
                'base_demand': base_demand,
                'size_available': np.random.choice(self.sizes, size=np.random.randint(3, 6), replace=False).tolist(),
                'is_popular': is_popular
            })
        
        return pd.DataFrame(products)
    
    def generate_customers(self, n_customers=10000):
        """Generate customer base with segments"""
        customers = []
        
        # Customer segments: High-value (10%), Medium (30%), Low (60%)
        segments = ['High-Value'] * int(n_customers * 0.1) + \
                  ['Medium'] * int(n_customers * 0.3) + \
                  ['Low'] * int(n_customers * 0.6)
        random.shuffle(segments)
        
        for i, segment in enumerate(segments):
            # Segment determines purchase frequency and average order value
            if segment == 'High-Value':
                purchase_freq = np.random.gamma(3, 2)  # More frequent
                avg_order_value = np.random.gamma(5, 50)  # Higher AOV
            elif segment == 'Medium':
                purchase_freq = np.random.gamma(2, 3)
                avg_order_value = np.random.gamma(3, 30)
            else:
                purchase_freq = np.random.gamma(1, 5)
                avg_order_value = np.random.gamma(2, 20)
            
            customers.append({
                'customer_id': f'CUST_{i:06d}',
                'segment': segment,
                'purchase_frequency': purchase_freq,
                'avg_order_value': avg_order_value,
                'first_purchase_date': None,  # Will be set when generating transactions
                'last_purchase_date': None
            })
        
        return pd.DataFrame(customers)
    
    def generate_sales_transactions(self, products_df, customers_df, 
                                   start_date='2022-01-01', end_date='2024-12-31',
                                   n_transactions=500000):
        """Generate sales transactions with seasonality"""
        transactions = []
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        # Seasonal multipliers (fashion retail has strong seasonality)
        seasonal_pattern = {
            1: 0.8,   # January (post-holiday)
            2: 0.7,   # February
            3: 1.0,   # March (spring)
            4: 1.2,   # April
            5: 1.1,   # May
            6: 1.0,   # June
            7: 0.9,   # July
            8: 1.0,   # August (back-to-school)
            9: 1.1,   # September
            10: 1.3,  # October (fall)
            11: 1.5,  # November (holiday prep)
            12: 2.0   # December (holiday peak)
        }
        
        # Generate transactions
        for _ in range(n_transactions):
            # Select date with seasonality
            date = pd.to_datetime(np.random.choice(date_range))
            seasonal_mult = seasonal_pattern[date.month]
            
            # Select customer (weighted by segment)
            customer = customers_df.sample(1).iloc[0]
            
            # Select product (long-tail distribution)
            if np.random.random() < 0.8:  # 80% from popular products
                product = products_df[products_df['is_popular']].sample(1).iloc[0]
            else:
                product = products_df.sample(1).iloc[0]
            
            # Quantity (usually 1-3 items)
            quantity = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
            
            # Apply seasonal multiplier to demand
            demand_mult = seasonal_mult * np.random.uniform(0.8, 1.2)
            base_demand = product['base_demand'] * demand_mult
            
            # Price (may have markdowns)
            price = product['price']
            if np.random.random() < 0.15:  # 15% chance of markdown
                discount = np.random.choice([0.1, 0.2, 0.3, 0.5], p=[0.4, 0.3, 0.2, 0.1])
                price = price * (1 - discount)
            
            # Store selection
            store = np.random.choice(self.stores)
            
            # Channel (online vs in-store)
            channel = np.random.choice(['Online', 'In-Store'], p=[0.4, 0.6])
            
            transactions.append({
                'transaction_id': f'TXN_{len(transactions):08d}',
                'date': date,
                'customer_id': customer['customer_id'],
                'product_id': product['product_id'],
                'store_id': store,
                'channel': channel,
                'quantity': quantity,
                'unit_price': round(price, 2),
                'total_amount': round(price * quantity, 2),
                'category': product['category'],
                'discount_applied': product['price'] - price if price < product['price'] else 0
            })
        
        transactions_df = pd.DataFrame(transactions)
        transactions_df = transactions_df.sort_values('date')
        
        return transactions_df
    
    def generate_returns(self, transactions_df, return_rate=0.15):
        """Generate return transactions (apparel has high return rates)"""
        returns = []
        
        # Sample transactions for returns (not all are returned)
        return_candidates = transactions_df.sample(frac=return_rate)
        
        for _, txn in return_candidates.iterrows():
            # Returns typically happen 3-30 days after purchase
            return_delay = np.random.randint(3, 31)
            return_date = txn['date'] + timedelta(days=return_delay)
            
            # Return reasons
            reasons = ['Size/Fit', 'Changed Mind', 'Defective', 'Not as Described', 'Other']
            reason = np.random.choice(reasons, p=[0.4, 0.3, 0.1, 0.1, 0.1])
            
            # Fraud indicators (small percentage)
            is_fraud = np.random.random() < 0.02  # 2% fraud rate
            
            returns.append({
                'return_id': f'RET_{len(returns):08d}',
                'transaction_id': txn['transaction_id'],
                'return_date': return_date,
                'return_reason': reason,
                'is_fraud': is_fraud,
                'customer_id': txn['customer_id'],
                'product_id': txn['product_id'],
                'amount_returned': txn['total_amount']
            })
        
        return pd.DataFrame(returns)
    
    def generate_inventory(self, products_df, stores, start_date='2022-01-01'):
        """Generate inventory levels over time"""
        inventory_records = []
        date_range = pd.date_range(start_date, '2024-12-31', freq='W')  # Weekly
        
        for date in date_range:
            for store in stores:
                for _, product in products_df.iterrows():
                    # Base stock level
                    base_stock = int(product['base_demand'] * np.random.uniform(0.5, 2.0))
                    
                    # Stockout events (low probability)
                    is_stockout = np.random.random() < 0.05
                    stock_level = 0 if is_stockout else max(0, base_stock + np.random.randint(-10, 10))
                    
                    inventory_records.append({
                        'date': date,
                        'store_id': store,
                        'product_id': product['product_id'],
                        'stock_level': stock_level,
                        'is_stockout': is_stockout
                    })
        
        return pd.DataFrame(inventory_records)
    
    def generate_all_datasets(self, output_dir='data/raw'):
        """Generate all datasets and save to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating products...")
        products_df = self.generate_products()
        products_df.to_csv(f'{output_dir}/products.csv', index=False)
        
        print("Generating customers...")
        customers_df = self.generate_customers()
        customers_df.to_csv(f'{output_dir}/customers.csv', index=False)
        
        print("Generating sales transactions...")
        transactions_df = self.generate_sales_transactions(products_df, customers_df)
        transactions_df.to_csv(f'{output_dir}/transactions.csv', index=False)
        
        print("Generating returns...")
        returns_df = self.generate_returns(transactions_df)
        returns_df.to_csv(f'{output_dir}/returns.csv', index=False)
        
        print("Generating inventory...")
        inventory_df = self.generate_inventory(products_df, self.stores)
        inventory_df.to_csv(f'{output_dir}/inventory.csv', index=False)
        
        print(f"All datasets generated and saved to {output_dir}/")
        
        return {
            'products': products_df,
            'customers': customers_df,
            'transactions': transactions_df,
            'returns': returns_df,
            'inventory': inventory_df
        }

