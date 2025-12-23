"""
Omnichannel Fulfillment & Allocation Model

Optimizes fulfillment decisions under uncertainty.
Balances shipping cost, delivery time, and store inventory health.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import linprog
import warnings
warnings.filterwarnings('ignore')


class OmnichannelFulfillmentModel:
    """
    Omnichannel fulfillment optimization model.
    
    Key features:
    - Demand prediction integration
    - Cost optimization
    - Constraint satisfaction
    - Real-time decision making
    """
    
    def __init__(self):
        self.demand_model = None
        self.store_costs = {}
        self.shipping_costs = {}
        
    def estimate_fulfillment_costs(self, stores, shipping_zones):
        """Estimate fulfillment costs for different channels"""
        # Store fulfillment costs (per unit)
        for store in stores:
            self.store_costs[store] = {
                'pick_cost': 2.0,  # Cost to pick item in store
                'pack_cost': 1.5,  # Cost to pack
                'ship_cost_local': 5.0,  # Local shipping
                'ship_cost_regional': 8.0,  # Regional shipping
                'ship_cost_national': 12.0  # National shipping
            }
        
        # Warehouse costs
        self.warehouse_costs = {
            'pick_cost': 1.0,
            'pack_cost': 1.0,
            'ship_cost_local': 4.0,
            'ship_cost_regional': 7.0,
            'ship_cost_national': 10.0
        }
        
        # Shipping zones (simplified)
        self.shipping_zones = shipping_zones
        
        return self
    
    def predict_demand(self, product_id, store_id, date, transactions_df):
        """Predict demand for a product at a store (simplified)"""
        # In production, would use actual demand forecasting model
        # For now, use historical average
        
        historical = transactions_df[
            (transactions_df['product_id'] == product_id) &
            (transactions_df['store_id'] == store_id)
        ]
        
        if len(historical) > 0:
            avg_demand = historical['quantity'].mean()
            # Add some uncertainty
            demand_std = historical['quantity'].std()
            predicted_demand = max(0, np.random.normal(avg_demand, demand_std * 0.5))
        else:
            # Cold start: use category average
            category = transactions_df[
                transactions_df['product_id'] == product_id
            ]['category'].iloc[0] if len(transactions_df[
                transactions_df['product_id'] == product_id
            ]) > 0 else 'Unknown'
            
            category_demand = transactions_df[
                transactions_df['category'] == category
            ]['quantity'].mean() if category != 'Unknown' else 1
            
            predicted_demand = category_demand
        
        return predicted_demand
    
    def optimize_fulfillment(self, order, inventory_df, stores, 
                            customer_location='Regional'):
        """
        Optimize fulfillment for a single order.
        
        Returns optimal fulfillment source and route.
        """
        product_id = order['product_id']
        quantity = order['quantity']
        customer_location = order.get('customer_location', 'Regional')
        
        # Get available inventory
        available_inventory = inventory_df[
            inventory_df['product_id'] == product_id
        ].copy()
        
        if len(available_inventory) == 0:
            return {
                'source': 'Warehouse',
                'cost': self.warehouse_costs['pick_cost'] + 
                       self.warehouse_costs['pack_cost'] +
                       self.warehouse_costs[f'ship_cost_{customer_location.lower()}'],
                'delivery_time_days': 3 if customer_location == 'Local' else 
                                     (5 if customer_location == 'Regional' else 7),
                'feasible': True
            }
        
        # Calculate costs for each fulfillment option
        options = []
        
        # Option 1: Warehouse
        warehouse_cost = (
            self.warehouse_costs['pick_cost'] +
            self.warehouse_costs['pack_cost'] +
            self.warehouse_costs[f'ship_cost_{customer_location.lower()}']
        )
        warehouse_time = 3 if customer_location == 'Local' else (
            5 if customer_location == 'Regional' else 7
        )
        
        options.append({
            'source': 'Warehouse',
            'type': 'warehouse',
            'cost': warehouse_cost,
            'delivery_time_days': warehouse_time,
            'inventory_available': True,  # Assume warehouse always has stock
            'feasible': True
        })
        
        # Option 2: Store fulfillment
        for store_id in stores:
            store_inventory = available_inventory[
                available_inventory['store_id'] == store_id
            ]
            
            if len(store_inventory) > 0 and store_inventory['stock_level'].iloc[0] >= quantity:
                store_cost = (
                    self.store_costs[store_id]['pick_cost'] +
                    self.store_costs[store_id]['pack_cost'] +
                    self.store_costs[store_id][f'ship_cost_{customer_location.lower()}']
                )
                
                # Local store = faster delivery
                if store_id == order.get('preferred_store'):
                    delivery_time = 1
                else:
                    delivery_time = 2 if customer_location == 'Local' else (
                        4 if customer_location == 'Regional' else 6
                    )
                
                options.append({
                    'source': store_id,
                    'type': 'store',
                    'cost': store_cost,
                    'delivery_time_days': delivery_time,
                    'inventory_available': True,
                    'feasible': True,
                    'store_id': store_id
                })
        
        # Select optimal option
        # Objective: minimize cost while meeting delivery time constraints
        # In production, would use more sophisticated optimization
        
        if len(options) == 0:
            # Fallback to warehouse
            return options[0] if len(options) > 0 else None
        
        # Score options (lower is better)
        # Weight: cost (70%), delivery time (30%)
        for option in options:
            cost_score = option['cost'] / max([o['cost'] for o in options])
            time_score = option['delivery_time_days'] / max([o['delivery_time_days'] for o in options])
            option['score'] = 0.7 * cost_score + 0.3 * time_score
        
        # Select best option
        best_option = min(options, key=lambda x: x['score'])
        
        return best_option
    
    def optimize_allocation(self, orders, inventory_df, stores, 
                           max_capacity_per_store=1000):
        """
        Optimize allocation of multiple orders across fulfillment sources.
        Uses linear programming for optimization.
        """
        # Prepare data
        order_list = []
        for idx, order in orders.iterrows():
            order_list.append({
                'order_id': order.get('order_id', idx),
                'product_id': order['product_id'],
                'quantity': order['quantity'],
                'customer_location': order.get('customer_location', 'Regional')
            })
        
        # Simple greedy allocation (in production, would use LP/MIP)
        allocations = []
        
        for order in order_list:
            allocation = self.optimize_fulfillment(
                order, inventory_df, stores, order['customer_location']
            )
            
            if allocation:
                allocation['order_id'] = order['order_id']
                allocation['product_id'] = order['product_id']
                allocation['quantity'] = order['quantity']
                allocations.append(allocation)
        
        return pd.DataFrame(allocations)
    
    def evaluate_fulfillment_strategy(self, allocations, orders):
        """Evaluate fulfillment strategy performance"""
        results = allocations.merge(orders, on='order_id', how='left')
        
        metrics = {
            'total_orders': len(allocations),
            'total_cost': results['cost'].sum(),
            'avg_cost_per_order': results['cost'].mean(),
            'avg_delivery_time': results['delivery_time_days'].mean(),
            'warehouse_fulfillments': (results['type'] == 'warehouse').sum(),
            'store_fulfillments': (results['type'] == 'store').sum(),
            'on_time_rate': (results['delivery_time_days'] <= 5).sum() / len(results)
        }
        
        return metrics
    
    def train(self, transactions_df, inventory_df, stores):
        """Train/initialize omnichannel model"""
        # Estimate costs
        shipping_zones = ['Local', 'Regional', 'National']
        self.estimate_fulfillment_costs(stores, shipping_zones)
        
        # Store transaction data for demand prediction
        self.transactions_df = transactions_df
        
        print("Omnichannel model initialized!")
        return self

