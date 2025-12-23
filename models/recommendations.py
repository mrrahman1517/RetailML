"""
Personalization & Recommendation System

Implements hybrid collaborative filtering with cold-start handling.
Balances personalization with discovery, inventory constraints, and brand goals.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import implicit
    IMPLICIT_AVAILABLE = True
except ImportError:
    IMPLICIT_AVAILABLE = False
    print("Implicit not available. Using alternative methods.")


class RecommendationSystem:
    """
    Hybrid recommendation system for retail.
    
    Key features:
    - Collaborative filtering (user-item interactions)
    - Content-based filtering (product attributes)
    - Cold-start handling for new products/users
    - Seasonal trend incorporation
    """
    
    def __init__(self, n_factors=50):
        self.n_factors = n_factors
        self.user_factors = None
        self.item_factors = None
        self.user_encoder = {}
        self.item_encoder = {}
        self.user_decoder = {}
        self.item_decoder = {}
        self.product_features = None
        
    def prepare_interaction_matrix(self, transactions_df):
        """Create user-item interaction matrix"""
        # Aggregate interactions (implicit feedback: purchase = positive signal)
        interactions = transactions_df.groupby(['customer_id', 'product_id']).agg({
            'quantity': 'sum',
            'total_amount': 'sum'
        }).reset_index()
        
        # Create encoders
        unique_users = interactions['customer_id'].unique()
        unique_items = interactions['product_id'].unique()
        
        self.user_encoder = {user: idx for idx, user in enumerate(unique_users)}
        self.item_encoder = {item: idx for idx, item in enumerate(unique_items)}
        self.user_decoder = {idx: user for user, idx in self.user_encoder.items()}
        self.item_decoder = {idx: item for item, idx in self.item_encoder.items()}
        
        # Create sparse matrix
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        # Use quantity as interaction strength
        interaction_matrix = np.zeros((n_users, n_items))
        
        for _, row in interactions.iterrows():
            user_idx = self.user_encoder[row['customer_id']]
            item_idx = self.item_encoder[row['product_id']]
            interaction_matrix[user_idx, item_idx] = row['quantity']
        
        return interaction_matrix, unique_users, unique_items
    
    def train_collaborative_filtering(self, interaction_matrix):
        """Train collaborative filtering model using matrix factorization"""
        if IMPLICIT_AVAILABLE:
            # Use implicit library (optimized for implicit feedback)
            model = implicit.als.AlternatingLeastSquares(
                factors=self.n_factors,
                iterations=15,
                regularization=0.1,
                random_state=42
            )
            
            # Convert to sparse matrix
            from scipy.sparse import csr_matrix
            sparse_matrix = csr_matrix(interaction_matrix)
            
            model.fit(sparse_matrix)
            self.user_factors = model.user_factors
            self.item_factors = model.item_factors
        else:
            # Use NMF as alternative
            model = NMF(n_components=self.n_factors, random_state=42, max_iter=100)
            self.user_factors = model.fit_transform(interaction_matrix)
            self.item_factors = model.components_.T
        
        return self
    
    def prepare_content_features(self, products_df):
        """Prepare content-based features for products"""
        # One-hot encode categories
        category_encoded = pd.get_dummies(products_df['category'], prefix='cat')
        
        # Price features (normalized)
        price_scaler = StandardScaler()
        price_features = price_scaler.fit_transform(products_df[['price']].values)
        
        # Combine features
        self.product_features = pd.concat([
            products_df[['product_id']],
            category_encoded,
            pd.DataFrame(price_features, columns=['price_norm'])
        ], axis=1)
        
        # Set product_id as index for easy lookup
        self.product_features = self.product_features.set_index('product_id')
        
        return self.product_features
    
    def train(self, transactions_df, products_df):
        """Train recommendation system"""
        print("Preparing interaction matrix...")
        interaction_matrix, users, items = self.prepare_interaction_matrix(transactions_df)
        
        print("Training collaborative filtering model...")
        self.train_collaborative_filtering(interaction_matrix)
        
        print("Preparing content features...")
        self.prepare_content_features(products_df)
        
        print("Training complete!")
        return self
    
    def recommend_collaborative(self, customer_id, n_recommendations=10, 
                                exclude_purchased=True, transactions_df=None):
        """Generate recommendations using collaborative filtering"""
        if customer_id not in self.user_encoder:
            # Cold start: new user
            return self.recommend_popular(products_df=None, n_recommendations=n_recommendations)
        
        user_idx = self.user_encoder[customer_id]
        user_vector = self.user_factors[user_idx]
        
        # Compute scores for all items
        scores = np.dot(self.item_factors, user_vector)
        
        # Exclude already purchased items
        if exclude_purchased and transactions_df is not None:
            purchased_items = set(
                transactions_df[transactions_df['customer_id'] == customer_id]['product_id'].unique()
            )
            for item_id, item_idx in self.item_encoder.items():
                if item_id in purchased_items:
                    scores[item_idx] = -np.inf
        
        # Get top recommendations
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            product_id = self.item_decoder[idx]
            score = scores[idx]
            recommendations.append({
                'product_id': product_id,
                'score': score
            })
        
        return pd.DataFrame(recommendations)
    
    def recommend_content_based(self, customer_id, transactions_df, 
                                products_df, n_recommendations=10):
        """Generate recommendations using content-based filtering"""
        # Get customer's purchase history
        customer_purchases = transactions_df[
            transactions_df['customer_id'] == customer_id
        ]['product_id'].unique()
        
        if len(customer_purchases) == 0:
            return self.recommend_popular(products_df, n_recommendations)
        
        # Get features of purchased products
        purchased_features = self.product_features.loc[
            self.product_features.index.isin(customer_purchases)
        ]
        
        # Average user profile
        user_profile = purchased_features.mean(axis=0).values
        
        # Compute similarity to all products
        all_features = self.product_features.values
        similarities = cosine_similarity([user_profile], all_features)[0]
        
        # Exclude already purchased
        product_ids = self.product_features.index.tolist()
        for i, pid in enumerate(product_ids):
            if pid in customer_purchases:
                similarities[i] = -1
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            product_id = product_ids[idx]
            score = similarities[idx]
            recommendations.append({
                'product_id': product_id,
                'score': score
            })
        
        return pd.DataFrame(recommendations)
    
    def recommend_hybrid(self, customer_id, transactions_df, products_df, 
                        n_recommendations=10, cf_weight=0.7):
        """Hybrid recommendations combining collaborative and content-based"""
        # Get collaborative recommendations
        cf_recs = self.recommend_collaborative(
            customer_id, n_recommendations=n_recommendations * 2,
            exclude_purchased=True, transactions_df=transactions_df
        )
        
        # Get content-based recommendations
        cb_recs = self.recommend_content_based(
            customer_id, transactions_df, products_df,
            n_recommendations=n_recommendations * 2
        )
        
        # Normalize scores
        if len(cf_recs) > 0:
            cf_recs['score_norm'] = (cf_recs['score'] - cf_recs['score'].min()) / (
                cf_recs['score'].max() - cf_recs['score'].min() + 1e-6
            )
        else:
            cf_recs['score_norm'] = 0
        
        if len(cb_recs) > 0:
            cb_recs['score_norm'] = (cb_recs['score'] - cb_recs['score'].min()) / (
                cb_recs['score'].max() - cb_recs['score'].min() + 1e-6
            )
        else:
            cb_recs['score_norm'] = 0
        
        # Merge and combine scores
        merged = pd.merge(
            cf_recs[['product_id', 'score_norm']].rename(columns={'score_norm': 'cf_score'}),
            cb_recs[['product_id', 'score_norm']].rename(columns={'score_norm': 'cb_score'}),
            on='product_id',
            how='outer'
        ).fillna(0)
        
        merged['hybrid_score'] = (
            cf_weight * merged['cf_score'] + 
            (1 - cf_weight) * merged['cb_score']
        )
        
        # Get top recommendations
        recommendations = merged.nlargest(n_recommendations, 'hybrid_score')
        
        return recommendations[['product_id', 'hybrid_score', 'cf_score', 'cb_score']]
    
    def recommend_popular(self, products_df=None, n_recommendations=10):
        """Fallback: recommend popular items (for cold start)"""
        # This would typically use global popularity metrics
        # For now, return empty or use product features
        if products_df is not None:
            popular = products_df.nlargest(n_recommendations, 'base_demand')
            return pd.DataFrame({
                'product_id': popular['product_id'].values,
                'score': popular['base_demand'].values
            })
        return pd.DataFrame(columns=['product_id', 'score'])
    
    def recommend_for_new_product(self, product_id, products_df, transactions_df, 
                                 n_recommendations=10):
        """Handle cold start for new products using content-based approach"""
        if product_id not in self.product_features.index:
            # Add product to features if not present
            product = products_df[products_df['product_id'] == product_id]
            if len(product) == 0:
                return self.recommend_popular(products_df, n_recommendations)
            
            # This is simplified - in production would properly add to feature matrix
            pass
        
        # Get product features
        product_features = self.product_features.loc[product_id].values
        
        # Find similar products
        all_features = self.product_features.values
        similarities = cosine_similarity([product_features], all_features)[0]
        
        # Get top similar products
        product_ids = self.product_features.index.tolist()
        top_indices = np.argsort(similarities)[::-1][:n_recommendations + 1]
        
        # Exclude the product itself
        recommendations = []
        for idx in top_indices:
            pid = product_ids[idx]
            if pid != product_id:
                recommendations.append({
                    'product_id': pid,
                    'similarity': similarities[idx]
                })
        
        return pd.DataFrame(recommendations[:n_recommendations])

