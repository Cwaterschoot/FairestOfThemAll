import pandas as pd
import numpy as np
import random
from typing import Dict, List, Any


def select_top_n(df, value_col, item_col='item', n=1, random_state=None):
    rng = np.random.default_rng(random_state)
    sorted_df = df.sort_values(value_col, ascending=False).reset_index(drop=True)
    
    results = []
    i = 0
    while len(results) < n and i < len(sorted_df):
        tied_items = sorted_df.loc[sorted_df[value_col] == sorted_df.loc[i, value_col], item_col].tolist()
        
        remaining_slots = n - len(results)
        if len(tied_items) <= remaining_slots:
            results.extend(tied_items)
        else:
            chosen = rng.choice(tied_items, size=remaining_slots, replace=False).tolist()
            results.extend(chosen)
        i += len(tied_items)
    
    return results

def ADD(df, n=10): 
                    counts = df.groupby('item')['rating'].sum().reset_index(name='sum_rating')
                    return select_top_n(counts, 'sum_rating', n=n, random_state=123)

def APP(df, threshold=6, n=10, random_state=None):
    above_threshold = df[df['rating'] > threshold]
    counts = above_threshold.groupby('item').size().reset_index(name='count_above_threshold')
    if counts.empty:
        return []
    return select_top_n(counts, 'count_above_threshold', n=n, random_state=random_state)


def LMS(df, n=10, random_state=None):
    counts = df.groupby('item')['rating'].min().reset_index(name='min_rating')
    return select_top_n(counts, 'min_rating', n=n, random_state=random_state)


def MPL(df, n=10, random_state=None):
    counts = df.groupby('item')['rating'].max().reset_index(name='max_rating')
    return select_top_n(counts, 'max_rating', n=n, random_state=random_state)


def MAJ(ratings_dict, n=10, random_state=None):
    df = pd.DataFrame(ratings_dict)
    max_ratings = df.max(axis=1)
    votes = df.eq(max_ratings, axis=0).astype(int)
    item_votes = votes.sum(axis=0).reset_index()
    item_votes.columns = ['item', 'votes']
    return select_top_n(item_votes, 'votes', n=n, random_state=random_state)




### DOES NTO WORK: FIX TOMORROW



def FAI(ratings_data: Dict[str, List[int]]) -> List[str]:

    if not ratings_data or 'names' not in ratings_data:
        return []
    
    names = ratings_data['names']
    restaurants = {k: v for k, v in ratings_data.items() if k != 'names'}
    restaurant_names = list(restaurants.keys())

    transformed = {
        name: [ratings[i] for ratings in restaurants.values()]
        for i, name in enumerate(names)
    }
    
    users = list(transformed.keys())
    ratings_matrix: List[List[int]] = [transformed[user].copy() for user in users]
    
    num_users = len(users)

    selected_restaurants: List[str] = []


    for user_index in range(num_users):
        user_name = users[user_index]
        user_ratings = ratings_matrix[user_index]
        
        # Find the maximum available rating
        max_rating = max(user_ratings)

        if max_rating == 0:
            print(f"Warning: {user_name} has no remaining unrated items. Skipping selection.")
            continue
            
        # Find all item indices that share the maximum rating
        max_indices = [
            item_idx for item_idx, rating in enumerate(user_ratings) 
            if rating == max_rating
        ]
        
        # Select one item index randomly
        chosen_item_index = random.choice(max_indices)
        
        # FIX 1: Retrieve the actual item name using the restaurant_names list
        chosen_item_name = restaurant_names[chosen_item_index] 
        
        # FIX 2: Append the chosen item name to the result list (This was missing!)
        selected_restaurants.append(chosen_item_name)
        
        # Optional: Print selection
        max_options = [restaurant_names[i] for i in max_indices]
        
        # Exclusion Step: Zero out the chosen item's rating for ALL users
        for i in range(num_users):
            ratings_matrix[i][chosen_item_index] = 0

    return selected_restaurants
