ratings = {
    'User1': {'Item1': 5, 'Item2': None, 'Item3': 3},
    'User2': {'Item1': 5, 'Item2': 2, 'Item3': 3},
    'User3': {'Item1': None, 'Item2': 5, 'Item3': 10},
    'User4': {'Item1':4, 'Item2':None, "Item3":3}
}


import math

# Cosine similarity
def cosine_similarity(vec1, vec2):
    #examle of vec1 and vec2
    vec1 = {'Item1': 5, 'Item2': None, 'Item3': 3}
    vec2 = {'Item1': 5, 'Item2': 2, 'Item3': 3}
    #common keys are Item1 and Item3
    #dot product is 5*5 + 3*3 = 25 + 9 = 34
    #norm1 is sqrt(5*5 + 0*0 + 3*3) = sqrt(25 + 9) = sqrt(34)
    #norm2 is sqrt(5*5 + 2*2 + 3*3) = sqrt(25 + 4 + 9) = sqrt(38)
    #cosine similarity is 34 / (sqrt(34) * sqrt(38)) = 0.9701425001442312
    #return 0.9701425001442312

    common_keys = set(vec1.keys()) & set(vec2.keys())
    if not common_keys:
        return 0
    dot = sum(vec1[k] * vec2[k] for k in common_keys)
    norm1 = math.sqrt(sum(vec1[k]**2 for k in common_keys))
    norm2 = math.sqrt(sum(vec2[k]**2 for k in common_keys))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0



# ============================================================================
# Content-Based Filtering
# ============================================================================

# Item features (content profile)
# Example: items could have features like genre, actors, keywords, etc.
item_features = {
    'Item1': {'action': 0.9, 'comedy': 0.1, 'drama': 0.3, 'sci-fi': 0.8},
    'Item2': {'action': 0.2, 'comedy': 0.9, 'drama': 0.7, 'sci-fi': 0.1},
    'Item3': {'action': 0.5, 'comedy': 0.3, 'drama': 0.9, 'sci-fi': 0.4}
}


def item_similarity(item1_features, item2_features):
    """
    Compute cosine similarity between two items based on their features
    """
    # Use existing cosine_similarity function
    return cosine_similarity(item1_features, item2_features)


def content_based_filtering(target_user, ratings, item_features):
    """
    Content-based filtering: recommend items similar to items user has rated
    Similar structure to user_based_cf but compares items instead of users
    
    For each unrated item:
    1. Find items user has rated
    2. Compute similarity between target item and rated items (using features)
    3. Weighted average prediction using similarities and ratings
    
    Parameters:
    -----------
    target_user : str
        Target user
    ratings : dict
        User ratings dictionary
    item_features : dict
        Item feature vectors (e.g., {'Item1': {'action': 0.9, 'comedy': 0.1, ...}})
    
    Returns:
    --------
    predictions : dict
        Predicted ratings for unrated items
    """
    predictions = {}
    user_ratings = ratings.get(target_user, {})
    
    # Find items user hasn't rated
    unrated_items = [item for item, rating in user_ratings.items() if rating is None]
    
    for target_item in unrated_items:
        if target_item not in item_features:
            continue
        
        numerator, denominator = 0, 0
        
        # For each item user HAS rated, compute similarity with target item
        for rated_item, rating in user_ratings.items():
            if rating is None:
                continue
            
            if rated_item not in item_features:
                continue
            
            # Compute similarity between target item and rated item (using features)
            similarity = item_similarity(item_features[target_item], item_features[rated_item])
            
            # Weighted average: similarity * rating
            numerator += similarity * rating
            denominator += similarity
        
        predictions[target_item] = numerator / denominator if denominator else None
    
    return predictions



def user_based_cf(target_user, ratings):
    similarities = {}
    for user, user_ratings in ratings.items():
        if user == target_user:
            continue
        # Compute similarity using common rated items
        common_items = {item: target_user_rating 
                        for item, target_user_rating in ratings[target_user].items() 
                        if target_user_rating is not None and user_ratings.get(item) is not None}
        
        print(target_user,user)
        print("cc",common_items)
        if not common_items:
            continue
        vec1 = {item: ratings[target_user][item] for item in common_items}
        vec2 = {item: user_ratings[item] for item in common_items}
        print(vec1,vec2)
        similarities[user] = cosine_similarity(vec1, vec2)
    # Predict missing ratings
    print("simi",similarities)
    predictions = {}
    for item, rating in ratings[target_user].items():
        if rating is not None:
            continue
        print("jkgdshf",target_user, item, rating)

        numerator, denominator = 0, 0
        for user, sim in similarities.items():
            if ratings[user].get(item) is not None:
                numerator += sim * ratings[user][item]
                denominator += sim
        predictions[item] = numerator / denominator if denominator else None
    return predictions


predictions_user1 = user_based_cf('User1', ratings)
print("Predicted ratings for User1:", predictions_user1)

def hybrid_recommendation(target_user, ratings, item_features, alpha=0.5):
    """
    Hybrid recommendation: combine user-based CF and content-based filtering
    
    Parameters:
    -----------
    target_user : str
        Target user
    ratings : dict
        User ratings dictionary
    item_features : dict
        Item feature vectors
    alpha : float, default=0.5
        Weight for user-based CF (1-alpha for content-based)
    
    Returns:
    --------
    predictions : dict
        Combined predicted ratings
    """
    # Get predictions from both methods
    cf_predictions = user_based_cf(target_user, ratings)
    cb_predictions = content_based_filtering(target_user, ratings, item_features)
    
    # Combine predictions
    combined = {}
    all_items = set(cf_predictions.keys()) | set(cb_predictions.keys())
    
    for item in all_items:
        cf_score = cf_predictions.get(item, 0)
        cb_score = cb_predictions.get(item, 0)
        
        # Normalize scores if both exist
        if cf_score and cb_score:
            # Normalize to 0-5 range
            cf_norm = cf_score / 5.0 if cf_score else 0
            cb_norm = cb_score / 5.0 if cb_score else 0
            combined[item] = (alpha * cf_norm + (1 - alpha) * cb_norm) * 5
        elif cf_score:
            combined[item] = cf_score
        elif cb_score:
            combined[item] = cb_score
    
    return combined


# ============================================================================
# Testing Content-Based Filtering
# ============================================================================

print("\n" + "=" * 60)
print("CONTENT-BASED FILTERING")
print("=" * 60)

# Test content-based recommendations
print("\n1. Content-Based Recommendations for User1:")
print("   Logic: For unrated items, find similarity with rated items")
print("   Prediction = weighted average of ratings, weighted by item similarity")
cb_predictions = content_based_filtering('User1', ratings, item_features)
print(f"   Predicted ratings: {cb_predictions}")

# Test hybrid recommendations
print("\n2. Hybrid Recommendations for User1:")
print("   Combines user-based CF and content-based filtering")
hybrid_predictions = hybrid_recommendation('User1', ratings, item_features, alpha=0.6)
print(f"   Predicted ratings: {hybrid_predictions}")

# Compare all methods
print("\n3. Comparison:")
print(f"   User-based CF:    {predictions_user1}")
print(f"   Content-based:    {cb_predictions}")
print(f"   Hybrid (α=0.6):   {hybrid_predictions}")
