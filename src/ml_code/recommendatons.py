ratings = {
    'User1': {'Item1': 5, 'Item2': None, 'Item3': 3},
    'User2': {'Item1': 5, 'Item2': 2, 'Item3': 3},
    'User3': {'Item1': None, 'Item2': 5, 'Item3': 10},
    'User4': {'Item1':4, 'Item2':None, "Item3":3}
}


import math

# Cosine similarity
def cosine_similarity(vec1, vec2):
    common_keys = set(vec1.keys()) & set(vec2.keys())
    if not common_keys:
        return 0
    dot = sum(vec1[k] * vec2[k] for k in common_keys)
    norm1 = math.sqrt(sum(vec1[k]**2 for k in common_keys))
    norm2 = math.sqrt(sum(vec2[k]**2 for k in common_keys))
    return dot / (norm1 * norm2) if norm1 and norm2 else 0

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
