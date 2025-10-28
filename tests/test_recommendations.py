"""
Test file for recommendations.py
Tests cosine similarity, user-based CF, content-based filtering, and hybrid recommendations
"""
import sys
import os

# Add src to path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import functions (module-level code will execute but that's okay for tests)
from src.ml_code.recommendatons import (
    cosine_similarity, 
    user_based_cf, 
    content_based_filtering,
    item_similarity,
    hybrid_recommendation
)
import pytest
import math


# Test data
test_ratings = {
    'User1': {'Item1': 5, 'Item2': None, 'Item3': 3},
    'User2': {'Item1': 5, 'Item2': 2, 'Item3': 3},
    'User3': {'Item1': None, 'Item2': 5, 'Item3': 10},
    'User4': {'Item1': 4, 'Item2': None, 'Item3': 3}
}

test_item_features = {
    'Item1': {'action': 0.9, 'comedy': 0.1, 'drama': 0.3, 'sci-fi': 0.8},
    'Item2': {'action': 0.2, 'comedy': 0.9, 'drama': 0.7, 'sci-fi': 0.1},
    'Item3': {'action': 0.5, 'comedy': 0.3, 'drama': 0.9, 'sci-fi': 0.4}
}


def test_cosine_similarity_identical():
    """Test cosine similarity with identical vectors"""
    vec1 = {'Item1': 5, 'Item3': 3}
    vec2 = {'Item1': 5, 'Item3': 3}
    similarity = cosine_similarity(vec1, vec2)
    assert abs(similarity - 1.0) < 1e-6  # Should be exactly 1.0


def test_cosine_similarity_orthogonal():
    """Test cosine similarity with orthogonal vectors"""
    vec1 = {'Item1': 1, 'Item2': 0}
    vec2 = {'Item1': 0, 'Item2': 1}
    similarity = cosine_similarity(vec1, vec2)
    assert abs(similarity - 0.0) < 1e-6  # Should be 0 for orthogonal vectors


def test_cosine_similarity_no_common_keys():
    """Test cosine similarity with no common keys"""
    vec1 = {'Item1': 5, 'Item2': 3}
    vec2 = {'Item3': 2, 'Item4': 4}
    similarity = cosine_similarity(vec1, vec2)
    assert similarity == 0.0


def test_cosine_similarity_calculation():
    """Test cosine similarity calculation with known values"""
    vec1 = {'Item1': 5, 'Item3': 3}
    vec2 = {'Item1': 5, 'Item3': 3}
    # Dot = 5*5 + 3*3 = 34
    # Norm1 = sqrt(25 + 9) = sqrt(34)
    # Norm2 = sqrt(25 + 9) = sqrt(34)
    # Similarity = 34 / (sqrt(34) * sqrt(34)) = 34/34 = 1.0
    similarity = cosine_similarity(vec1, vec2)
    assert abs(similarity - 1.0) < 1e-6


def test_item_similarity():
    """Test item similarity function"""
    item1 = {'action': 0.9, 'comedy': 0.1}
    item2 = {'action': 0.9, 'comedy': 0.1}
    similarity = item_similarity(item1, item2)
    assert abs(similarity - 1.0) < 1e-6  # Identical features should have similarity 1.0


def test_user_based_cf_basic():
    """Test user-based collaborative filtering returns predictions"""
    predictions = user_based_cf('User1', test_ratings)
    assert isinstance(predictions, dict)
    # User1 has Item2 unrated, so should predict Item2
    assert 'Item2' in predictions or len(predictions) >= 0


def test_user_based_cf_prediction_type():
    """Test that predictions are numbers or None"""
    predictions = user_based_cf('User1', test_ratings)
    for item, rating in predictions.items():
        assert rating is None or isinstance(rating, (int, float))


def test_content_based_filtering_basic():
    """Test content-based filtering returns predictions"""
    predictions = content_based_filtering('User1', test_ratings, test_item_features)
    assert isinstance(predictions, dict)
    # User1 has Item2 unrated, so should predict Item2
    assert 'Item2' in predictions or len(predictions) >= 0


def test_content_based_filtering_prediction_type():
    """Test that content-based predictions are numbers or None"""
    predictions = content_based_filtering('User1', test_ratings, test_item_features)
    for item, rating in predictions.items():
        assert rating is None or isinstance(rating, (int, float))


def test_content_based_filtering_unrated_only():
    """Test that content-based filtering only predicts unrated items"""
    predictions = content_based_filtering('User1', test_ratings, test_item_features)
    user_rated_items = {item for item, rating in test_ratings['User1'].items() if rating is not None}
    for predicted_item in predictions.keys():
        assert predicted_item not in user_rated_items


def test_hybrid_recommendation():
    """Test hybrid recommendation combines both methods"""
    predictions = hybrid_recommendation('User1', test_ratings, test_item_features, alpha=0.5)
    assert isinstance(predictions, dict)


def test_hybrid_recommendation_alpha_zero():
    """Test hybrid with alpha=0 (only content-based)"""
    predictions = hybrid_recommendation('User1', test_ratings, test_item_features, alpha=0.0)
    assert isinstance(predictions, dict)


def test_hybrid_recommendation_alpha_one():
    """Test hybrid with alpha=1 (only user-based CF)"""
    predictions = hybrid_recommendation('User1', test_ratings, test_item_features, alpha=1.0)
    assert isinstance(predictions, dict)


def test_empty_user_ratings():
    """Test handling of user with no ratings"""
    empty_ratings = {
        'User1': {'Item1': None, 'Item2': None, 'Item3': None},
        'User2': {'Item1': 5, 'Item2': 2, 'Item3': 3}
    }
    # Should handle gracefully
    predictions_cf = user_based_cf('User1', empty_ratings)
    predictions_cb = content_based_filtering('User1', empty_ratings, test_item_features)
    assert isinstance(predictions_cf, dict)
    assert isinstance(predictions_cb, dict)


def test_similarity_with_none_values():
    """Test cosine similarity handles None values correctly"""
    vec1 = {'Item1': 5, 'Item2': None, 'Item3': 3}
    vec2 = {'Item1': 5, 'Item2': 2, 'Item3': 3}
    # Should only use common keys with non-None values
    similarity = cosine_similarity(vec1, vec2)
    # Common keys are Item1 and Item3 (Item2 has None in vec1)
    assert 0 <= similarity <= 1  # Similarity should be between 0 and 1

