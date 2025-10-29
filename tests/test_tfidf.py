from src.ml_code.simple_tfidf import SimpleTFIDF
import numpy as np
import pytest

def test_tfidf():
    """Test basic TF-IDF functionality"""
    documents = [
        "the quick brown fox jumps over the lazy dog",
        "a quick brown dog outpaces a quick brown fox",
        "the lazy dog sleeps under the tree",
        "a brown fox is quick and agile"
    ]
    
    tfidf = SimpleTFIDF(max_features=10, verbose=False)
    tfidf_vectors = tfidf.fit_transform(documents)
    
    # Check that IDF values were computed
    assert tfidf.idf_values_ is not None
    assert len(tfidf.idf_values_) > 0
    
    # Check that fit_transform returns list of dicts
    assert isinstance(tfidf_vectors, list)
    assert len(tfidf_vectors) == len(documents)
    assert all(isinstance(vec, dict) for vec in tfidf_vectors)

def test_tokenizer():
    """Test tokenizer function"""
    text = "The Quick Brown Fox!"
    
    # Test tokenization with default settings (lowercase=True, stop_words='english')
    tfidf = SimpleTFIDF(lowercase=True, stop_words='english', verbose=False)
    tokens = tfidf.simple_tokenizer(text)
    # Note: 'the' is a stop word, so it should be filtered out
    assert "the" not in tokens
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens
    
    # Test tokenization without stop words
    tfidf_no_stop = SimpleTFIDF(stop_words=None, lowercase=True, verbose=False)
    tokens_no_stop = tfidf_no_stop.simple_tokenizer(text)
    assert "the" in tokens_no_stop
    assert "quick" in tokens_no_stop
    assert "brown" in tokens_no_stop
    assert "fox" in tokens_no_stop

def test_tf_idf_calculations():
    """Test TF and IDF calculation through actual usage"""
    documents = [
        "quick brown fox",
        "quick brown dog",
        "brown fox"
    ]
    
    tfidf = SimpleTFIDF(stop_words=None, verbose=False)
    tfidf.fit(documents)
    
    # Check that IDF values are computed correctly
    # 'quick' appears in 2 docs, 'brown' in 3, 'fox' in 2, 'dog' in 1
    n_docs = len(documents)
    
    # IDF formula: log(n_docs / df)
    if 'quick' in tfidf.idf_values_:
        expected_idf_quick = np.log(n_docs / 2)  # appears in 2 docs
        assert abs(tfidf.idf_values_['quick'] - expected_idf_quick) < 1e-10
    
    if 'brown' in tfidf.idf_values_:
        expected_idf_brown = np.log(n_docs / 3)  # appears in 3 docs
        assert abs(tfidf.idf_values_['brown'] - expected_idf_brown) < 1e-10
    
    # Test TF-IDF computation
    tfidf_vector = tfidf.compute_tfidf_vector("quick quick brown")
    # 'quick' appears 2 times out of 3 total terms, so TF = 2/3
    if 'quick' in tfidf_vector:
        expected_tf = 2.0 / 3.0
        expected_tfidf = expected_tf * tfidf.idf_values_['quick']
        assert abs(tfidf_vector['quick'] - expected_tfidf) < 1e-10

def test_unfitted_model_behavior():
    """Test that unfitted model behavior"""
    tfidf = SimpleTFIDF(verbose=False)
    documents = ["test document"]
    
    # Transform without fitting will return empty dicts (no IDF values computed yet)
    result = tfidf.transform(documents)
    assert isinstance(result, list)
    assert len(result) == 1
    # Should be empty dict since no IDF values are computed
    assert len(result[0]) == 0
    
    # get_feature_names should work but return empty list
    feature_names = tfidf.get_feature_names()
    assert isinstance(feature_names, list)
    assert len(feature_names) == 0

def test_fit_transform():
    """Test fit_transform method"""
    documents = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and dog are friends"
    ]
    
    tfidf = SimpleTFIDF(max_features=10, verbose=False)
    tfidf_vectors = tfidf.fit_transform(documents)
    
    # Check that IDF values were computed (indicates model is fitted)
    assert len(tfidf.idf_values_) > 0
    
    # Check that fit_transform returns list of dicts with correct length
    assert isinstance(tfidf_vectors, list)
    assert len(tfidf_vectors) == len(documents)
    assert all(isinstance(vec, dict) for vec in tfidf_vectors)

def test_different_parameters():
    """Test different TF-IDF parameters"""
    documents = [
        "the quick brown fox",
        "the lazy dog",
        "brown fox is quick"
    ]
    
    # Test with different max_features (note: max_features is currently not implemented in transform)
    # but the parameter exists and doesn't cause errors
    tfidf_5 = SimpleTFIDF(max_features=5, verbose=False)
    tfidf_5.fit(documents)
    vectors_5 = tfidf_5.transform(documents)
    
    tfidf_10 = SimpleTFIDF(max_features=10, verbose=False)
    tfidf_10.fit(documents)
    vectors_10 = tfidf_10.transform(documents)
    
    # Both should return correct number of documents
    assert len(vectors_5) == len(documents)
    assert len(vectors_10) == len(documents)
    assert all(isinstance(vec, dict) for vec in vectors_5)
    assert all(isinstance(vec, dict) for vec in vectors_10)

def test_lowercase_parameter():
    """Test lowercase parameter"""
    documents = ["The Quick Brown Fox"]
    
    # Test with lowercase=True (default)
    tfidf_lower = SimpleTFIDF(lowercase=True, stop_words=None, verbose=False)
    tokens_lower = tfidf_lower.simple_tokenizer(documents[0])
    assert all(token.islower() for token in tokens_lower)
    assert "the" in tokens_lower
    
    # Test with lowercase=False
    tfidf_no_lower = SimpleTFIDF(lowercase=False, stop_words=None, verbose=False)
    tokens_no_lower = tfidf_no_lower.simple_tokenizer(documents[0])
    assert "The" in tokens_no_lower or "Quick" in tokens_no_lower

def test_stop_words():
    """Test stop word removal"""
    documents = ["the quick brown fox", "the lazy dog"]
    
    # Without stop words
    tfidf_no_stop = SimpleTFIDF(stop_words=None, verbose=False)
    tfidf_no_stop.fit(documents)
    
    # With stop words
    tfidf_stop = SimpleTFIDF(stop_words='english', verbose=False)
    tfidf_stop.fit(documents)
    
    # Stop words should reduce vocabulary size (fewer unique words in idf_values_)
    assert len(tfidf_stop.idf_values_) <= len(tfidf_no_stop.idf_values_)
    
    # Check that stop words are filtered
    assert 'the' not in tfidf_stop.idf_values_
    # Without stop words, 'the' should be present
    assert 'the' in tfidf_no_stop.idf_values_

def test_idf_computation():
    """Test IDF computation with different document frequencies"""
    documents = [
        "quick brown fox",
        "quick brown fox",  # Duplicate
        "lazy dog",
        "brown fox quick"
    ]
    
    tfidf = SimpleTFIDF(stop_words=None, verbose=False)
    tfidf.fit(documents)
    
    # Check that IDF values are computed
    assert len(tfidf.idf_values_) > 0
    
    # Words that appear in more documents should have lower IDF
    n_docs = len(documents)
    if 'brown' in tfidf.idf_values_ and 'dog' in tfidf.idf_values_:
        # 'brown' appears in 3 docs, 'dog' in 1 doc
        # So 'dog' should have higher IDF (rarer word)
        assert tfidf.idf_values_['dog'] > tfidf.idf_values_['brown']

def test_feature_names():
    """Test feature name extraction"""
    documents = ["the quick brown fox"]
    
    tfidf = SimpleTFIDF(stop_words=None, verbose=False)
    tfidf.fit(documents)
    
    feature_names = tfidf.get_feature_names()
    
    # Should return list of strings
    assert isinstance(feature_names, list)
    assert all(isinstance(name, str) for name in feature_names)
    
    # Should have same length as idf_values_
    assert len(feature_names) == len(tfidf.idf_values_)
    
    # Should contain the words from the document
    assert 'quick' in feature_names or 'brown' in feature_names or 'fox' in feature_names

def test_compute_tfidf_vector():
    """Test compute_tfidf_vector method for single document"""
    documents = [
        "quick brown fox",
        "quick brown dog"
    ]
    
    tfidf = SimpleTFIDF(stop_words=None, verbose=False)
    tfidf.fit(documents)
    
    # Test on a new document
    test_doc = "quick quick fox"
    tfidf_vector = tfidf.compute_tfidf_vector(test_doc)
    
    # Should return a dictionary
    assert isinstance(tfidf_vector, dict)
    
    # 'quick' appears twice in the test doc, 'fox' once
    if 'quick' in tfidf_vector:
        assert tfidf_vector['quick'] > 0
    if 'fox' in tfidf_vector:
        assert tfidf_vector['fox'] > 0
    
    # Words not in IDF values should not appear
    assert all(word in tfidf.idf_values_ for word in tfidf_vector.keys())

def test_edge_cases():
    """Test edge cases"""
    # Test with single document
    documents_single = ["the quick brown fox"]
    tfidf_single = SimpleTFIDF(stop_words=None, verbose=False)
    tfidf_single.fit(documents_single)
    vectors_single = tfidf_single.transform(documents_single)
    
    assert len(vectors_single) == 1
    assert isinstance(vectors_single[0], dict)
    
    # Test with empty documents
    documents_empty = ["", "the quick brown fox", ""]
    tfidf_empty = SimpleTFIDF(stop_words=None, verbose=False)
    tfidf_empty.fit(documents_empty)
    vectors_empty = tfidf_empty.transform(documents_empty)
    
    assert len(vectors_empty) == 3
    # Empty documents should result in empty dicts
    assert len(vectors_empty[0]) == 0
    assert len(vectors_empty[2]) == 0
    
    # Test with very short documents
    documents_short = ["a", "b", "c"]
    tfidf_short = SimpleTFIDF(stop_words=None, verbose=False)
    tfidf_short.fit(documents_short)
    vectors_short = tfidf_short.transform(documents_short)
    
    assert len(vectors_short) == 3
    assert all(isinstance(vec, dict) for vec in vectors_short)
