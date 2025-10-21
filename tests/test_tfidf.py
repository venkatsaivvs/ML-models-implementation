from src.mlcode.tfidf import TFIDF
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
    
    tfidf = TFIDF(max_features=10)
    tfidf_matrix = tfidf.fit_transform(documents)
    
    assert tfidf.vocabulary_ is not None
    assert tfidf.idf_ is not None
    assert tfidf.is_fitted_ == True
    assert tfidf_matrix.shape[0] == len(documents)

def test_preprocessing_functions():
    """Test text preprocessing functions"""
    text = "The Quick Brown Fox!"
    
    # Test preprocessing
    processed = TFIDF.preprocess_text(text, lowercase=True, remove_punctuation=True)
    assert processed == "the quick brown fox"
    
    # Test tokenization
    tokens = TFIDF.tokenize(processed)
    assert tokens == ["the", "quick", "brown", "fox"]

def test_tf_idf_calculations():
    """Test TF and IDF calculation functions"""
    # Test TF calculation
    tf = TFIDF.compute_tf(term_count=3, total_terms=10)
    assert tf == 0.3
    
    # Test IDF calculation (smoothed)
    idf_smooth = TFIDF.compute_idf(n_docs=100, doc_freq=10, smooth=True)
    expected_idf = np.log((100 + 1) / (10 + 1)) + 1
    assert abs(idf_smooth - expected_idf) < 1e-10
    
    # Test IDF calculation (not smoothed)
    idf_no_smooth = TFIDF.compute_idf(n_docs=100, doc_freq=10, smooth=False)
    expected_idf = np.log(100 / 10)
    assert abs(idf_no_smooth - expected_idf) < 1e-10

def test_unfitted_model_errors():
    """Test that unfitted model raises appropriate errors"""
    tfidf = TFIDF()
    documents = ["test document"]
    
    # All methods should raise errors before fitting
    with pytest.raises(ValueError, match="TF-IDF must be fitted before transforming"):
        tfidf.transform(documents)
    
    with pytest.raises(ValueError, match="TF-IDF must be fitted before getting feature names"):
        tfidf.get_feature_names()

def test_fit_transform():
    """Test fit_transform method"""
    documents = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and dog are friends"
    ]
    
    tfidf = TFIDF(max_features=10, verbose=False)
    tfidf_matrix = tfidf.fit_transform(documents)
    
    # Check that model is fitted
    assert tfidf.is_fitted_ == True
    
    # Check matrix shape
    assert tfidf_matrix.shape[0] == len(documents)
    assert tfidf_matrix.shape[1] <= 10  # max_features

def test_different_parameters():
    """Test different TF-IDF parameters"""
    documents = [
        "the quick brown fox",
        "the lazy dog",
        "brown fox is quick"
    ]
    
    # Test with different max_features
    tfidf_5 = TFIDF(max_features=5, verbose=False)
    tfidf_5.fit(documents)
    matrix_5 = tfidf_5.transform(documents)
    
    tfidf_10 = TFIDF(max_features=10, verbose=False)
    tfidf_10.fit(documents)
    matrix_10 = tfidf_10.transform(documents)
    
    assert matrix_5.shape[1] <= 5
    assert matrix_10.shape[1] <= 10

def test_ngram_ranges():
    """Test different n-gram ranges"""
    documents = ["the quick brown fox"]
    
    # Test unigrams (1,1)
    tfidf_uni = TFIDF(ngram_range=(1, 1), verbose=False)
    tfidf_uni.fit(documents)
    
    # Test bigrams (1,2)
    tfidf_bi = TFIDF(ngram_range=(1, 2), verbose=False)
    tfidf_bi.fit(documents)
    
    # Bigrams should have more features
    assert len(tfidf_bi.vocabulary_) >= len(tfidf_uni.vocabulary_)

def test_stop_words():
    """Test stop word removal"""
    documents = ["the quick brown fox", "the lazy dog"]
    
    # Without stop words
    tfidf_no_stop = TFIDF(stop_words=None, verbose=False)
    tfidf_no_stop.fit(documents)
    
    # With stop words
    tfidf_stop = TFIDF(stop_words='english', verbose=False)
    tfidf_stop.fit(documents)
    
    # Stop words should reduce vocabulary size
    assert len(tfidf_stop.vocabulary_) <= len(tfidf_no_stop.vocabulary_)

def test_min_max_df():
    """Test min_df and max_df parameters"""
    documents = [
        "the quick brown fox",
        "the quick brown fox",  # Duplicate
        "the lazy dog",
        "brown fox is quick"
    ]
    
    # Test min_df
    tfidf_min = TFIDF(min_df=2, verbose=False)
    tfidf_min.fit(documents)
    
    # Test max_df
    tfidf_max = TFIDF(max_df=0.5, verbose=False)  # 50% of documents
    tfidf_max.fit(documents)
    
    # Both should produce valid results
    assert tfidf_min.is_fitted_ == True
    assert tfidf_max.is_fitted_ == True

def test_feature_names():
    """Test feature name extraction"""
    documents = ["the quick brown fox"]
    
    tfidf = TFIDF(verbose=False)
    tfidf.fit(documents)
    
    feature_names = tfidf.get_feature_names()
    
    # Should return list of strings
    assert isinstance(feature_names, list)
    assert all(isinstance(name, str) for name in feature_names)
    
    # Should have same length as vocabulary
    assert len(feature_names) == len(tfidf.vocabulary_)

def test_inverse_transform():
    """Test inverse transform functionality"""
    documents = [
        "the quick brown fox",
        "the lazy dog"
    ]
    
    tfidf = TFIDF(verbose=False)
    tfidf_matrix = tfidf.fit_transform(documents)
    
    # Test inverse transform
    reconstructed = tfidf.inverse_transform(tfidf_matrix)
    
    # Should return list of strings
    assert isinstance(reconstructed, list)
    assert len(reconstructed) == len(documents)
    assert all(isinstance(doc, str) for doc in reconstructed)

def test_parameter_management():
    """Test parameter get/set methods"""
    tfidf = TFIDF(max_features=10, min_df=2, verbose=False)
    
    # Test get_params
    params = tfidf.get_params()
    assert 'max_features' in params
    assert 'min_df' in params
    assert params['max_features'] == 10
    assert params['min_df'] == 2
    
    # Test set_params
    result = tfidf.set_params(max_features=20, min_df=1)
    assert result is tfidf  # Should return self
    assert tfidf.max_features == 20
    assert tfidf.min_df == 1

def test_edge_cases():
    """Test edge cases"""
    # Test with single document
    documents_single = ["the quick brown fox"]
    tfidf_single = TFIDF(verbose=False)
    tfidf_single.fit(documents_single)
    matrix_single = tfidf_single.transform(documents_single)
    
    assert matrix_single.shape[0] == 1
    
    # Test with empty documents
    documents_empty = ["", "the quick brown fox", ""]
    tfidf_empty = TFIDF(verbose=False)
    tfidf_empty.fit(documents_empty)
    matrix_empty = tfidf_empty.transform(documents_empty)
    
    assert matrix_empty.shape[0] == 3
    
    # Test with very short documents
    documents_short = ["a", "b", "c"]
    tfidf_short = TFIDF(verbose=False)
    tfidf_short.fit(documents_short)
    matrix_short = tfidf_short.transform(documents_short)
    
    assert matrix_short.shape[0] == 3
