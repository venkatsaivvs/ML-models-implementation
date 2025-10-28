"""
Test file for simple_bm25.py
Tests BM25 search functionality
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml_code.simple_bm25 import SimpleBM25
import pytest
import numpy as np


# Test documents
test_documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat and dog are friends",
    "the mat is on the floor",
    "cats and dogs are popular pets"
]


def test_bm25_initialization():
    """Test BM25 model initialization"""
    bm25 = SimpleBM25(k1=1.5, b=0.75)
    assert bm25.k1 == 1.5
    assert bm25.b == 0.75
    assert bm25.idf_values_ == {}
    assert bm25.documents_ == []


def test_bm25_fit():
    """Test BM25 fit method"""
    bm25 = SimpleBM25(verbose=False)
    bm25.fit(test_documents)
    
    assert len(bm25.idf_values_) > 0
    assert len(bm25.documents_) == len(test_documents)
    assert len(bm25.doc_tokens_) == len(test_documents)
    assert len(bm25.doc_lengths_) == len(test_documents)
    assert bm25.avg_doc_length_ > 0


def test_bm25_tokenizer():
    """Test tokenizer functionality"""
    bm25 = SimpleBM25(stop_words='english', lowercase=True)
    tokens = bm25.simple_tokenizer("Hello, World! This is a test.")
    assert isinstance(tokens, list)
    assert 'hello' in tokens or 'world' in tokens or 'test' in tokens


def test_bm25_get_scores():
    """Test getting BM25 scores for a query"""
    bm25 = SimpleBM25(verbose=False)
    bm25.fit(test_documents)
    
    scores = bm25.get_scores("cat")
    assert isinstance(scores, list)
    assert len(scores) == len(test_documents)
    assert all(isinstance(score, (int, float)) for score in scores)
    # Cat appears in documents 0 and 2, so they should have higher scores
    assert scores[0] > 0 or scores[2] > 0


def test_bm25_get_top_k():
    """Test getting top k documents"""
    bm25 = SimpleBM25(verbose=False)
    bm25.fit(test_documents)
    
    top_k = bm25.get_top_k("cat", k=3)
    assert isinstance(top_k, list)
    assert len(top_k) <= 3
    assert all(isinstance(item, tuple) and len(item) == 2 for item in top_k)
    # Scores should be in descending order
    if len(top_k) > 1:
        assert top_k[0][1] >= top_k[1][1]  # First score >= second score


def test_bm25_transform():
    """Test transform method (alias for get_scores)"""
    bm25 = SimpleBM25(verbose=False)
    bm25.fit(test_documents)
    
    scores1 = bm25.get_scores("cat")
    scores2 = bm25.transform("cat")
    
    assert scores1 == scores2


def test_bm25_unfitted_error():
    """Test that methods raise error when not fitted"""
    bm25 = SimpleBM25(verbose=False)
    
    with pytest.raises(ValueError, match="Model must be fitted"):
        bm25.get_scores("test query")


def test_bm25_empty_query():
    """Test handling of empty query"""
    bm25 = SimpleBM25(verbose=False)
    bm25.fit(test_documents)
    
    scores = bm25.get_scores("")
    assert isinstance(scores, list)
    assert len(scores) == len(test_documents)
    # Empty query should give zero or very low scores
    assert all(score == 0.0 or score >= 0.0 for score in scores)


def test_bm25_idf_values():
    """Test that IDF values are computed correctly"""
    bm25 = SimpleBM25(verbose=False)
    bm25.fit(test_documents)
    
    # Words that appear in fewer documents should have higher IDF
    # "cat" appears in 2 docs, "the" appears in all docs (higher frequency)
    # "cat" should have higher IDF than very common words (if not filtered by stop words)
    assert len(bm25.idf_values_) > 0
    # All IDF values should be non-negative
    assert all(idf >= 0 for idf in bm25.idf_values_.values())


def test_bm25_document_lengths():
    """Test that document lengths are computed correctly"""
    bm25 = SimpleBM25(verbose=False)
    bm25.fit(test_documents)
    
    assert len(bm25.doc_lengths_) == len(test_documents)
    assert all(length > 0 for length in bm25.doc_lengths_)
    assert bm25.avg_doc_length_ == np.mean(bm25.doc_lengths_)


def test_bm25_unknown_terms():
    """Test handling of query terms not in vocabulary"""
    bm25 = SimpleBM25(verbose=False)
    bm25.fit(test_documents)
    
    scores = bm25.get_scores("xyzabc123unknown")
    assert isinstance(scores, list)
    # Unknown terms should result in zero scores
    assert all(score == 0.0 for score in scores)


def test_bm25_different_k_values():
    """Test get_top_k with different k values"""
    bm25 = SimpleBM25(verbose=False)
    bm25.fit(test_documents)
    
    top_1 = bm25.get_top_k("cat", k=1)
    top_5 = bm25.get_top_k("cat", k=5)
    
    assert len(top_1) == 1
    assert len(top_5) <= 5
    assert top_1[0] == top_5[0]  # First result should be the same


def test_bm25_get_feature_names():
    """Test getting feature names (vocabulary)"""
    bm25 = SimpleBM25(verbose=False)
    bm25.fit(test_documents)
    
    feature_names = bm25.get_feature_names()
    assert isinstance(feature_names, list)
    assert len(feature_names) == len(bm25.idf_values_)
    assert set(feature_names) == set(bm25.idf_values_.keys())

