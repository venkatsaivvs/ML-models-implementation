"""
BM25 (Best Matching 25) Implementation for Interview Preparation
BM25 is a ranking function used in information retrieval, improved over TF-IDF

BM25 term score: 
log((N - n(q) + 0.5) / (n(q) + 0.5))
N - total number of documents
n(q) - number of documents containing term q

BM25 formula:
BM25(d, q) = Σ IDF(qᵢ) * (f(qᵢ, d) * (k₁ + 1)) / (f(qᵢ, d) + k₁ * (1 - b + b * (|d| / avgdl)))
numerator = f * (self.k1 + 1)
denominator = f + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length_))            
term_score = idf * (numerator / denominator)
f is frequency of term in document
k1 is term frequency saturation parameter
b is length normalization parameter
avgdl is average document length in the collection
idf is inverse document frequency
numerator is the numerator of the BM25 formula
denominator is the denominator of the BM25 formula
term_score is the BM25 score for the term


Functions and classes:
1. simple_tokenizer - tokenize text without regex patterns
2. compute_idf - compute BM25 IDF values from documents
3. compute_bm25_score - compute BM25 score for a query against a document
4. fit - fit the BM25 model to documents
5. transform - transform query to BM25 scores for all documents
6. get_scores - get BM25 scores for a query
7. get_top_k - get top k documents for a query
"""

import numpy as np
from collections import Counter, defaultdict


class SimpleBM25:
    """
    Simplified BM25 Vectorizer - Easy to implement and understand
    Perfect for interview preparation
    """
    
    # Default English stop words
    DEFAULT_STOP_WORDS = {'the', 'is', 'on', 'and', 'are', 'a', 'an', 'in', 'of', 'to', 
                          'at', 'be', 'by', 'for', 'from', 'has', 'he', 'it', 'its', 
                          'or', 'that', 'this', 'was', 'will', 'with'}
    
    def __init__(self, k1: float = 1.5, b: float = 0.75, lowercase: bool = True, 
                 stop_words: str | None = 'english', verbose: bool = False):
        """
        Initialize BM25 model
        
        Parameters:
        -----------
        k1 : float, default=1.5
            Term frequency saturation parameter
        b : float, default=0.75
            Length normalization parameter (0 = no normalization, 1 = full normalization)
        lowercase : bool, default=True
            Convert text to lowercase
        stop_words : str or None, default='english'
            Stop words to remove ('english' or None)
        verbose : bool, default=False
            Print progress information
        """
        self.k1 = k1
        self.b = b
        self.lowercase = lowercase
        
        # Handle stop_words parameter
        if stop_words == 'english':
            self.stop_words = self.DEFAULT_STOP_WORDS
        elif stop_words is None:
            self.stop_words = set()
        else:
            self.stop_words = stop_words
        
        self.verbose = verbose
        
        # Will be set during fit
        self.idf_values_ = {}
        self.doc_lengths_ = []  # Length of each document
        self.avg_doc_length_ = 0.0
        self.documents_ = []
        self.doc_tokens_ = []  # Tokenized documents
    
    def simple_tokenizer(self, text: str) -> list[str]:
        """
        Simple tokenizer without regex patterns
        """
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation by replacing with spaces
        punctuation = ".,!?;:\"'()[]{}@#$%^&*+=|\\/<>~`"
        for punct in punctuation:
            text = text.replace(punct, ' ')
        
        # Split and filter empty strings and stop words
        tokens = [token for token in text.split() if token and token not in self.stop_words]
        return tokens
    
    def compute_idf(self, documents: list[str]) -> None:
        """
        Compute BM25 IDF values from documents
        BM25 IDF formula: log((N - n(q) + 0.5) / (n(q) + 0.5))
        """
        if self.verbose:
            print("Computing BM25 IDF values...")
        
        n_docs = len(documents)
        doc_counts = defaultdict(int)  # How many docs contain each word
        
        # Count document frequency for each word
        for doc in documents:
            tokens = self.simple_tokenizer(doc)
            unique_tokens = set(tokens)  # Only count each word once per document
            
            for token in unique_tokens:
                doc_counts[token] += 1
        
        # Compute BM25 IDF for each word
        for word, df in doc_counts.items():
            # BM25 IDF: log((N - n(q) + 0.5) / (n(q) + 0.5))
            self.idf_values_[word] = np.log((n_docs - df + 0.5) / (df + 0.5))
        
        if self.verbose:
            print(f"BM25 IDF values computed for {len(self.idf_values_)} words")
    
    def compute_bm25_score(self, query_tokens: list[str], doc_tokens: list[str], 
                          doc_length: int) -> float:
        # Count term frequencies in document
        doc_tf = Counter(doc_tokens)
        
        score = 0.0
        
        # Calculate BM25 for each query term
        for term in query_tokens:
            if term not in self.idf_values_:
                continue  # Skip unknown terms
            
            # Term frequency in document
            f = doc_tf.get(term, 0)
            
            if f == 0:
                continue  # Term not in document
            
            # BM25 term score: IDF * (f * (k1 + 1)) / (f + k1 * (1 - b + b * (|d| / avgdl)))
            idf = self.idf_values_[term]
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length_))
            
            term_score = idf * (numerator / denominator)
            score += term_score
        
        return float(score)
    
    def fit(self, documents: list[str]) -> "SimpleBM25":
        """
        Fit the BM25 model to documents
        
        Parameters:
        -----------
        documents : list of str
            Training documents
        """
        if self.verbose:
            print("Fitting BM25 model...")
        
        # Store documents
        self.documents_ = documents
        
        # Compute IDF values
        self.compute_idf(documents)
        
        # Compute document lengths and average
        self.doc_tokens_ = []
        self.doc_lengths_ = []
        
        for doc in documents:
            tokens = self.simple_tokenizer(doc)
            self.doc_tokens_.append(tokens)
            self.doc_lengths_.append(len(tokens))
        
        self.avg_doc_length_ = np.mean(self.doc_lengths_) if self.doc_lengths_ else 0.0
        
        if self.verbose:
            print(f"BM25 model fitted successfully!")
            print(f"  - Number of documents: {len(documents)}")
            print(f"  - Average document length: {self.avg_doc_length_:.2f}")
        
        return self
    
    def get_scores(self, query: str) -> list[float]:
        """
        Get BM25 scores for a query against all documents
        
        Parameters:
        -----------
        query : str
            Search query
        
        Returns:
        --------
        scores : list of float
            BM25 scores for each document
        """
        if not self.documents_:
            raise ValueError("Model must be fitted before scoring. Call fit() first.")
        
        query_tokens = self.simple_tokenizer(query)
        scores = []
        
        for i, doc_tokens in enumerate(self.doc_tokens_):
            doc_length = self.doc_lengths_[i]
            score = self.compute_bm25_score(query_tokens, doc_tokens, doc_length)
            scores.append(score)
        
        return scores
    
    def get_top_k(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """
        Get top k documents for a query
        
        Parameters:
        -----------
        query : str
            Search query
        k : int, default=5
            Number of top documents to return
        
        Returns:
        --------
        top_k : list of tuple (document_index, score)
            Top k documents with their scores, sorted by score (descending)
        """
        scores = self.get_scores(query)
        
        # Get indices sorted by score (descending)
        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        return indexed_scores[:k]
    
    def transform(self, query: str) -> list[float]:
        """
        Transform query to BM25 scores for all documents
        Alias for get_scores() to match sklearn-style interface
        
        Parameters:
        -----------
        query : str
            Search query
        
        Returns:
        --------
        scores : list of float
            BM25 scores for each document
        """
        return self.get_scores(query)
    
    def get_feature_names(self) -> list[str]:
        """
        Get all words that have IDF values
        """
        return list(self.idf_values_.keys())


# Example usage and testing
if __name__ == "__main__":
    # Sample documents
    documents = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and dog are friends",
        "the mat is on the floor",
        "cats and dogs are popular pets"
    ]
    
    print("=" * 60)
    print("DEMONSTRATION: BM25 Search")
    print("=" * 60)
    
    # Create and fit BM25 model
    print("\n1. Creating and fitting BM25 model:")
    bm25 = SimpleBM25(k1=1.5, b=0.75, stop_words='english', verbose=True)
    bm25.fit(documents)
    
    # Test queries
    queries = ["cat mat", "dog friends", "pets"]
    
    print("\n" + "=" * 60)
    print("2. Testing queries:")
    print("=" * 60)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        scores = bm25.get_scores(query)
        print(f"Scores: {[f'{s:.3f}' for s in scores]}")
        
        # Get top 3 results
        top_k = bm25.get_top_k(query, k=3)
        print(f"Top 3 results:")
        for idx, score in top_k:
            print(f"  [{idx}] Score: {score:.3f} - \"{documents[idx]}\"")

