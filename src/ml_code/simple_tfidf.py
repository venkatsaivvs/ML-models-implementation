"""
tf - formula: count of term in document / total number of terms in document
idf - formula: np.log(total_documents / documents_containing_term)

list all functions and classes in the file
1. simple_tokenizer - tokenize text without regex patterns
2. compute_idf - compute IDF values from documents
3. compute_tfidf_vector - compute TF-IDF for a single document (returns dict)
4. fit - fit the TF-IDF model to documents
5. transform - transform documents to TF-IDF representations (returns list of dicts)
6. fit_transform - fit and transform in one step
7. get_feature_names - get all words with IDF values
"""

import numpy as np
from collections import Counter, defaultdict

class SimpleTFIDF:
    """
    Simplified TF-IDF Vectorizer - Easy to implement and understand
    Perfect for interview preparation in 40 minutes
    """
    
    # Default English stop words
    DEFAULT_STOP_WORDS = {'the', 'is', 'on', 'and', 'are', 'a', 'an', 'in', 'of', 'to', 
                          'at', 'be', 'by', 'for', 'from', 'has', 'he', 'it', 'its', 
                          'or', 'that', 'this', 'was', 'will', 'with'}
    
    def __init__(self, max_features=None, lowercase=True, stop_words='english', verbose=False):
        self.max_features = max_features
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
    
    def simple_tokenizer(self, text):
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
    
    def compute_idf(self, documents):
        """
        Compute IDF values directly from documents
        Simple approach for interview prep
        """
        if self.verbose:
            print("Computing IDF values...")
        
        n_docs = len(documents)
        doc_counts = defaultdict(int)  # How many docs contain each word
        
        # Count document frequency for each word
        for doc in documents:
            tokens = self.simple_tokenizer(doc)
            unique_tokens = set(tokens)  # Only count each word once per document
            
            for token in unique_tokens:
                doc_counts[token] += 1
        
        # Compute IDF for each word
        for word, df in doc_counts.items():
            self.idf_values_[word] = np.log(n_docs / df)
        
        if self.verbose:
            print(f"IDF values computed for {len(self.idf_values_)} words")
    
    def compute_tfidf_vector(self, document):
        """
        Compute TF-IDF vector for a single document
        Returns a dictionary: {word: tfidf_value}
        """
        tokens = self.simple_tokenizer(document)
        
        # Count term frequencies
        tf_counts = Counter(tokens)
        total_terms = len(tokens)
        
        # Compute TF-IDF for each word
        tfidf_dict = {}
        for word, count in tf_counts.items():
            if word in self.idf_values_:
                tf = count / total_terms  # Term frequency
                idf = self.idf_values_[word]  # Inverse document frequency
                tfidf_dict[word] = tf * idf
        
        return tfidf_dict
    
    def fit(self, documents):
        """
        Fit the TF-IDF model to documents
        """
        if self.verbose:
            print("Fitting TF-IDF model...")
        
        # Compute IDF values
        self.compute_idf(documents)
        
        if self.verbose:
            print("TF-IDF model fitted successfully!")
        
        return self
    
    def transform(self, documents):
        """
        Transform documents to TF-IDF representations
        Returns list of dictionaries: [{word: tfidf_value}, ...]
        """
        if self.verbose:
            print("Transforming documents...")
        
        tfidf_vectors = []
        for doc in documents:
            tfidf_dict = self.compute_tfidf_vector(doc)
            tfidf_vectors.append(tfidf_dict)
        
        return tfidf_vectors
    
    def fit_transform(self, documents):
        """
        Fit and transform in one step
        """
        return self.fit(documents).transform(documents)
    
    def get_feature_names(self):
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
        "the mat is on the floor"
    ]
    
    # Example 1: With default English stop words
    print("=" * 60)
    print("WITH Default English Stop Words:")
    print("=" * 60)
    tfidf = SimpleTFIDF(stop_words='english', verbose=True)
    tfidf_vectors = tfidf.fit_transform(documents)
    
    print("\nTF-IDF Vectors (list of dicts):")
    for i, vec in enumerate(tfidf_vectors):
        print(f"Document {i}: {vec}")
    
    print("\nIDF Values:")
    print(tfidf.idf_values_)
    
    # Example 2: Without stop words
    print("\n" + "=" * 60)
    print("WITHOUT Stop Words:")
    print("=" * 60)
    tfidf_none = SimpleTFIDF(stop_words=None, verbose=True)
    tfidf_vectors_none = tfidf_none.fit_transform(documents)
    
    print("\nFirst document TF-IDF:")
    print(tfidf_vectors_none[0])
