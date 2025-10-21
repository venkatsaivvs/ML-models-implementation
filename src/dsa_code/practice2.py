import numpy as np
import re

X = np.array([[1,2,3],
               [3,4,5]])

Y = np.array([[4,5,6],
              [7,8,9],
              [1,2,3]])


def dot_product(x:np.ndarray, y:np.ndarray) -> np.ndarray:
    return np.dot(x, y)


def simple_tokenizer(text):
    """
    Simple tokenizer without using regex patterns.
    Splits text on whitespace and removes punctuation.
    """
    # Define punctuation characters to remove
    punctuation = ".,!?;:\"'()[]{}@#$%^&*+=|\\/<>~`"
    
    # Convert to lowercase and remove punctuation
    #remove punctuation without for loop
    cleaned_text = text.lower()
    cleaned_text = re.sub(r'[^\w\s]', ' ', cleaned_text)
    
    cleaned_text = text.lower()

    for punct in punctuation:
        cleaned_text = cleaned_text.replace(punct, ' ')
    
    # Split on whitespace and filter out empty strings
    tokens = [token for token in cleaned_text.split() if token]
    
    return tokens




# Example usage
if __name__ == "__main__":
    sample_text = "Hello, world! I'm learning Python. It's amazing!"
    
    print("Simple tokenizer:")
    print(simple_tokenizer(sample_text))
    
    print("\nAdvanced tokenizer:")
    print(advanced_tokenizer(sample_text))


