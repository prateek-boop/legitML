"""
Character-Level URL Tokenizer
Converts URLs to integer sequences for the CNN branch
"""

import numpy as np
from typing import List, Union


class URLTokenizer:
    """
    Character-level tokenizer for URL processing.
    Maps each character to a unique integer ID for neural network input.
    """
    
    def __init__(self, max_length: int = 200):
        self.max_length = max_length
        self.char_to_idx = {}
        self.idx_to_char = {}
        self._build_vocabulary()
    
    def _build_vocabulary(self):
        """Build character vocabulary mapping."""
        chars = []
        
        # Lowercase letters (0-25)
        chars.extend([chr(i) for i in range(ord('a'), ord('z') + 1)])
        
        # Uppercase letters (26-51)
        chars.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)])
        
        # Digits (52-61)
        chars.extend([str(i) for i in range(10)])
        
        # Special characters common in URLs (62-73)
        special_chars = [
            '.', '/', '-', '_', ':', '?', '=', '&', 
            '#', '%', '@', '+'
        ]
        chars.extend(special_chars)
        
        # Build mappings (0 is reserved for padding, 1 for unknown)
        self.char_to_idx = {char: idx + 2 for idx, char in enumerate(chars)}
        self.char_to_idx['<PAD>'] = 0
        self.char_to_idx['<UNK>'] = 1
        
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
    
    def tokenize(self, url: str) -> np.ndarray:
        """
        Convert a single URL to integer sequence.
        
        Args:
            url: URL string to tokenize
            
        Returns:
            numpy array of shape (max_length,) with integer IDs
        """
        # Convert each character to its ID
        tokens = []
        for char in url[:self.max_length]:
            if char in self.char_to_idx:
                tokens.append(self.char_to_idx[char])
            else:
                tokens.append(self.char_to_idx['<UNK>'])
        
        # Pad to max_length
        if len(tokens) < self.max_length:
            tokens.extend([0] * (self.max_length - len(tokens)))
        
        return np.array(tokens, dtype=np.int32)
    
    def tokenize_batch(self, urls: List[str]) -> np.ndarray:
        """
        Convert multiple URLs to integer sequences.
        
        Args:
            urls: List of URL strings
            
        Returns:
            numpy array of shape (batch_size, max_length)
        """
        return np.array([self.tokenize(url) for url in urls], dtype=np.int32)
    
    def decode(self, tokens: Union[np.ndarray, List[int]]) -> str:
        """Convert integer sequence back to URL string."""
        chars = []
        for token in tokens:
            if token == 0:  # PAD
                break
            char = self.idx_to_char.get(int(token), '?')
            if char not in ['<PAD>', '<UNK>']:
                chars.append(char)
            elif char == '<UNK>':
                chars.append('?')
        return ''.join(chars)


if __name__ == "__main__":
    # Test the tokenizer
    tokenizer = URLTokenizer(max_length=200)
    
    test_urls = [
        "https://www.google.com/search?q=test",
        "http://paypa1-secure-login.xyz/verify",
        "https://example.com/path/to/page#section",
    ]
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print()
    
    for url in test_urls:
        tokens = tokenizer.tokenize(url)
        decoded = tokenizer.decode(tokens)
        print(f"Original: {url}")
        print(f"Tokens shape: {tokens.shape}")
        print(f"First 20 tokens: {tokens[:20]}")
        print(f"Decoded: {decoded}")
        print()
