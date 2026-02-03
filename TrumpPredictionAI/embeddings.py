import pandas as pd
import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, List, Union

# Handle imports that might be flagged by linter if not in env
try:
    from sentence_transformers import SentenceTransformer  
except ImportError:
    SentenceTransformer = None  # ty:ignore[invalid-assignment]

import gensim.downloader as api

class BertVectorizer(BaseEstimator, TransformerMixin):
    """
    Vectorizes text using Sentence Transformers (MiniLM).
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model: Any = None # Type hint to Any to avoid linter confusion
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def fit(self, X, y=None):
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed.")
            
        print(f"[*] Loading SentenceTransformer: {self.model_name}")
        print(f"    Device: {self.device}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        return self

    def transform(self, X):
        # Auto-fit if called directly
        if self.model is None:
            self.fit(X)
            
        # Safety check for linter
        if self.model is None:
            raise RuntimeError("Model failed to load.")

        # Convert to list of strings
        if isinstance(X, pd.Series):
            texts = X.astype(str).tolist()
        elif isinstance(X, pd.DataFrame):
            texts = X.iloc[:, 0].astype(str).tolist()
        else:
            texts = list(X)
            
        print(f"[*] Encoding {len(texts)} texts with BERT...")
        # Linter might still complain about .encode if type is Any, but it's better
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        return embeddings

class GloveVectorizer(BaseEstimator, TransformerMixin):
    """
    Vectorizes text using GloVe (averaging word vectors).
    Uses gensim to load 'glove-wiki-gigaword-50' (lightweight, 50 dim).
    """
    def __init__(self, model_name: str = 'glove-wiki-gigaword-50'):
        self.model_name = model_name
        self.glove: Any = None # Type hint

    def fit(self, X, y=None):
        print(f"[*] Loading GloVe model: {self.model_name} (this may take a moment)...")
        self.glove = api.load(self.model_name)
        print("    [OK] GloVe loaded.")
        return self

    def transform(self, X):
        if self.glove is None:
            self.fit(X)
        
        # Safety check
        if self.glove is None:
             raise RuntimeError("GloVe model failed to load.")
            
        if isinstance(X, pd.Series):
            texts = X.astype(str).tolist()
        else:
            texts = list(X)
            
        print(f"[*] Encoding {len(texts)} texts with GloVe...")
        vectors = []
        # Get vector size safely
        vector_size = getattr(self.glove, 'vector_size', 50) 
        
        for text in texts:
            # Simple tokenization and averaging
            words = text.lower().split()
            # Check for word existence safely
            word_vecs = [self.glove[w] for w in words if w in self.glove]
            
            if len(word_vecs) > 0:
                vectors.append(np.mean(word_vecs, axis=0))
            else:
                # Fallback for empty/unknown text: zero vector
                vectors.append(np.zeros(vector_size))
                
        return np.array(vectors)
