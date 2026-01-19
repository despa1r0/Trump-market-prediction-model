"""
Feature Engineering - Creating new features (2p)
"""
import pandas as pd
import numpy as np
from textblob import TextBlob
import re
from typing import List



class FeatureEngineer:
    """Class for creating additional features"""
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Args:
            dataframe: DataFrame with text data
        """
        self.df = dataframe.copy()
        
    def add_text_length_features(self):
        """Text length features"""
        print("📏 Creating text length features...")
        
        self.df['text_length'] = self.df['clean_text_nlp'].str.len()
        self.df['word_count'] = self.df['clean_text_nlp'].str.split().str.len()
        self.df['avg_word_length'] = self.df['text_length'] / (self.df['word_count'] + 1)
        
        print(f"   ✅ text_length, word_count, avg_word_length")
        
    def add_sentiment_features(self):
        """Text sentiment features"""
        print("😊 Creating sentiment features...")
        
        # Simple sentiment calculation (may be slow on large data)
        def get_sentiment(text):
            try:
                return TextBlob(str(text)).sentiment.polarity
            except:
                return 0.0
        
        # For speed, take only first 1000 characters
        self.df['sentiment_polarity'] = self.df['clean_text_nlp'].str[:1000].apply(get_sentiment)
        
        print(f"   ✅ sentiment_polarity")
        
    def add_keyword_features(self):
        """Keyword presence features"""
        print("🔑 Creating keyword features...")
        
        # Economic keywords
        economy_keywords = ['economy', 'tax', 'trade', 'tariff', 'jobs', 'business', 
                           'market', 'stock', 'inflation', 'growth']
        
        # Political keywords
        politics_keywords = ['congress', 'election', 'vote', 'president', 'senate',
                            'democrat', 'republican', 'bill', 'law']
        
        def count_keywords(text, keywords):
            text_lower = str(text).lower()
            return sum(1 for kw in keywords if kw in text_lower)
        
        self.df['economy_keyword_count'] = self.df['clean_text_nlp'].apply(
            lambda x: count_keywords(x, economy_keywords)
        )
        
        self.df['politics_keyword_count'] = self.df['clean_text_nlp'].apply(
            lambda x: count_keywords(x, politics_keywords)
        )
        
        print(f"   ✅ economy_keyword_count, politics_keyword_count")
        
    def add_capitalization_features(self):
        """Capitalization features"""
        print("🔠 Creating capitalization features...")
        
        self.df['uppercase_ratio'] = self.df['clean_text_nlp'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1)
        )
        
        self.df['has_all_caps_word'] = self.df['clean_text_nlp'].apply(
            lambda x: int(any(word.isupper() and len(word) > 2 for word in str(x).split()))
        )
        
        print(f"   ✅ uppercase_ratio, has_all_caps_word")
        
    def add_punctuation_features(self):
        """Punctuation features"""
        print("❗ Creating punctuation features...")
        
        self.df['exclamation_count'] = self.df['clean_text_nlp'].str.count('!')
        self.df['question_count'] = self.df['clean_text_nlp'].str.count(r'\?')
        
        print(f"   ✅ exclamation_count, question_count")
        
    def add_category_features(self):
        """Category features (if available)"""
        if 'categories' in self.df.columns:
            print("🏷️  Creating category features...")
            
            # One-hot encoding for top categories
            all_categories = []
            for cats in self.df['categories']:
                if isinstance(cats, str) and cats.strip():
                    # Parse ['CAT1', 'CAT2'] format
                    cats_clean = cats.strip('[]').replace("'", "").split(', ')
                    all_categories.extend([c.strip() for c in cats_clean if c.strip()])
            
            # Top-5 categories
            from collections import Counter
            top_cats = [cat for cat, _ in Counter(all_categories).most_common(5)]
            
            for cat in top_cats:
                self.df[f'has_{cat}'] = self.df['categories'].apply(
                    lambda x: int(cat in str(x)) if pd.notna(x) else 0
                )
            
            print(f"   ✅ Created {len(top_cats)} category features")
    
    def add_temporal_features(self):
        """Temporal features"""
        if 'datetime' in self.df.columns:
            print("⏰ Creating temporal features...")
            
            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            self.df['hour'] = self.df['datetime'].dt.hour
            self.df['day_of_week'] = self.df['datetime'].dt.dayofweek
            self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
            
            print(f"   ✅ hour, day_of_week, is_weekend")
    
    def create_all_features(self):
        """Create all features"""
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING")
        print("=" * 60 + "\n")
        
        initial_cols = len(self.df.columns)
        
        self.add_text_length_features()
        # self.add_sentiment_features()  # May be slow
        self.add_keyword_features()
        self.add_capitalization_features()
        self.add_punctuation_features()
        self.add_category_features()
        self.add_temporal_features()
        
        new_cols = len(self.df.columns) - initial_cols
        
        print(f"\n✅ Created {new_cols} new features")
        print(f"📊 Total columns: {len(self.df.columns)}")
        
        return self.df
    
    def get_feature_names(self) -> List[str]:
        """Returns list of created features"""
        # Exclude original columns
        base_cols = ['datetime', 'tweet_text', 'clean_text_nlp', 'categories', 
                     'is_noise', 'Market_Impact', 'is_weekend_news', 'target']
        
        feature_cols = [col for col in self.df.columns if col not in base_cols]
        return feature_cols



# Usage example
if __name__ == "__main__":
    from data_loader import DataLoader
    
    loader = DataLoader("ready_for_ml_training.csv")
    loader.load_data()
    loader.clean_data()
    loader.create_target(threshold=0.005)
    
    engineer = FeatureEngineer(loader.get_full_dataframe())
    df_with_features = engineer.create_all_features()
    
    print("\n" + "=" * 60)
    print("NEW FEATURES:")
    print("=" * 60)
    for feat in engineer.get_feature_names():
        print(f"  • {feat}")
    
    print(f"\n📊 Sample data:")
    print(df_with_features[engineer.get_feature_names()].head())
