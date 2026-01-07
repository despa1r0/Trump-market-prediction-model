"""
Data Loader - Class for loading and initial data processing
"""
import pandas as pd
import numpy as np
from typing import Tuple
import os



class DataLoader:
    """Class for loading and preprocessing Twitter/Market data"""
    
    def __init__(self, filepath: str):
        """
        Args:
            filepath: path to CSV file
        """
        self.filepath = filepath
        self.df = None
        self.X = None
        self.y = None
        
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV"""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File {self.filepath} not found!")
        
        self.df = pd.read_csv(self.filepath)
        print(f"✅ Loaded {len(self.df)} records")
        return self.df
    
    def clean_data(self) -> pd.DataFrame:
        """Clean data from missing values"""
        if self.df is None:
            raise ValueError("Load data first using load_data()")
        
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=['clean_text_nlp', 'Market_Impact'])
        
        print(f"🧹 Removed {initial_len - len(self.df)} rows with missing values")
        return self.df
    
    def create_target(self, threshold: float = 0.005) -> pd.Series:
        """
        Create target variable based on Market_Impact
        
        Args:
            threshold: threshold for noise (default: 0.005 = 0.5%)
        
        Returns:
            pd.Series with labels: -1 (Drop), 0 (Noise), 1 (Rise)
        """
        def classify_impact(x):
            if abs(x) < threshold:
                return 0  # Noise
            return 1 if x > 0 else -1
        
        self.df['target'] = self.df['Market_Impact'].apply(classify_impact)
        
        # Class statistics
        class_dist = self.df['target'].value_counts().sort_index()
        print(f"\n📊 Class Distribution:")
        print(f"  Drop (-1): {class_dist.get(-1, 0)} ({class_dist.get(-1, 0)/len(self.df)*100:.1f}%)")
        print(f"  Noise (0):  {class_dist.get(0, 0)} ({class_dist.get(0, 0)/len(self.df)*100:.1f}%)")
        print(f"  Rise (1):   {class_dist.get(1, 0)} ({class_dist.get(1, 0)/len(self.df)*100:.1f}%)")
        
        return self.df['target']
    
    def prepare_features_and_target(self) -> Tuple[pd.Series, pd.Series]:
        """
        Prepare features (X) and target variable (y)
        
        Returns:
            X: text features
            y: target variable
        """
        if self.df is None or 'target' not in self.df.columns:
            raise ValueError("Load and process data first")
        
        self.X = self.df['clean_text_nlp'].astype(str)
        self.y = self.df['target']
        
        print(f"\n✅ Prepared:")
        print(f"   X (texts): {len(self.X)} records")
        print(f"   y (labels): {len(self.y)} records")
        
        return self.X, self.y
    
    def get_full_dataframe(self) -> pd.DataFrame:
        """Returns full DataFrame with additional columns"""
        return self.df



# Usage example
if __name__ == "__main__":
    loader = DataLoader("ready_for_ml_training.csv")
    loader.load_data()
    loader.clean_data()
    loader.create_target(threshold=0.005)
    X, y = loader.prepare_features_and_target()
    
    print("\n" + "="*50)
    print("Sample data:")
    print(X.head(3))
