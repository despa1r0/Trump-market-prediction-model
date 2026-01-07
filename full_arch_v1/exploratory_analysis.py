"""
Exploratory Data Analysis - Exploratory data analysis (2p)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import DataLoader



class ExploratoryAnalysis:
    """Class for data analysis"""
    
    def __init__(self, dataframe: pd.DataFrame):
        """
        Args:
            dataframe: DataFrame with data
        """
        self.df = dataframe
        
    def basic_statistics(self):
        """Basic statistics of data"""
        print("=" * 60)
        print("BASIC STATISTICS")
        print("=" * 60)
        
        print(f"\n📏 Dataset size: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        print("\n📊 Market_Impact statistics:")
        print(self.df['Market_Impact'].describe())
        
        print("\n📝 Data types:")
        print(self.df.dtypes)
        
        print("\n❓ Missing values:")
        print(self.df.isnull().sum())
        
    def text_statistics(self):
        """Text data statistics"""
        print("\n" + "=" * 60)
        print("TEXT STATISTICS")
        print("=" * 60)
        
        # Text lengths
        text_lengths = self.df['clean_text_nlp'].str.len()
        word_counts = self.df['clean_text_nlp'].str.split().str.len()
        
        print(f"\n📏 Text length (characters):")
        print(f"   Average: {text_lengths.mean():.0f}")
        print(f"   Median: {text_lengths.median():.0f}")
        print(f"   Min/Max: {text_lengths.min()}/{text_lengths.max()}")
        
        print(f"\n📝 Word count:")
        print(f"   Average: {word_counts.mean():.1f}")
        print(f"   Median: {word_counts.median():.0f}")
        print(f"   Min/Max: {word_counts.min()}/{word_counts.max()}")
        
    def target_distribution(self):
        """Target variable distribution"""
        print("\n" + "=" * 60)
        print("CLASS DISTRIBUTION")
        print("=" * 60)
        
        if 'target' in self.df.columns:
            dist = self.df['target'].value_counts().sort_index()
            total = len(self.df)
            
            for label, count in dist.items():
                pct = count / total * 100
                label_name = {-1: "Drop", 0: "Noise", 1: "Rise"}.get(label, label)
                print(f"   {label_name:12s} ({label:2d}): {count:4d} ({pct:5.1f}%)")
        
    def correlation_analysis(self):
        """Correlation analysis"""
        print("\n" + "=" * 60)
        print("CORRELATION ANALYSIS")
        print("=" * 60)
        
        # Numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            print("\n📈 Correlation matrix:")
            print(corr_matrix)
            
            # Save visualization
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.savefig('correlation_matrix.png')
            print("\n✅ Chart saved: correlation_matrix.png")
            plt.close()
        else:
            print("⚠️ Not enough numeric features for correlation")
            
    def market_impact_distribution(self):
        """Market_Impact distribution"""
        print("\n" + "=" * 60)
        print("MARKET IMPACT DISTRIBUTION")
        print("=" * 60)
        
        mi = self.df['Market_Impact']
        
        print(f"\n📊 Statistics:")
        print(f"   Mean:       {mi.mean():.6f}")
        print(f"   Median:     {mi.median():.6f}")
        print(f"   Std:        {mi.std():.6f}")
        print(f"   Min:        {mi.min():.6f}")
        print(f"   Max:        {mi.max():.6f}")
        
        # Histogram
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(mi, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Market Impact')
        plt.ylabel('Frequency')
        plt.title('Distribution of Market Impact')
        plt.axvline(0, color='red', linestyle='--', linewidth=2)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(mi, vert=True)
        plt.ylabel('Market Impact')
        plt.title('Box Plot of Market Impact')
        
        plt.tight_layout()
        plt.savefig('market_impact_distribution.png')
        print("\n✅ Chart saved: market_impact_distribution.png")
        plt.close()
        
    def noise_analysis(self):
        """Noise vs real movements analysis"""
        print("\n" + "=" * 60)
        print("NOISE ANALYSIS")
        print("=" * 60)
        
        if 'is_noise' in self.df.columns:
            noise_count = self.df['is_noise'].sum()
            total = len(self.df)
            print(f"\n🔇 Noise tweets:     {noise_count} ({noise_count/total*100:.1f}%)")
            print(f"📢 Significant tweets: {total - noise_count} ({(total-noise_count)/total*100:.1f}%)")
            
    def generate_full_report(self):
        """Generate full report"""
        print("\n" + "🔍 " * 20)
        print("FULL DATA ANALYSIS")
        print("🔍 " * 20 + "\n")
        
        self.basic_statistics()
        self.text_statistics()
        self.target_distribution()
        self.market_impact_distribution()
        self.noise_analysis()
        self.correlation_analysis()
        
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETED")
        print("=" * 60)



# Usage example
if __name__ == "__main__":
    loader = DataLoader("ready_for_ml_training.csv")
    loader.load_data()
    loader.clean_data()
    loader.create_target(threshold=0.005)
    
    analyzer = ExploratoryAnalysis(loader.get_full_dataframe())
    analyzer.generate_full_report()
