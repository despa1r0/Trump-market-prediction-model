"""
Exploratory Data Analysis - Exploratory data analysis (2p)
Analiza eksploracyjna danych - Analiza wstępna (2p)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os # Path handling / Zarządzanie ścieżkami
from data_loader import DataLoader

class ExploratoryAnalysis:
    """Class for data analysis / Klasa do analizy danych"""

    def __init__(self, dataframe: pd.DataFrame, output_dir: str = "market_insights"):
        """
        Args:
            dataframe: DataFrame with data / Ramka danych
            output_dir: Directory for saving charts / Katalog do zapisu wykresów
        """
        self.df = dataframe
        self.output_dir = output_dir

    def basic_statistics(self):
        """Basic statistics of data / Podstawowe statystyki danych"""
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
        """Text data statistics / Statystyki danych tekstowych"""
        print("\n" + "=" * 60)
        print("TEXT STATISTICS")
        print("=" * 60)

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
        """Target variable distribution / Rozkład zmiennej docelowej"""
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
        """Correlation analysis / Analiza korelacji"""
        print("\n" + "=" * 60)
        print("CORRELATION ANALYSIS")
        print("=" * 60)

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            print("\n📈 Correlation matrix:")
            print(corr_matrix)

            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            plt.tight_layout()

            # Save to specific directory / Zapisz w określonym katalogu
            save_path = os.path.join(self.output_dir, 'correlation_matrix.png')
            plt.savefig(save_path)
            print(f"\n✅ Chart saved: {save_path}")
            plt.close()
        else:
            print("⚠️ Not enough numeric features for correlation")

    def market_impact_distribution(self):
        """Market_Impact distribution / Rozkład wpływu na rynek"""
        print("\n" + "=" * 60)
        print("MARKET IMPACT DISTRIBUTION")
        print("=" * 60)

        mi = self.df['Market_Impact']

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
        save_path = os.path.join(self.output_dir, 'market_impact_distribution.png')
        plt.savefig(save_path)
        print(f"\n✅ Chart saved: {save_path}")
        plt.close()

    def noise_analysis(self):
        """Noise vs real movements analysis / Analiza szumu i ruchów znaczących"""
        print("\n" + "=" * 60)
        print("NOISE ANALYSIS")
        print("=" * 60)

        if 'is_noise' in self.df.columns:
            noise_count = self.df['is_noise'].sum()
            total = len(self.df)
            print(f"\n🔇 Noise tweets:     {noise_count} ({noise_count/total*100:.1f}%)")
            print(f"📢 Significant tweets: {total - noise_count} ({(total-noise_count)/total*100:.1f}%)")

    def generate_full_report(self):
        """Generate full report / Generuj pełny raport"""
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