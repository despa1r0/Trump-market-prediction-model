import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yfinance as yf
import os
from datetime import timedelta

# === SETTINGS ===
OUTPUT_DIR = "combined_analytics_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_NAME = 'ready_for_ml_training.csv'
MIN_IMPACT_FILTER = 1.2   # For advanced matrix
TOPICS_THRESHOLD = 0.5    # For topics ranking (lower threshold)
MARKET_THRESHOLD = 1.5    # For market match visualization

def weighted_corr(x, y, w):
    """Calculates weighted correlation"""
    w_normalized = w / w.sum()
    mean_x = np.sum(w_normalized * x)
    mean_y = np.sum(w_normalized * y)
    
    cov = np.sum(w_normalized * (x - mean_x) * (y - mean_y))
    std_x = np.sqrt(np.sum(w_normalized * (x - mean_x)**2))
    std_y = np.sqrt(np.sum(w_normalized * (y - mean_y)**2))
    
    return cov / (std_x * std_y)

def fetch_market_data(start_date, end_date, ticker="^GSPC"):
    """Downloads market data for background context"""
    print(f"📥 Downloading market context for {ticker}...")
    
    # Expand window for better visualization
    start = (start_date - timedelta(days=5)).strftime('%Y-%m-%d')
    end = (end_date + timedelta(days=5)).strftime('%Y-%m-%d')
    
    data = yf.download(ticker, start=start, end=end, interval="1h", 
                       auto_adjust=False, multi_level_index=False, progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    data['Returns'] = data['Close'].pct_change() * 100
    
    # Convert to UTC
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC')
    else:
        data.index = data.index.tz_convert('UTC')
        
    print(f"   ✓ Market data loaded: {len(data)} hours")
    return data

def plot_market_match(market, ml_data, threshold):
    """Creates market match visualization"""
    print(f"\n🔹 Creating market match visualization...")
    
    plt.figure(figsize=(16, 8))
    
    # 1. Market price line (background)
    plt.plot(market.index, market['Close'], label='S&P 500 Price', 
             color='blue', alpha=0.5, linewidth=1.5)
    
    # 2. Market shocks (dots)
    shocks = market[market['Returns'].abs() >= threshold]
    drops = shocks[shocks['Returns'] <= -threshold]
    gains = shocks[shocks['Returns'] >= threshold]
    
    plt.scatter(drops.index, drops['Close'], color='red', s=30, 
                label=f'Market Drop > {threshold}%', zorder=3, alpha=0.6)
    plt.scatter(gains.index, gains['Close'], color='green', s=30, 
                label=f'Market Gain > {threshold}%', zorder=3, alpha=0.6)
    
    # 3. High-impact tweets (vertical lines)
    if 'Market_Impact' in ml_data.columns:
        significant_tweets = ml_data[ml_data['Market_Impact'].abs() >= threshold]
        
        if not significant_tweets.empty:
            unique_dates = significant_tweets['datetime'].unique()
            print(f"   Plotting {len(unique_dates)} high-impact tweet events...")
            
            for date in unique_dates:
                plt.axvline(x=date, color='purple', linestyle='--', 
                           alpha=0.5, linewidth=1.2)
            
            # Legend for tweets
            plt.axvline(x=unique_dates[0], color='purple', linestyle='--', 
                       alpha=0.5, label=f'High Impact Tweet (≥{threshold}%)')
    
    plt.title(f'Market Match: High-Impact Tweets vs Market Movements (Threshold {threshold}%)', 
              fontsize=16, pad=15)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/01_market_match.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: 01_market_match.png")

def plot_topics_ranking(df, threshold):
    """Creates topics ranking visualization with custom threshold"""
    print(f"\n🔹 Creating topics ranking (threshold: {threshold}%)...")
    
    # Filter data for topics ranking separately
    df_topics = df[df['Abs_Impact'] >= threshold].copy()
    print(f"   Using {len(df_topics)} tweets for topics (>={threshold}%)")
    
    df_exploded = df_topics.assign(
        categories=df_topics['categories'].str.split(',')
    ).explode('categories')
    df_exploded['categories'] = df_exploded['categories'].str.strip()
    
    topic_counts = df_exploded['categories'].value_counts()
    
    plt.figure(figsize=(12, 7))
    colors = plt.cm.magma(np.linspace(0.2, 0.9, len(topic_counts)))
    
    bars = plt.bar(range(len(topic_counts)), topic_counts.values, color=colors, 
                   edgecolor='black', linewidth=0.5)
    
    plt.xticks(range(len(topic_counts)), topic_counts.index, rotation=45, ha='right')
    plt.title(f'Topics Causing >{threshold}% Market Moves', fontsize=16, pad=15)
    plt.ylabel('Count of High-Impact Tweets', fontsize=12)
    plt.xlabel('Category', fontsize=12)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, topic_counts.values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/02_topics_ranking.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: 02_topics_ranking.png")

def plot_extended_matrix(df_filtered):
    """Creates extended correlation matrix"""
    print(f"\n🔹 Creating extended correlation matrix...")
    
    # Feature engineering
    df_filtered['is_POLITICS'] = df_filtered['categories'].str.contains(
        'DOMESTIC_POLITICS', na=False).astype(int)
    df_filtered['is_TRADE'] = df_filtered['categories'].str.contains(
        'TRADE_WAR', na=False).astype(int)
    df_filtered['is_CRYPTO'] = df_filtered['categories'].str.contains(
        'CRYPTO', na=False).astype(int)
    df_filtered['is_GEOPOLITICS'] = df_filtered['categories'].str.contains(
        'GEOPOLITICS', na=False).astype(int)
    df_filtered['is_ECONOMY'] = df_filtered['categories'].str.contains(
        'ECONOMY_MACRO', na=False).astype(int)
    
    # Interaction features
    df_filtered['is_news_politics'] = (
        (df_filtered['is_noise'] == 0) & (df_filtered['is_POLITICS'] == 1)
    ).astype(int)
    df_filtered['is_news_trade'] = (
        (df_filtered['is_noise'] == 0) & (df_filtered['is_TRADE'] == 1)
    ).astype(int)
    
    # Select columns for correlation
    cols_extended = [
        'is_noise', 'Market_Impact', 'Abs_Impact',
        'is_POLITICS', 'is_TRADE', 'is_CRYPTO', 
        'is_GEOPOLITICS', 'is_ECONOMY',
        'is_news_politics', 'is_news_trade'
    ]
    
    available_cols = [c for c in cols_extended 
                     if c in df_filtered.columns and df_filtered[c].nunique() > 1]
    
    plt.figure(figsize=(14, 12))
    corr_matrix = df_filtered[available_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f", 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1)
    
    plt.title(f'Extended Correlation Matrix (Impact >= {MIN_IMPACT_FILTER}%)', 
              fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/03_extended_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: 03_extended_matrix.png")
    
    return df_filtered

def print_summary_stats(df, df_filtered, market):
    """Prints summary statistics"""
    print("\n" + "="*70)
    print("📈 SUMMARY STATISTICS")
    print("="*70)
    
    print(f"\n📊 Dataset Overview:")
    print(f"   Total tweets in dataset:              {len(df)}")
    print(f"   High-impact tweets (≥{MIN_IMPACT_FILTER}%):   {len(df_filtered)}")
    print(f"   Noise ratio (high-impact):            {df_filtered['is_noise'].mean():.1%}")
    print(f"   Average |impact| (high-impact):       {df_filtered['Abs_Impact'].mean():.2f}%")
    
    print(f"\n📅 Time Range:")
    print(f"   First tweet:  {df['datetime'].min()}")
    print(f"   Last tweet:   {df['datetime'].max()}")
    print(f"   Duration:     {(df['datetime'].max() - df['datetime'].min()).days} days")
    
    print(f"\n🎯 Key Correlations:")
    normal_corr = df_filtered[['is_noise', 'Market_Impact']].corr().iloc[0, 1]
    print(f"   is_noise vs Market_Impact:         {normal_corr:.4f}")
    
    if 'is_POLITICS' in df_filtered.columns:
        pol_corr = df_filtered[['is_POLITICS', 'Market_Impact']].corr().iloc[0,1]
        print(f"   is_POLITICS vs Market_Impact:      {pol_corr:.4f}")
    
    if 'is_TRADE' in df_filtered.columns:
        trade_corr = df_filtered[['is_TRADE', 'Market_Impact']].corr().iloc[0,1]
        print(f"   is_TRADE vs Market_Impact:         {trade_corr:.4f}")
    
    print(f"\n📉 Market Context:")
    print(f"   Market data points:                {len(market)}")
    print(f"   Market drops >{MARKET_THRESHOLD}%:           {len(market[market['Returns'] <= -MARKET_THRESHOLD])}")
    print(f"   Market gains >{MARKET_THRESHOLD}%:           {len(market[market['Returns'] >= MARKET_THRESHOLD])}")
    
    print("\n" + "="*70)
    print(f"✅ All visualizations saved to folder: {OUTPUT_DIR}")
    print("="*70)

def run_combined_analytics():
    """Main function that runs all analytics"""
    print("="*70)
    print("🚀 COMBINED ANALYTICS PIPELINE")
    print("="*70)
    
    # Load data
    print(f"\n📂 Loading dataset: {FILE_NAME}...")
    try:
        df = pd.read_csv(FILE_NAME)
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)
        df = df.dropna(subset=['datetime'])
        df = df.sort_values('datetime')
        print(f"   ✓ Successfully loaded {len(df)} rows")
    except FileNotFoundError:
        print(f"   ❌ File not found: {FILE_NAME}")
        return
    except Exception as e:
        print(f"   ❌ Error loading file: {e}")
        return
    
    # Calculate absolute impact
    df['Abs_Impact'] = df['Market_Impact'].abs()
    
    # Filter for high-impact tweets
    print(f"\n🔍 Filtering tweets with impact >= {MIN_IMPACT_FILTER}%...")
    df_filtered = df[df['Abs_Impact'] >= MIN_IMPACT_FILTER].copy()
    df_filtered['is_noise'] = df_filtered['is_noise'].astype(int)
    print(f"   Original tweets:     {len(df)}")
    print(f"   High-impact tweets:  {len(df_filtered)}")
    
    if len(df_filtered) < 2:
        print("   ⚠️ Too few data points after filtering!")
        return
    
    # Fetch market data
    min_date = df['datetime'].min()
    max_date = df['datetime'].max()
    market = fetch_market_data(min_date, max_date)
    
    if market.empty:
        print("   ⚠️ Could not fetch market data!")
        return
    
    # Generate visualizations
    print("\n" + "="*70)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*70)
    
    # 1. Market Match
    plot_market_match(market, df, MARKET_THRESHOLD)
    
    # 2. Topics Ranking (uses separate threshold)
    plot_topics_ranking(df, TOPICS_THRESHOLD)
    
    # 3. Extended Matrix (uses MIN_IMPACT_FILTER)
    df_filtered = plot_extended_matrix(df_filtered)
    
    # Print summary
    print_summary_stats(df, df_filtered, market)

if __name__ == "__main__":
    run_combined_analytics()