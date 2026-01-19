import sys
import os
import pandas as pd
import yfinance as yf
from datetime import timedelta, datetime
import re

# Append current directory to path / Dodaj bieżący katalog do ścieżki
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Display settings / Ustawienia wyświetlania
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


class TrumpMarketPipeline:
    """Pipeline for syncing tweets with market data / Rurociąg do synchronizacji tweetów z danymi rynkowymi."""

    def __init__(self, raw_tweets_file, final_output_file):
        self.raw_tweets_file = raw_tweets_file
        self.final_output_file = final_output_file
        self.ticker = "SPY"  # S&P 500 ETF
        self.START_DATE = "2025-01-01"  # Wider range for more data / Szerszy zakres dla większej ilości danych

        # 1. NOISE TRIGGER WORDS (Priority #1) / SŁOWA KLUCZOWE SZUMU (Priorytet nr 1)
        # If these words are present, the tweet is immediately considered garbage / Jeśli te słowa występują, tweet jest od razu uznawany za śmieciowy
        self.NOISE_TRIGGERS = [
            "happy birthday", "congratulations", "endorse", "endorsement",
            "great honor", "thank you", "rally", "crowd", "tune in",
            "vote for", "poll", "leading big", "make america great again",
            "enjoy", "merry christmas", "happy new year", "great job",
            "total support", "never let you down"
        ]

        # 2. CLEAN TOPICS (Priority #2) / CZYSTE TEMATY (Priorytet nr 2)
        # Excluded "fake news", "maga", etc., as they created false signals / Wykluczono "fake news", "maga" itp., ponieważ tworzyły fałszywe sygnały
        self.topic_config = {
            "ECONOMY": [
                "inflation", "cpi", "economy", "jobs", "tax", "tariff", "trade",
                "price", "cost", "market", "stock", "federal reserve", "fed", "cut"
            ],
            "GEOPOLITICS": [
                "war", "china", "russia", "ukraine", "israel", "iran", "border",
                "military", "peace", "conflict", "nato"
            ],
            "CRYPTO": [
                "crypto", "bitcoin", "btc", "blockchain", "digital"
            ],
            "DOMESTIC_POLICY": [
                "law", "order", "supreme court", "constitution", "bill", "legislation",
                "senate", "house", "department"
            ]
        }

    def load_and_clean_tweets(self):
        """Loads raw tweets and applies initial filtering / Ładuje surowe tweety i stosuje wstępne filtrowanie."""
        print("1. [Pipeline] Loading and filtering tweets...")
        if not os.path.exists(self.raw_tweets_file):
            print(f"❌ File {self.raw_tweets_file} not found!")
            return pd.DataFrame()

        try:
            df = pd.read_csv(self.raw_tweets_file, sep='|', header=None,
                             names=['datetime', 'tweet_text'], on_bad_lines='skip', engine='python')
        except Exception as e:
            print(f"❌ Reading error: {e}")
            return pd.DataFrame()

        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)
        df = df.dropna(subset=['datetime'])
        df = df[df['datetime'] >= pd.Timestamp(self.START_DATE, tz='UTC')]

        # --- RETWEET FILTERING / FILTROWANIE RETWEETÓW ---
        # Keep original tweets OR self-retweets / Zachowaj oryginalne tweety LUB własne retweety
        mask_clean = (~df['tweet_text'].str.startswith('RT @')) | \
                     (df['tweet_text'].str.startswith('RT @realDonaldTrump'))
        df = df[mask_clean].copy()

        # Clean text from links / Oczyść tekst z linków
        df['tweet_text'] = df['tweet_text'].astype(str).apply(lambda x: re.sub(r'http\S+', '', x).strip())
        df = df[df['tweet_text'].str.len() > 10]  # Remove very short tweets / Usuń bardzo krótkie tweety

        print(f"   -> Tweets after filtering: {len(df)}")
        return df

    def add_market_data(self, df_tweets):
        """Syncs tweets with ticker price movements / Synchronizuje tweety z ruchami cen tickerów."""
        print(f"2. [Pipeline] Downloading market data ({self.ticker})...")
        try:
            # interval="1h" allows seeing market opening / interwał 1h pozwala zobaczyć otwarcie rynku
            market_data = yf.download(self.ticker, start=self.START_DATE, interval="1h", progress=False)
        except Exception as e:
            print(f"❌ yfinance error: {e}")
            return pd.DataFrame()

        if market_data.empty:
            print("❌ No market data found!")
            return pd.DataFrame()

        market_data.index = market_data.index.tz_convert('UTC')

        # Use Close price for reaction / Użyj ceny zamknięcia dla reakcji
        if isinstance(market_data.columns, pd.MultiIndex):
            price = market_data['Close'].iloc[:, 0]
        else:
            price = market_data['Close']

        market_data['Return'] = price.pct_change() * 100
        market_times = pd.Series(market_data.index, index=market_data.index).sort_index()

        def get_market_impact(tweet_time):
            # Find the nearest future point / Znajdź najbliższy punkt w przyszłości
            try:
                idx = market_times.index.get_indexer([tweet_time], method='bfill')[0]
                if idx == -1: return None

                matched_time = market_times.index[idx]

                # Ignore if gap is > 4 days (holidays) / Ignoruj, jeśli przerwa wynosi > 4 dni (święta)
                if (matched_time - tweet_time) > timedelta(days=4):
                    return None

                return market_data.loc[matched_time, 'Return']
            except:
                return None

        print("   -> Synchronization successful (Handles weekends)...")
        df_tweets['Market_Impact'] = df_tweets['datetime'].apply(get_market_impact)

        return df_tweets.dropna(subset=['Market_Impact']).copy()

    def categorize_and_finalize(self, df):
        """Marks tweets as NOISE or SIGNAL / Oznacza tweety jako SZUM lub SYGNAŁ."""
        print("3. [Pipeline] Categorizing NOISE vs SIGNAL...")

        def analyze_text(text):
            text_lower = text.lower()

            # STEP 1: Check for NOISE (Priority!) / KROK 1: Sprawdź pod kątem SZUMU (Priorytet!)
            if any(trigger in text_lower for trigger in self.NOISE_TRIGGERS):
                return "OTHER", 1  # Category OTHER, is_noise = 1

            # STEP 2: Search for real topics / KROK 2: Szukaj prawdziwych tematów
            found_topics = []
            for topic, keywords in self.topic_config.items():
                if any(k in text_lower for k in keywords):
                    found_topics.append(topic)

            if found_topics:
                return ",".join(found_topics), 0  # Has topic, is_noise = 0
            else:
                return "OTHER", 1  # No topic -> Noise

        # Apply analysis and unpack results / Zastosuj analizę i rozpakuj wyniki
        df[['categories', 'is_noise']] = df['tweet_text'].apply(
            lambda x: pd.Series(analyze_text(x))
        )

        # Prepare text for NLP processing / Przygotuj tekst do przetwarzania NLP
        df['clean_text_nlp'] = df['tweet_text'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s.,!?]', '', x).strip())

        cols = ['datetime', 'tweet_text', 'clean_text_nlp', 'categories', 'is_noise', 'Market_Impact']
        return df[cols]

    def run(self):
        """Execute the full pipeline / Uruchom pełny rurociąg."""
        df = self.load_and_clean_tweets()
        if df.empty: return
        df = self.add_market_data(df)
        if df.empty: return
        df = self.categorize_and_finalize(df)

        df.to_csv(self.final_output_file, index=False)

        # Output summary statistics / Wyświetl statystyki podsumowujące
        noise_count = df['is_noise'].sum()
        total = len(df)
        print(f"\n✅ DONE! Saved to {self.final_output_file}")
        print(f"📊 Noise:  {noise_count} ({noise_count / total * 100:.1f}%)")
        print(f"📊 Signal: {total - noise_count} ({(total - noise_count) / total * 100:.1f}%)")


if __name__ == "__main__":
    # Manual path configuration / Ręczna konfiguracja ścieżki
    current_dir = r"C:\My folder\whole Project\full_arch_v1\DataPrep and Graphs"

    # Assemble full paths / Skonstruuj pełne ścieżki
    input_path = os.path.join(current_dir, "tweets_output.csv")
    output_path = os.path.join(current_dir, "ready_for_ml_training.csv")

    print(f"📂 Working directory: {current_dir}")
    print(f"📂 Searching for input file: {input_path}")

    # Validation before execution / Walidacja przed wykonaniem
    if not os.path.exists(input_path):
        print("❌ ERROR: tweets_output.csv still not found at this path!")
        print("   Ensure the file is located in this specific folder.")
    else:
        # Launch pipeline / Uruchom rurociąg
        pipeline = TrumpMarketPipeline(input_path, output_path)
        pipeline.run()