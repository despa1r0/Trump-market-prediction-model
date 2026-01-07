import pandas as pd
import yfinance as yf
from datetime import timedelta, datetime
import pytz
import re
import os
import sys

# Настройки отображения в консоли
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

class TrumpMarketPipeline:
    def __init__(self, raw_tweets_file, final_output_file):
        self.raw_tweets_file = raw_tweets_file
        self.final_output_file = final_output_file
        self.ticker = "SPY" # S&P 500 ETF
        
        # 🗓 НАСТРОЙКА ДАТЫ: Берем только 2025 год, как ты просил
        self.START_DATE = "2025-01-01"  
        
        # 🧠 БАЗА ЗНАНИЙ (Темы для категоризации)
        self.topic_config = {
            "ECONOMY_MACRO": [
                "fed", "federal reserve", "powell", "inflation", "cpi", "interest rate", 
                "rate hike", "recession", "gdp", "economy", "jobs", "unemployment", 
                "treasury", "yield", "debt", "central bank"
            ],
            "TRADE_WAR": [
                "tariff", "tax", "china", "trade", "deal", "mexico", "canada", "duty", 
                "export", "import", "deficit", "sanction", "currency", "eu", "europe"
            ],
            "CORPORATE": [
                "google", "apple", "facebook", "meta", "amazon", "boeing", "lockheed", 
                "ford", "gm", "general motors", "toyota", "tsmc", "chips", "tech", 
                "media", "disney", "cbs", "abc", "fake news"
            ],
            "CRYPTO": [
                "crypto", "bitcoin", "btc", "ethereum", "eth", "coinbase", "sec", 
                "gensler", "defi", "blockchain", "digital dollar", "cbdc"
            ],
            "GEOPOLITICS": [
                "war", "ukraine", "russia", "putin", "zelensky", "nato", "israel", 
                "gaza", "hamas", "iran", "north korea", "kim jong", "china", "taiwan"
            ],
            "DOMESTIC_POLITICS": [
                "border", "wall", "immigrant", "election", "biden", "democrat", 
                "republican", "senate", "house", "congress", "maga", "woke", "radical left"
            ]
        }

    # === ЭТАП 1: Загрузка и очистка твитов ===
    def load_and_clean_tweets(self):
        print("1. [Pipeline] Загрузка сырых твитов...")
        if not os.path.exists(self.raw_tweets_file):
             print(f"❌ Файл {self.raw_tweets_file} не найден!")
             return pd.DataFrame()

        try:
            # Читаем файл с разделителем '|'
            df = pd.read_csv(
                self.raw_tweets_file, 
                sep='|', 
                header=None, 
                names=['datetime', 'tweet_text'], 
                on_bad_lines='skip', 
                engine='python'
            )
        except Exception as e:
            print(f"❌ Ошибка чтения файла твитов: {e}")
            return pd.DataFrame()

        # Парсим даты в формат UTC
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', utc=True)
        
        # Удаляем пустые даты
        df = df.dropna(subset=['datetime'])

        # Фильтр по дате (Только 2025 год и новее)
        # Важно сделать это ДО тяжелой обработки текста
        df = df[df['datetime'] >= pd.Timestamp(self.START_DATE, tz='UTC')]

        if df.empty:
            print(f"⚠️ Внимание: Нет твитов после {self.START_DATE}. Проверь файл tweets_output.csv!")
            return pd.DataFrame()

        # Фильтрация ретвитов (Оставляем только RT самого себя)
        mask_clean = (~df['tweet_text'].str.startswith('RT @')) | \
                     (df['tweet_text'].str.startswith('RT @realDonaldTrump'))
        df = df[mask_clean].copy()

        # Чистка текста от мусора
        def clean_content(text):
            if not isinstance(text, str): return ""
            text = re.sub(r'http\S+', '', text) # Убрать ссылки
            text = re.sub(r'\s+', ' ', text).strip() # Убрать лишние пробелы
            return text

        df['tweet_text'] = df['tweet_text'].apply(clean_content)
        df = df.dropna(subset=['tweet_text'])
        
        print(f"   -> Твитов после очистки и фильтра даты: {len(df)}")
        return df

    # === ЭТАП 2: Рыночные данные и синхронизация ===
    def add_market_data(self, df_tweets):
        print(f"2. [Pipeline] Скачивание данных {self.ticker} с {self.START_DATE}...")
        
        try:
            # ⚠️ ВАЖНО: Используем интервал "1h" (1 час)
            # "30m" данные доступны только за последние 60 дней.
            # "1h" данные доступны за 730 дней (2 года). Это решает проблему ошибки.
            market_data = yf.download(
                self.ticker, 
                start=self.START_DATE, 
                interval="1h", 
                progress=False
            )
        except Exception as e:
            print(f"❌ Ошибка библиотеки yfinance: {e}")
            return pd.DataFrame()
        
        if market_data.empty:
            print("❌ ОШИБКА: Не удалось скачать данные SPY.")
            print("   Возможно, yfinance заблокирован или нет интернета.")
            return pd.DataFrame()

        # Приводим индексы к UTC
        market_data.index = market_data.index.tz_convert('UTC')
        
        # Расчет изменения цены (Return)
        # Исправление для новых версий yfinance (убираем мультииндекс, если есть)
        if isinstance(market_data.columns, pd.MultiIndex):
            close_price = market_data['Close'].iloc[:, 0] 
        else:
            close_price = market_data['Close']

        market_data['Return'] = close_price.pct_change() * 100
        
        # Создаем временную шкалу для быстрого поиска
        market_times = pd.Series(market_data.index, index=market_data.index).sort_index()

        def get_market_impact(tweet_time):
            # ТРЮК: Ищем ближайшую будущую свечу (bfill)
            try:
                idx = market_times.index.get_indexer([tweet_time], method='bfill')[0]
                if idx == -1: return None # Нет данных в будущем
                
                matched_time = market_times.index[idx]
                
                # Если рынок был закрыт более 4 дней (праздники), связь теряется
                if (matched_time - tweet_time) > timedelta(days=4): 
                    return None
                    
                return market_data.loc[matched_time, 'Return']
            except:
                return None

        print("   -> Синхронизация твитов с рынком (поиск реакции)...")
        df_tweets['Market_Impact'] = df_tweets['datetime'].apply(get_market_impact)
        
        # Создаем флаги
        df_tweets['is_weekend_news'] = df_tweets.apply(
            lambda x: 1 if (pd.notna(x['Market_Impact'])) else 0, axis=1
        )
        df_tweets['day_of_week'] = df_tweets['datetime'].dt.dayofweek
        df_tweets['is_weekend_real'] = df_tweets['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

        # Удаляем твиты, для которых не нашли рыночные данные
        df_final = df_tweets.dropna(subset=['Market_Impact']).copy()
        
        if df_final.empty:
            print("⚠️ ВНИМАНИЕ: 0 твитов совпало с рынком. Проверь даты в файлах!")
            return pd.DataFrame()

        return df_final

    # === ЭТАП 3: Категоризация и финализация ===
    def categorize_and_finalize(self, df):
        print("3. [Pipeline] Категоризация тем и подготовка к ML...")
        
        # Проверка целостности
        if 'Market_Impact' not in df.columns:
            print("❌ Ошибка: Нет колонки Market_Impact.")
            return pd.DataFrame()

        # Определение темы твита
        def get_categories(text):
            text_lower = text.lower()
            found_topics = []
            for topic, keywords in self.topic_config.items():
                if any(k in text_lower for k in keywords):
                    found_topics.append(topic)
            return ",".join(found_topics) if found_topics else "OTHER"

        df['categories'] = df['tweet_text'].apply(get_categories)
        df['is_noise'] = df['categories'].apply(lambda x: 1 if x == "OTHER" else 0)

        # Очистка текста для нейросети (только буквы и цифры)
        def clean_for_bert(text):
            text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
            return text.strip()

        df['clean_text_nlp'] = df['tweet_text'].apply(clean_for_bert)

        # Финальный отбор колонок
        cols = ['datetime', 'tweet_text', 'clean_text_nlp', 'categories', 'is_noise', 'Market_Impact', 'is_weekend_real']
        return df[cols]

    # === ГЛАВНЫЙ ЗАПУСК ===
    def run(self):
        # 1. Загрузка
        df = self.load_and_clean_tweets()
        if df.empty: return
        
        # 2. Рынок
        df = self.add_market_data(df)
        if df.empty: return

        # 3. Финализация
        df = self.categorize_and_finalize(df)
        if df.empty: return
        
        # 4. Сохранение
        print(f"4. [Pipeline] Сохранение результата в {self.final_output_file}")
        df.to_csv(self.final_output_file, index=False)
        print(f"✅ УСПЕХ! Обработано записей: {len(df)}")
       

if __name__ == "__main__":
    pipeline = TrumpMarketPipeline(
        raw_tweets_file="tweets_output.csv", 
        final_output_file="ready_for_ml_training.csv"
    )
    pipeline.run()