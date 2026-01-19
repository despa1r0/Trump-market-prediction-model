import sys
import os
import re
import joblib
import pandas as pd
import numpy as np
import datetime
#win garbage
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# models list
MODEL_FILES = [
    'best_model.pkl',
    'model_LogReg_Combined.pkl',
    'model_RandomForest_Combined.pkl',
    'model_LogReg_TextOnly.pkl'
]

# wordslist
TEST_PHRASES = [
    # dr
    "HUGE TARIFFS ON CHINA AND MEXICO! TRADE WAR!",
    "I WILL BAN ALL IMPORTS. ECONOMY IS BAD AND DANGEROUS!",
    "INFLATION IS KILLING THE USA. WORST CRISIS EVER!",
    
    # rs
    "STOCK MARKET HIT ALL TIME HIGH! JOBS ARE BOOMING!",
    "GREATEST ECONOMY IN HISTORY! AMERICA WINS AGAIN!",
    "SIGNED A HUGE TRADE DEAL! TRILLIONS OF DOLLARS!",
    
    # ns
    "I am going to play golf in Florida today. Beautiful weather.",
    "Happy Birthday to the First Lady! Have a great day!",
    "ban drop and risk", 
    "covfefe"
]

def extract_features(text):
    text_str = str(text)
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text_str.lower())
    words = clean_text.split()
    now = datetime.datetime.now()
    
    features = {
        'datetime': now,
        'tweet_text': text_str,
        'clean_text_nlp': clean_text,
        'text_length': len(clean_text),
        'word_count': len(words),
        'avg_word_length': len(clean_text) / (len(words) + 1) if len(words) > 0 else 0,
        'uppercase_ratio': sum(1 for c in text_str if c.isupper()) / len(text_str) if len(text_str) > 0 else 0,
        'exclamation_count': text_str.count('!'),
        'question_count': text_str.count('?'),
        'economy_keyword_count': sum(1 for w in words if w in ['tariff', 'tariffs', 'trade', 'economy', 'market', 'markets', 'tax', 'taxes', 'china', 'money', 'jobs', 'inflation']),
        'politics_keyword_count': sum(1 for w in words if w in ['democrats', 'republicans', 'election', 'court', 'border', 'congress', 'senate', 'biden']),
        'has_all_caps_word': 1 if any(w.isupper() and len(w) > 2 for w in text_str.split()) else 0,
        'hour': now.hour, 'day_of_week': now.weekday(), 'is_weekend': 1 if now.weekday() >= 5 else 0,
        'categories': 'OTHER', 
        'has_OTHER': 0, 'has_GEOPOLITICS': 0, 'has_ECONOMY': 0, 'has_POLITICS': 0, 
        'has_PERSONNEL': 0, 'has_DOMESTIC_POLICY': 0, 'has_GEOPOLITICS,DOMESTIC_POLICY': 0,
        'is_noise': 0, 'Market_Impact': 0
    }
    
    all_categories = ['OTHER', 'GEOPOLITICS', 'ECONOMY', 'POLITICS', 'PERSONNEL', 'DOMESTIC_POLICY', 'GEOPOLITICS,DOMESTIC_POLICY']
    for cat in all_categories:
        col_name = f'has_{cat}'
        if cat.lower().replace('_', ' ') in text_str.lower():
             features[col_name] = 1

    return pd.DataFrame([features])

# diastart
print("--- ЗАПУСК ДИАГНОСТИКИ ---")

loaded_models = {}

# loadfrst
for fname in MODEL_FILES:
    if os.path.exists(fname):
        try:
            data = joblib.load(fname)
            if isinstance(data, dict) and 'model' in data:
                model = data['model']
            else:
                model = data
            loaded_models[fname] = model
            print(f"✅ Модель загружена: {fname}")
        except Exception as e:
            print(f"❌ Ошибка загрузки {fname}: {e}")
    else:
        print(f"⚠️ Файл не найден: {fname}")

print("\n--- СРАВНЕНИЕ ПРЕДСКАЗАНИЙ ---")

# scnd
for text in TEST_PHRASES:
    print(f"\n📝 Текст: '{text}'")
    features = extract_features(text)
    
    for m_name, model in loaded_models.items():
        try:
            try:
                probs = model.predict_proba(features)[0]
                pred = model.predict(features)[0]
            except:
                probs = model.predict_proba([text])[0]
                pred = model.predict([text])[0]
            
            classes = getattr(model, 'classes_', [-1, 0, 1])
            
            
            res_str = f"UNKNOWN ({pred})"
            if pred == 1: res_str = "🟢 RISE"
            elif pred == -1: res_str = "🔴 DROP"
            elif pred == 0: res_str = "⚪️ NEUTRAL"
            
            confidence = max(probs) * 100
            
            print(f"   🔹 {m_name}: {res_str} (Conf: {confidence:.1f}%) | Probs: {probs}")
            
        except Exception as e:
            print(f"   🔸 {m_name}: Ошибка ({e})")

print("\n--- КОНЕЦ ---")