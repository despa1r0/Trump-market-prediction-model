import sys
import os
import re
import joblib
import pandas as pd
import numpy as np
import datetime
from flask import Flask, render_template, request, jsonify

# fix coding on windows bo blendy wyskakuja
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = Flask(__name__)

# tu trzymamy zaladowane modele
loaded_models = {}
# lista plikow
model_files = {
    'best_model.pkl': 'Best Model',
    'model_LogReg_Combined.pkl': 'LogReg Combined',
    'model_LogReg_TextOnly.pkl': 'LogReg TextOnly',
    'model_RandomForest_Combined.pkl': 'RandomForest Combined'
}

# ladowanie modeli przy starcie
print("loading models...")
for f, name in model_files.items():
    if os.path.exists(f):
        try:
            tmp = joblib.load(f)
            # check if dict or model directly
            if isinstance(tmp, dict) and 'model' in tmp:
                loaded_models[f] = tmp['model']
                print(f"ok: {name} (dict)")
            else:
                loaded_models[f] = tmp
                print(f"ok: {name} (raw)")
        except Exception as e:
            print(f"error loading {f}: {e}")
    else:
        print(f"nie ma pliku: {f}")

# funkcja do wyciagania features z tekstu
def get_features(txt):
    s = str(txt)
    # czyszczenie smieci
    clean = re.sub(r'[^a-zA-Z\s]', '', s.lower())
    arr = clean.split()
    now = datetime.datetime.now()
    
    # feature engineering wtf
    res = {
        'datetime': now,
        'tweet_text': s,
        'clean_text_nlp': clean,
        'text_length': len(clean),
        'word_count': len(arr),
        'avg_word_length': len(clean) / (len(arr) + 1) if len(arr) > 0 else 0,
        'uppercase_ratio': sum(1 for c in s if c.isupper()) / len(s) if len(s) > 0 else 0,
        'exclamation_count': s.count('!'),
        'question_count': s.count('?'),
        # slowa kluczowe check
        'economy_keyword_count': sum(1 for w in arr if w in ['tariff', 'tariffs', 'trade', 'economy', 'market', 'markets', 'tax', 'taxes', 'china', 'money', 'jobs', 'inflation']),
        'politics_keyword_count': sum(1 for w in arr if w in ['democrats', 'republicans', 'election', 'court', 'border', 'congress', 'senate', 'biden']),
        'has_all_caps_word': 1 if any(w.isupper() and len(w) > 2 for w in s.split()) else 0,
        'hour': now.hour, 
        'day_of_week': now.weekday(), 
        'is_weekend': 1 if now.weekday() >= 5 else 0,
        # categories placeholders
        'categories': 'OTHER', 
        'has_OTHER': 0, 'has_GEOPOLITICS': 0, 'has_ECONOMY': 0, 
        'has_POLITICS': 0, 'has_PERSONNEL': 0, 'has_DOMESTIC_POLICY': 0, 
        'has_GEOPOLITICS,DOMESTIC_POLICY': 0,
        'is_noise': 0, 'Market_Impact': 0 
    }
    
    # proste sprawdzanie kategorii
    cats = ['OTHER', 'GEOPOLITICS', 'ECONOMY', 'POLITICS', 'PERSONNEL', 'DOMESTIC_POLICY', 'GEOPOLITICS,DOMESTIC_POLICY']
    for c in cats:
        if c.lower().replace('_', ' ') in s.lower():
            res[f'has_{c}'] = 1

    return pd.DataFrame([res])

@app.route('/')
def main_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        dt = request.json
        tweet = dt.get('tweet', '')
        selected_model = dt.get('model_name', 'best_model.pkl')
        
        if not tweet or len(tweet.strip()) == 0:
            return jsonify({'prediction': 'DAJ TEKST', 'confidence': '0%'})

        model = loaded_models.get(selected_model)
        if model is None:
             return jsonify({'prediction': 'MODEL ERROR', 'confidence': '???'})

        # data prep
        df = get_features(tweet)
        
        try:
            probs = model.predict_proba(df)[0]
            cls = model.predict(df)[0]
        except:
            # fallback jakby dataframe nie dzialal
            try:
                probs = model.predict_proba([tweet])[0]
                cls = model.predict([tweet])[0]
            except Exception as e:
                return jsonify({'prediction': 'ERROR', 'confidence': 'Logi'})

        # logika indexow
        if len(probs) == 3:
            p_drop = probs[0]
            p_rise = probs[2]
        elif len(probs) == 2:
            p_drop = probs[0]
            p_rise = probs[1]
        else:
            p_drop = 0
            p_rise = 0

        # mapping wynikow
        res_map = {
            -1: 'MARKET DROP (BEARISH)',
             0: 'NO IMPACT (UNCERTAIN)',
             1: 'MARKET RISE (BULLISH)'
        }
        final_text = res_map.get(cls, 'UNKNOWN')
        
        # fix: threshold 8% bo model jest zbyt pewny siebie czasem
        # jak roznica mniejsza niz 0.08 to niepewne
        diff = abs(p_rise - p_drop)
        conf = max(p_drop, p_rise)
        
        suffix = ""
        if diff < 0.08:
            final_text = 'NO IMPACT (UNCERTAIN)'
            suffix = " (Too Close)"

        conf_str = f"{conf*100:.1f}%{suffix}"

        return jsonify({
            'prediction': final_text,
            'confidence': conf_str
        })
        
    except Exception as x:
        print(f"cos jeblo: {x}")
        return jsonify({'prediction': 'CRASH', 'confidence': str(x)})

if __name__ == '__main__':
    print("serwer odpalony...")
    app.run(host='127.0.0.1', port=5000, debug=True)