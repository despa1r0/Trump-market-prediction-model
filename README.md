# Trump Tweet Market Predictor
**Analiza wpływu tweetów Donalda Trumpa na rynek S&P 500**

## 1. Wprowadzenie
Celem projektu **Trump Tweet Market Predictor** jest zbadanie i modelowanie zależności pomiędzy treścią tweetów Donalda Trumpa a krótkoterminowymi reakcjami rynku finansowego, reprezentowanego przez indeks **S&P 500 (ETF SPY)**.

System automatycznie:
- pobiera i czyści tweety,
- synchronizuje je z danymi rynkowymi,
- klasyfikuje ich treść tematycznie,
- mierzy rzeczywisty wpływ na rynek,
- buduje modele uczenia maszynowego przewidujące kierunek ruchu rynku:
  - **-1**: spadek
  - **0**: szum
  - **1**: wzrost

Całość została zaprojektowana jako modularny ekosystem składający się z 8 współpracujących komponentów.

---

## 2. Architektura systemu
Projekt jest podzielony na dwa główne etapy:

**ETAP 0 – Budowa zestawu danych (`DataPrep and Graphs`)**
Zanim możliwe jest trenowanie modeli ML, surowe tweety muszą zostać przekształcone w uporządkowany zbiór danych.

**ETAP 1 – Uczenie maszynowe i analiza predykcyjna**
Gotowy zestaw danych trafia do pipeline’u ML, gdzie powstają cechy, modele i raporty ewaluacyjne.

---

## 3. ETAP 0 – Przygotowanie danych

### 3.1. `tweets_output.csv` – Dane źródłowe
Plik wejściowy zawiera surowe tweety Donalda Trumpa.
- **Separator**: `|`
- **Kodowanie**: `UTF-8`
- **Kolumny**: `datetime`, `tweet_text`

Zawartość obejmuje oryginalne tweety, retweety, linki, znaki specjalne oraz treści polityczne, gospodarcze i prywatne.

### 3.2. `data_pipeline.py` – Pipeline przetwarzania danych
Klasa **TrumpMarketPipeline** odpowiada za pełne przygotowanie danych do uczenia maszynowego.
- **Ticker rynku**: `SPY`
- **Data początkowa**: `2025-01-01`
- **Źródło danych rynkowych**: `yfinance`
- **Interwał**: `1 godzina`

### 3.3. Logika klasyfikacji tweetów

**Poziom 1 – Filtr szumu (`NOISE_TRIGGERS`)**
Jeżeli tweet zawiera frazy typu: `happy birthday`, `congratulations`, `thank you`, `rally`, `vote for`, `merry christmas` — jest automatycznie oznaczany jako szum (**Noise**).

**Poziom 2 – Klasyfikacja tematyczna**
Tweet otrzymuje jedną lub kilka kategorii, jeśli zawiera słowa kluczowe:

| Kategoria | Przykładowe słowa kluczowe |
| :--- | :--- |
| **ECONOMY** | inflation, economy, tax, tariff, fed, market |
| **GEOPOLITICS** | war, china, russia, ukraine, nato |
| **CRYPTO** | bitcoin, crypto, blockchain |
| **DOMESTIC_POLICY** | law, senate, house, bill, supreme court |

Jeśli nie pasuje do żadnej kategorii → **OTHER** + szum.

### 3.4. Synchronizacja z rynkiem
Dla każdego tweeta:
1. wyszukiwany jest najbliższy przyszły punkt notowań,
2. obliczana jest procentowa zmiana ceny,
3. jeśli przerwa przekracza 4 dni (weekendy, święta) → tweet jest ignorowany.

### 3.5. Finalny zestaw danych
Wynikiem ETAPU 0 jest plik `ready_for_ml_training.csv`.

---

## 4. Wizualizacja danych – `graph.py`
Moduł generuje trzy główne wykresy:

1. **Market Match**
   - Pokazuje linię ceny S&P 500, punkty dużych wzrostów i spadków oraz momenty tweetów.
2. **Topics Ranking**
   - Ranking kategorii tweetów o największym wpływie rynkowym.
3. **Extended Correlation Matrix**
   - Mapa korelacji pomiędzy kategoriami tweetów, cechami szumu i faktycznymi ruchami rynku.

---

## 5. ETAP 1 – Pipeline uczenia maszynowego

### 5.1. `data_loader`
- ładuje dane,
- czyści braki,
- tworzy zmienną docelową (Target): `[-1, 0, 1]`.

### 5.2. `feature_engineer`
Tworzy ~15 dodatkowych cech:
- **tekstowe**: długość tekstu, liczba słów, sentyment (TextBlob)
- **słowa kluczowe**: zliczanie wystąpień
- **emocjonalność**: WIELKIE LITERY, wykrzykniki (!!!)
- **czasowe**: godzina, dzień tygodnia
- **kategorie**: binarne flagi tematyczne

### 5.3. `embeddings.py` (NOWE)
Moduł odpowiedzialny za wektoryzację tekstu.
- Ładuje pretrenowane wektory **GloVe** (Global Vectors for Word Representation).
- Obsługuje brakujące słowa i tworzy gęste wektory (50 wymiarów) dla każdego tweeta.
- Wymagany zarówno podczas treningu (`TrumpMarketPredict`), jak i na serwerze (`TrumpPredictionAI`).

### 5.4. `model_trainer`
Trenuje modele w dwóch wariantach (TF-IDF oraz GloVe). Modele:
1. **Logistic Regression**
2. **Random Forest**
3. **Gradient Boosting**

Każdy model działa w dwóch wariantach:
- **TF-IDF Vectorizer**: klasyczne podejście statystyczne (liczenie słów)
- **GloVe Embeddings**: wektorowa reprezentacja semantyczna (zrozumienie kontekstu)

Modele są oceniane przy pomocy: `accuracy`, `precision`, `recall`, `F1-score`.
Najlepszy model trafia do **GridSearchCV** w celu dalszej optymalizacji.

### 5.5. `model_evaluator`
Generuje raporty klasyfikacji, macierze pomyłek i analizuje błędy predykcji.

---

## 6. Główne skrypty sterujące projektem

### `main.py` — Główny punkt wejścia (pełny pipeline)
Uruchamia cały proces end-to-end – od surowych danych aż po porównanie modeli.
1. **Ładowanie danych** (DataLoader)
2. **Inżynieria cech** (FeatureEngineer)
3. **Analiza eksploracyjna** (opcjonalna)
4. **Trenowanie modeli** (ModelTrainer)
5. **Dostrajanie najlepszego modelu** (GridSearchCV)
6. **Szczegółowa ewaluacja** (raporty, macierze)
7. **Porównanie modeli** (wykresy porównawcze)

### `save_model.py` — Tryb interaktywny i zapis modeli
Udostępnia interaktywny interfejs tekstowy.
- **Train & Optimize**: to samo co `main.py`, ale z interaktywnym podglądem.
- **Save Model**: wylistowanie modeli, podświetlenie najlepszego (zielony kolor) i zapis do pliku `.pkl`.

Plik `.pkl` zawiera: pipeline, listę kolumn, nazwę modelu i metryki.

### `test_models.py` — Testy jednostkowe i ewaluacja
Całkowicie przebudowany moduł testowy:
- **Testy integralności**: Sprawdza poprawność ładowania danych, czyszczenia i formatu wyjścia.
- **Weryfikacja Embeddings**: Upewnia się, że wektory GloVe są poprawnie generowane i mają odpowiedni wymiar (50d).
- **Symulacja treningu**: Trenuje lekki model RandomForest na próbce danych, aby wykryć błędy przed pełnym treningiem.
- **Raportowanie**: Generuje czytelny raport w konsoli (Classification Report) bez konieczności uruchamiania pełnego `main.py`.

Uruchomienie: `python test_models.py`

---

## 7. Gdzie program zapisuje wyniki?

Wszystkie wyniki trafiają do `DataPrep and Graphs/analytics/`.

- **`market_insights/`** — Analizy danych (korelacje, rozkłady wpływu).
- **`confusion_matrix/`** — Raporty jakości (macierze pomyłek, porównanie modeli).
- **`best_model.pkl`** — Finalny, zoptymalizowany model gotowy do użycia.

---

## 8. Tryby uruchamiania projektu

```bash
# Pełny pipeline
python main.py

# Tryb interaktywny (trening + zapis modelu)
python save_model.py

# Testy jednostkowe
python test_models.py
```

---

## 9. Interfejs Webowy / Client App (Dodatek)

Oprócz samego treningu dostępny jest interfejs webowy do testowania modeli "na żywo".
Wygląda jak **Terminal Orbitalny**, pozwala wpisywać własne tweety.

**Jak to działa:**
- **Serwer**: Flask (`main_server.py`), działa w trybie *inference*.
- **Dynamic Loading**: Skanuje folder i automatycznie dodaje nowe modele (`.pkl`) do menu.
- **Obsługa Embeddings**: Wykorzystuje `embeddings.py` do ładowania modeli GloVe.
- **Safety Valve**: "Bezpiecznik" — jeśli pewność modelu jest niska (różnica między RISE a DROP < 8%), serwer zwraca **UNCERTAIN**.
- **Frontend**: Czysty HTML/CSS/JS (folder `templates/` i `static/`), lekki, bez frameworków.

**Uruchomienie:**
```bash
python main_server.py
# Otwórz w przeglądarce: http://127.0.0.1:5000
```

<br>
<br>

---

# English Documentation / Dokumentacja w języku angielskim

## 1. Introduction
The **Trump Tweet Market Predictor** project aims to investigate and model the relationship between the content of Donald Trump's tweets and short-term financial market reactions, represented by the **S&P 500 index (ETF SPY)**.

The system automatically:
- fetches and cleans tweets,
- synchronizes them with market data,
- classifies their content by topic,
- measures the actual impact on the market,
- builds machine learning models to predict the market direction:
  - **-1**: Drop
  - **0**: Noise
  - **1**: Rise

The entire system is designed as a modular ecosystem consisting of 8 cooperating components.

---

## 2. System Architecture
The project is divided into two main stages:

**STAGE 0 – Dataset Construction (`DataPrep and Graphs`)**
Before ML models can be trained, raw tweets must be transformed into a structured dataset.

**STAGE 1 – Machine Learning and Predictive Analysis**
The ready-made dataset goes into the ML pipeline, where features, models, and evaluation reports are created.

---

## 3. STAGE 0 – Data Preparation

### 3.1. `tweets_output.csv` – Source Data
The input file contains raw Donald Trump tweets.
- **Separator**: `|`
- **Encoding**: `UTF-8`
- **Columns**: `datetime`, `tweet_text`

Content includes original tweets, retweets, links, special characters, and political, economic, and private content.

### 3.2. `data_pipeline.py` – Data Processing Pipeline
The **TrumpMarketPipeline** class is responsible for fully preparing the data for machine learning.
- **Market Ticker**: `SPY`
- **Start Date**: `2025-01-01`
- **Market Data Source**: `yfinance`
- **Interval**: `1 hour`

### 3.3. Tweet Classification Logic

**Level 1 – Noise Filter (`NOISE_TRIGGERS`)**
If a tweet contains phrases like: `happy birthday`, `congratulations`, `thank you`, `rally`, `vote for`, `merry christmas` — it is automatically marked as **Noise**.

**Level 2 – Topic Classification**
A tweet receives one or multiple categories if it contains keywords:

| Category | Sample Keywords |
| :--- | :--- |
| **ECONOMY** | inflation, economy, tax, tariff, fed, market |
| **GEOPOLITICS** | war, china, russia, ukraine, nato |
| **CRYPTO** | bitcoin, crypto, blockchain |
| **DOMESTIC_POLICY** | law, senate, house, bill, supreme court |

If it doesn't match any category → **OTHER** + Noise.

### 3.4. Market Synchronization
For each tweet:
1. the nearest future trading point is searched,
2. the percentage price change is calculated,
3. if the gap exceeds 4 days (weekends, holidays) → the tweet is ignored.

### 3.5. Final Dataset
The result of STAGE 0 is the `ready_for_ml_training.csv` file.

---

## 4. Data Visualization – `graph.py`
The module generates three main charts:

1. **Market Match**
   - Shows the S&P 500 price line, points of large rises and drops, and tweet moments.
2. **Topics Ranking**
   - Ranking of tweet categories with the highest market impact.
3. **Extended Correlation Matrix**
   - Correlation map between tweet categories, noise features, and actual market movements.

---

## 5. STAGE 1 – Machine Learning Pipeline

### 5.1. `data_loader`
- loads data,
- cleans missing values,
- creates the Target variable: `[-1, 0, 1]`.

### 5.2. `feature_engineer`
Creates ~15 additional features:
- **textual**: text length, word count, sentiment (TextBlob)
- **keywords**: occurrence counting
- **emotionality**: UPPERCASE, exclamation marks (!!!)
- **temporal**: hour, day of the week
- **categories**: binary thematic flags

### 5.3. `embeddings.py` (NEW)
Module responsible for text vectorization.
- Loads pre-trained **GloVe** (Global Vectors for Word Representation) vectors.
- Handles missing words and creates dense vectors (50 dimensions) for each tweet.
- Required both during training (`TrumpMarketPredict`) and on the server (`TrumpPredictionAI`).

### 5.4. `model_trainer`
Trains models in two variants (TF-IDF and GloVe). Models:
1. **Logistic Regression**
2. **Random Forest**
3. **Gradient Boosting**

Each model works in two variants:
- **TF-IDF Vectorizer**: classic statistical approach
- **GloVe Embeddings**: semantic understanding (using `glove-wiki-gigaword-50`)

Models are evaluated using: `accuracy`, `precision`, `recall`, `F1-score`.
The best model goes to **GridSearchCV** for further optimization.

### 5.5. `model_evaluator`
Generates classification reports, confusion matrices, and analyzes prediction errors.

---

## 6. Main Project Control Scripts

### `main.py` — Main Entry Point (Full Pipeline)
Runs the entire end-to-end process – from raw data to model comparison.
1. **Data Loading** (DataLoader)
2. **Feature Engineering** (FeatureEngineer)
3. **Exploratory Analysis** (optional)
4. **Model Training** (ModelTrainer)
5. **Fine-tuning the Best Model** (GridSearchCV)
6. **Detailed Evaluation** (reports, matrices)
7. **Model Comparison** (comparison charts)

### `save_model.py` — Interactive Mode and Model Saving
Provides an interactive text interface.
- **Train & Optimize**: same as `main.py`, but with interactive preview.
- **Save Model**: lists models, highlights the best one (green color), and saves it to a `.pkl` file.

The `.pkl` file contains: pipeline, column list, model name, and metrics.

### `test_models.py` — Unit Tests and Evaluation
Completely rebuilt testing module:
- **Integrity Tests**: Checks data loading, cleaning, and output format correctness.
- **Embeddings Verification**: Ensures GloVe vectors are generated correctly and have the proper dimension (50d).
- **Training Simulation**: Trains a lightweight RandomForest model on a data sample to catch errors before full training.
- **Reporting**: Generates a readable console report (Classification Report) without needing to run the full `main.py`.

Run: `python test_models.py`

---

## 7. Where Does the Program Save Results?

All results go to `DataPrep and Graphs/analytics/`.

- **`market_insights/`** — Data analysis (correlations, impact distributions).
- **`confusion_matrix/`** — Quality reports (confusion matrices, model comparison).
- **`best_model.pkl`** — Final, optimized model ready for use.

---

## 8. Project Execution Modes

```bash
# Full Pipeline
python main.py

# Interactive Mode (Training + Model Saving)
python save_model.py

# Unit Tests
python test_models.py
```

---

## 9. Web Interface / Client App (Bonus)

In addition to training, a web interface is available for "live" model testing.
It looks like an **Orbital Terminal** and allows you to enter your own tweets.

**How it works:**
- **Server**: Flask (`main_server.py`), runs in *inference* mode.
- **Dynamic Loading**: Skanuje folder i automatically adds new `.pkl` models to the menu.
- **Embeddings Support**: Uses `embeddings.py` to load GloVe-based models.
- **Safety Valve**: If model confidence is low (difference between RISE and DROP < 8%), the server returns **UNCERTAIN**.
- **Frontend**: Pure HTML/CSS/JS (folders `templates/` and `static/`), lightweight, no frameworks.

**Run:**
```bash
python main_server.py
# Open in browser: http://127.0.0.1:5000
```
