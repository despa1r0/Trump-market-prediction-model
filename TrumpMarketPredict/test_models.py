import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
import warnings
import os

# Import custom modules
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from embeddings import BertVectorizer, GloveVectorizer
from model_evaluator import ModelEvaluator

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# PART 1: CRITICAL UNIT TESTS
# -------------------------------------------------------------------------

class TestFeatureEngineer(unittest.TestCase):
    """
    Critical tests to ensure feature engineering works correctly.
    """
    
    def setUp(self):
        """Prepare minimal test data"""
        test_data = {
            'datetime': pd.date_range('2025-01-01', periods=3),
            'clean_text_nlp': [
                'market is booming today',
                'stocks crash due to panic',
                'neutral stance on trade'
            ],
            'categories': ['[]', "['ECONOMY']", '[]'],
            'Market_Impact': [0.01, -0.02, 0.0],
            'target': [1, -1, 0]
        }
        self.test_df = pd.DataFrame(test_data)
    
    def test_feature_creation(self):
        """Test that features are created without errors"""
        engineer = FeatureEngineer(self.test_df)
        df = engineer.create_all_features()
        
        # Check for key feature columns
        expected_cols = ['text_length', 'word_count', 'uppercase_ratio', 'hour', 'day_of_week']
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Feature '{col}' missing")
            
        # Check no NaNs were introduced
        self.assertFalse(df[expected_cols].isnull().any().any(), "Features contain NaNs")


class TestDataLoader(unittest.TestCase):
    """
    Critical tests for data loading.
    """
    def setUp(self):
        self.test_file = 'test_temp.csv'
        pd.DataFrame({'a': [1, 2], 'clean_text_nlp': ['x', 'y'], 'Market_Impact': [0.1, -0.1]}).to_csv(self.test_file)

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_load_and_clean(self):
        loader = DataLoader(self.test_file)
        loader.load_data()
        df = loader.clean_data()
        self.assertIn('clean_text_nlp', df.columns)
        self.assertEqual(len(df), 2)


# -------------------------------------------------------------------------
# PART 2: EMBEDDING COMPARISON
# -------------------------------------------------------------------------

def get_numeric_features(df):
    """Identifies numeric features to keep."""
    exclude_cols = ['datetime', 'tweet_text', 'clean_text_nlp', 'categories', 
                    'is_noise', 'Market_Impact', 'is_weekend_news', 'target']
    
    numeric_features = [col for col in df.columns 
                       if col not in exclude_cols 
                       and df[col].dtype in ['int64', 'float64']]
    return numeric_features

def run_embedding_comparison(): 
    print("=" * 60)
    print(" EMBEDDING COMPARISON: TF-IDF vs GloVe vs BERT")
    print(" MODEL: RANDOM FOREST CLASSIFIER")
    print("=" * 60)

    # 1. LOAD DATA
    print("\n[*] Loading data...")
    try:
        loader = DataLoader("ready_for_ml_training.csv")
        loader.load_data()
        loader.clean_data()
        loader.create_target(threshold=0.002)
        
        engineer = FeatureEngineer(loader.get_full_dataframe())
        df = engineer.create_all_features()
        print(f"    [INFO] Data shape: {df.shape}")
    except Exception as e:
        print(f"    [ERROR] Could not load processed data: {e}")
        print("    [TIP] Run 'python TrumpMarketPredict/main.py' first to generate data.")
        return

    if 'target' not in df.columns:
        print("[ERROR] 'target' column missing.")
        return

    # 2. SPLIT
    numeric_features = get_numeric_features(df)
    X = df
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    [INFO] Train size: {len(X_train)} | Test size: {len(X_test)}")

    # 3. PIPELINES (Random Forest)
    # n_jobs=-1 uses all CPU cores for speed
    rf_settings = dict(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)

    pipelines = {}

    # TF-IDF
    pipelines['TF-IDF'] = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('text', TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), 'clean_text_nlp'),
            ('numeric', StandardScaler(), numeric_features)
        ])),
        ('clf', RandomForestClassifier(**rf_settings))
    ])

    # GloVe
    pipelines['GloVe'] = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('text', GloveVectorizer(model_name='glove-wiki-gigaword-50'), 'clean_text_nlp'),
            ('numeric', StandardScaler(), numeric_features)
        ])),
        ('clf', RandomForestClassifier(**rf_settings))
    ])

    # BERT
    pipelines['BERT (MiniLM)'] = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('text', BertVectorizer(model_name='all-MiniLM-L6-v2'), 'clean_text_nlp'),
            ('numeric', StandardScaler(), numeric_features)
        ])),
        ('clf', RandomForestClassifier(**rf_settings))
    ])

    # 4. TRAINING & EVALUATION
    results = []
    
    for name, pipeline in pipelines.items():
        print(f"\n[*] Training {name}...")
        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # --- MODEL EVALUATOR INTEGRATION ---
            evaluator = ModelEvaluator(y_test, y_pred, model_name=name)
            
            # Print text-based reports (No graphs)
            evaluator.print_classification_report()
            evaluator.print_metrics_summary()
            evaluator.analyze_errors()
            
            # Collect simple metrics for final leaderboard
            metrics = evaluator.calculate_metrics()
            results.append({
                'Model': name, 
                'F1-Score': metrics['f1_weighted'], 
                'Accuracy': metrics['accuracy']
            })
            
        except Exception as e:
            print(f"    [FAIL] {name}: {e}")

    # 5. RESULTS
    print("\n" + "=" * 60)
    print(" FINAL LEADERBOARD (Random Forest)")
    print("=" * 60)
    res_df = pd.DataFrame(results).sort_values('F1-Score', ascending=False)
    print(res_df.to_string(index=False))


    print("=" * 60)
    print(" EMBEDDING COMPARISON: TF-IDF vs GloVe vs BERT")
    print(" MODEL: RANDOM FOREST CLASSIFIER (MANUAL WEIGHTS)")
    print("=" * 60)

    # 1. LOAD DATA
    print("\n[*] Loading data...")
    try:
        loader = DataLoader("ready_for_ml_training.csv")
        loader.load_data()
        loader.clean_data()
        # ВАЖНО: Можете также попробовать уменьшить порог здесь, например до 0.0025
        loader.create_target(threshold=0.005) 
        
        engineer = FeatureEngineer(loader.get_full_dataframe())
        df = engineer.create_all_features()
        print(f"    [INFO] Data shape: {df.shape}")
    except Exception as e:
        print(f"    [ERROR] Could not load processed data: {e}")
        print("    [TIP] Run 'python TrumpMarketPredict/main.py' first to generate data.")
        return

    if 'target' not in df.columns:
        print("[ERROR] 'target' column missing.")
        return

    # 2. SPLIT
    numeric_features = get_numeric_features(df)
    X = df
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    [INFO] Train size: {len(X_train)} | Test size: {len(X_test)}")

    # 3. PIPELINES (Random Forest)
    
    # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
    # Ручная настройка весов классов.
    # Формат: {класс: вес}
    # -1 (Drop): Вес 5 (Самый важный, ошибки здесь стоят дорого)
    #  0 (Noise): Вес 1 (Базовый, их и так много)
    #  1 (Rise): Вес 3 (Важный, но падение для нас критичнее)
    manual_weights = {-1: 10, 0: 1, 1: 5}
    
    rf_settings = dict(
        n_estimators=100, 
        class_weight=manual_weights,  # Используем ручные веса вместо 'balanced'
        random_state=42, 
        n_jobs=-1
    )
    # -----------------------

    pipelines = {}

    # TF-IDF
    pipelines['TF-IDF'] = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('text', TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), 'clean_text_nlp'),
            ('numeric', StandardScaler(), numeric_features)
        ])),
        ('clf', RandomForestClassifier(**rf_settings))
    ])

    # GloVe
    pipelines['GloVe'] = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('text', GloveVectorizer(model_name='glove-wiki-gigaword-50'), 'clean_text_nlp'),
            ('numeric', StandardScaler(), numeric_features)
        ])),
        ('clf', RandomForestClassifier(**rf_settings))
    ])

    # BERT
    pipelines['BERT (MiniLM)'] = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('text', BertVectorizer(model_name='all-MiniLM-L6-v2'), 'clean_text_nlp'),
            ('numeric', StandardScaler(), numeric_features)
        ])),
        ('clf', RandomForestClassifier(**rf_settings))
    ])

    # 4. TRAINING & EVALUATION
    results = []
    
    for name, pipeline in pipelines.items():
        print(f"\n[*] Training {name}...")
        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # --- MODEL EVALUATOR INTEGRATION ---
            evaluator = ModelEvaluator(y_test, y_pred, model_name=name)
            
            evaluator.print_classification_report()
            evaluator.print_metrics_summary()
            evaluator.analyze_errors()
            
            metrics = evaluator.calculate_metrics()
            results.append({
                'Model': name, 
                'F1-Score': metrics['f1_weighted'], 
                'Accuracy': metrics['accuracy']
            })
            
        except Exception as e:
            print(f"    [FAIL] {name}: {e}")

    # 5. RESULTS
    print("\n" + "=" * 60)
    print(" FINAL LEADERBOARD (Random Forest - Manual Weights)")
    print("=" * 60)
    res_df = pd.DataFrame(results).sort_values('F1-Score', ascending=False)
    print(res_df.to_string(index=False))

if __name__ == '__main__':
    # Step 1: Run Critical Unit Tests
    print("Running Critical Unit Tests...")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureEngineer))
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoader))
    
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    
    if not result.wasSuccessful():
        print("\n[!] Unit tests failed. Fix them before running the comparison.")
        exit(1)
        
    print("\n[OK] Unit tests passed. Starting Embedding Comparison...\n")
    
    # Step 2: Run The Real Battle
    run_embedding_comparison()
