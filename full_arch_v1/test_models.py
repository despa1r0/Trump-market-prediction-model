"""
Unit Tests - Tests for all components (2p)
"""
import unittest
import pandas as pd
import numpy as np
from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer
import os



class TestDataLoader(unittest.TestCase):
    """Tests for DataLoader"""
    
    def setUp(self):
        """Prepare test data"""
        # Create minimal test CSV
        test_data = {
            'datetime': ['2025-01-01 12:00:00'] * 10,
            'tweet_text': ['test tweet'] * 10,
            'clean_text_nlp': ['this is a test tweet about market'] * 10,
            'categories': ['[]'] * 10,
            'is_noise': [True] * 10,
            'Market_Impact': [0.001, -0.002, 0.01, -0.008, 0.0, 0.003, -0.001, 0.02, -0.015, 0.005],
            'is_weekend_news': [False] * 10
        }
        self.test_df = pd.DataFrame(test_data)
        self.test_file = 'test_data.csv'
        self.test_df.to_csv(self.test_file, index=False)
        
    def tearDown(self):
        """Delete test files"""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_load_data(self):
        """Test data loading"""
        loader = DataLoader(self.test_file)
        df = loader.load_data()
        
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 10)
        self.assertIn('clean_text_nlp', df.columns)
    
    def test_clean_data(self):
        """Test data cleaning"""
        loader = DataLoader(self.test_file)
        loader.load_data()
        df = loader.clean_data()
        
        self.assertFalse(df['clean_text_nlp'].isnull().any())
        self.assertFalse(df['Market_Impact'].isnull().any())
    
    def test_create_target(self):
        """Test target variable creation"""
        loader = DataLoader(self.test_file)
        loader.load_data()
        loader.clean_data()
        target = loader.create_target(threshold=0.005)
        
        self.assertIn('target', loader.df.columns)
        self.assertTrue(set(target.unique()).issubset({-1, 0, 1}))
    
    def test_prepare_features(self):
        """Test feature preparation"""
        loader = DataLoader(self.test_file)
        loader.load_data()
        loader.clean_data()
        loader.create_target()
        X, y = loader.prepare_features_and_target()
        
        self.assertEqual(len(X), len(y))
        self.assertTrue(all(isinstance(x, str) for x in X))



class TestFeatureEngineer(unittest.TestCase):
    """Tests for FeatureEngineer"""
    
    def setUp(self):
        """Prepare test data"""
        test_data = {
            'datetime': pd.date_range('2025-01-01', periods=5),
            'clean_text_nlp': [
                'this is a test',
                'VERY IMPORTANT MESSAGE!!!',
                'economy and trade news',
                'short',
                'congress votes on new bill today'
            ],
            'categories': ['[]', "['ECONOMY']", '[]', "['POLITICS']", '[]'],
            'Market_Impact': [0.001, -0.002, 0.01, 0.0, 0.003],
            'target': [0, 0, 1, 0, 0]
        }
        self.test_df = pd.DataFrame(test_data)
    
    def test_text_length_features(self):
        """Test text length features creation"""
        engineer = FeatureEngineer(self.test_df)
        engineer.add_text_length_features()
        
        self.assertIn('text_length', engineer.df.columns)
        self.assertIn('word_count', engineer.df.columns)
        self.assertIn('avg_word_length', engineer.df.columns)
        
        # Check calculation correctness
        self.assertEqual(engineer.df.loc[0, 'word_count'], 4)
    
    def test_keyword_features(self):
        """Test keyword features creation"""
        engineer = FeatureEngineer(self.test_df)
        engineer.add_keyword_features()
        
        self.assertIn('economy_keyword_count', engineer.df.columns)
        self.assertIn('politics_keyword_count', engineer.df.columns)
        
        # Check: row 2 has economy and trade
        self.assertGreater(engineer.df.loc[2, 'economy_keyword_count'], 0)
    
    def test_capitalization_features(self):
        """Test capitalization features"""
        engineer = FeatureEngineer(self.test_df)
        engineer.add_capitalization_features()
        
        self.assertIn('uppercase_ratio', engineer.df.columns)
        self.assertIn('has_all_caps_word', engineer.df.columns)
        
        # Row 1 has many capital letters
        self.assertGreater(engineer.df.loc[1, 'uppercase_ratio'], 0.3)
    
    def test_punctuation_features(self):
        """Test punctuation features"""
        engineer = FeatureEngineer(self.test_df)
        engineer.add_punctuation_features()
        
        self.assertIn('exclamation_count', engineer.df.columns)
        self.assertIn('question_count', engineer.df.columns)
        
        # Row 1 has exclamation marks
        self.assertEqual(engineer.df.loc[1, 'exclamation_count'], 3)
    
    def test_temporal_features(self):
        """Test temporal features"""
        engineer = FeatureEngineer(self.test_df)
        engineer.add_temporal_features()
        
        self.assertIn('hour', engineer.df.columns)
        self.assertIn('day_of_week', engineer.df.columns)
        self.assertIn('is_weekend', engineer.df.columns)



class TestModelTrainer(unittest.TestCase):
    """Tests for ModelTrainer"""
    
    def setUp(self):
        """Prepare training data"""
        np.random.seed(42)
        
        # Create synthetic data
        texts = [
            'positive market news economy growth',
            'negative decline stocks falling',
            'neutral market no change',
        ] * 20  # 60 samples
        
        labels = [1, -1, 0] * 20
        
        self.X = pd.Series(texts)
        self.y = pd.Series(labels)
    
    def test_split_data(self):
        """Test data splitting"""
        trainer = ModelTrainer(self.X, self.y, test_size=0.2)
        trainer.split_data()
        
        self.assertIsNotNone(trainer.X_train)
        self.assertIsNotNone(trainer.X_test)
        self.assertEqual(len(trainer.X_train) + len(trainer.X_test), len(self.X))
    
    def test_create_models(self):
        """Test model creation"""
        trainer = ModelTrainer(self.X, self.y)
        trainer.split_data()
        trainer.create_models()
        
        self.assertGreater(len(trainer.models), 0)
        self.assertIn('LogisticRegression', trainer.models)
    
    def test_train_models(self):
        """Test model training"""
        trainer = ModelTrainer(self.X, self.y)
        trainer.split_data()
        trainer.create_models()
        trainer.train_all_models()
        
        self.assertGreater(len(trainer.results), 0)
        
        for name, result in trainer.results.items():
            self.assertIn('accuracy', result)
            self.assertIn('f1_score', result)
            self.assertGreaterEqual(result['accuracy'], 0)
            self.assertLessEqual(result['accuracy'], 1)



class TestModelPredictions(unittest.TestCase):
    """Tests for model predictions"""
    
    def test_prediction_format(self):
        """Test prediction format"""
        texts = ['test'] * 10
        labels = [0] * 10
        
        trainer = ModelTrainer(pd.Series(texts), pd.Series(labels))
        trainer.split_data()
        trainer.create_models()
        trainer.train_all_models()
        
        for name, result in trainer.results.items():
            predictions = result['predictions']
            
            # Predictions should be from set {-1, 0, 1}
            self.assertTrue(set(predictions).issubset({-1, 0, 1}))
            
            # Number of predictions = number of test samples
            self.assertEqual(len(predictions), len(trainer.y_test))



def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureEngineer))
    suite.addTests(loader.loadTestsFromTestCase(TestModelTrainer))
    suite.addTests(loader.loadTestsFromTestCase(TestModelPredictions))
    
    # Run
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Final result
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"✅ Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failed: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")
    
    return result.wasSuccessful()



if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
