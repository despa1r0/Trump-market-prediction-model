"""
Model Trainer - Model training and comparison (2p + 2p)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from typing import Dict, Any



class ModelTrainer:
    """Class for training and comparing models"""
    
    def __init__(self, X: pd.Series, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
        """
        Args:
            X: features (text)
            y: target variable
            test_size: test set size
            random_state: seed for reproducibility
        """
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.models = {}
        self.results = {}
        
    def split_data(self):
        """Split into train/test"""
        print("✂️  Splitting data into train/test...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=self.y
        )
        
        print(f"   📊 Train: {len(self.X_train)} samples")
        print(f"   📊 Test:  {len(self.X_test)} samples")
        print(f"\n   Class distribution in train:")
        print(self.y_train.value_counts().sort_index())
        
    def create_models(self):
        """Create different models"""
        print("\n🤖 Creating models...")
        
        # 1. Logistic Regression (baseline)
        self.models['LogisticRegression'] = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000, random_state=self.random_state))
        ])
        
        # 2. Naive Bayes
        self.models['NaiveBayes'] = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', MultinomialNB())
        ])
        
        # 3. Random Forest
        self.models['RandomForest'] = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000)),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=self.random_state))
        ])
        
        # 4. Gradient Boosting
        self.models['GradientBoosting'] = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000)),
            ('clf', GradientBoostingClassifier(n_estimators=100, random_state=self.random_state))
        ])
        
        # 5. SVM (optional, slow)
        # self.models['SVM'] = Pipeline([
        #     ('tfidf', TfidfVectorizer(max_features=2000)),
        #     ('clf', SVC(kernel='linear', random_state=self.random_state))
        # ])
        
        print(f"   ✅ Created {len(self.models)} models")
        
    def train_all_models(self):
        """Train all models"""
        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)
        
        for name, model in self.models.items():
            print(f"\n🔄 Training {name}...")
            try:
                model.fit(self.X_train, self.y_train)
                
                # Predictions
                y_pred = model.predict(self.X_test)
                
                # Metrics
                from sklearn.metrics import accuracy_score, f1_score
                acc = accuracy_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred, average='weighted')
                
                self.results[name] = {
                    'model': model,
                    'accuracy': acc,
                    'f1_score': f1,
                    'predictions': y_pred
                }
                
                print(f"   ✅ Accuracy: {acc:.4f}")
                print(f"   ✅ F1-Score: {f1:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                
    def fine_tune_best_model(self):
        """Fine tune best model"""
        print("\n" + "=" * 60)
        print("FINE TUNING BEST MODEL")
        print("=" * 60)
        
        # Find best model
        best_name = max(self.results.keys(), key=lambda k: self.results[k]['accuracy'])
        print(f"\n🏆 Best model: {best_name}")
        
        # Grid Search for LogisticRegression
        if best_name == 'LogisticRegression':
            print("\n🔧 Running GridSearchCV...")
            
            param_grid = {
                'tfidf__max_features': [3000, 5000, 7000],
                'tfidf__ngram_range': [(1, 1), (1, 2)],
                'clf__C': [0.1, 1.0, 10.0],
                'clf__solver': ['lbfgs', 'liblinear']
            }
            
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer()),
                ('clf', LogisticRegression(max_iter=1000, random_state=self.random_state))
            ])
            
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            print(f"\n✅ Best parameters:")
            for param, value in grid_search.best_params_.items():
                print(f"   {param}: {value}")
            
            print(f"\n✅ Best score: {grid_search.best_score_:.4f}")
            
            # Update results
            y_pred = grid_search.predict(self.X_test)
            from sklearn.metrics import accuracy_score, f1_score
            
            self.results['LogisticRegression_Tuned'] = {
                'model': grid_search.best_estimator_,
                'accuracy': accuracy_score(self.y_test, y_pred),
                'f1_score': f1_score(self.y_test, y_pred, average='weighted'),
                'predictions': y_pred
            }
            
    def save_best_model(self, filepath: str = 'best_model.pkl'):
        """Save best model"""
        best_name = max(self.results.keys(), key=lambda k: self.results[k]['accuracy'])
        best_model = self.results[best_name]['model']
        
        joblib.dump(best_model, filepath)
        print(f"\n💾 Model {best_name} saved to {filepath}")
        
    def get_comparison_table(self) -> pd.DataFrame:
        """Model comparison table"""
        comparison = []
        
        for name, result in self.results.items():
            comparison.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'F1-Score': result['f1_score']
            })
        
        df = pd.DataFrame(comparison).sort_values('Accuracy', ascending=False)
        return df



# Usage example
if __name__ == "__main__":
    from data_loader import DataLoader
    
    loader = DataLoader("ready_for_ml_training.csv")
    loader.load_data()
    loader.clean_data()
    loader.create_target(threshold=0.005)
    X, y = loader.prepare_features_and_target()
    
    trainer = ModelTrainer(X, y, test_size=0.2, random_state=42)
    trainer.split_data()
    trainer.create_models()
    trainer.train_all_models()
    
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(trainer.get_comparison_table().to_string(index=False))
    
    trainer.fine_tune_best_model()
    trainer.save_best_model()
