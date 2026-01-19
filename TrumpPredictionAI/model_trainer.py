"""
Model Trainer using ALL Features (Text + Engineered Features)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import joblib


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects specific columns from the DataFrame"""
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.columns]


class TextSelector(BaseEstimator, TransformerMixin):
    """Extracts the text column"""
    def __init__(self, key='clean_text_nlp'):
        self.key = key
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.key].astype(str)
        return X


class ModelTrainer:
    """Class for training models USING ALL FEATURES"""
    
    def __init__(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """
        Args:
            df: DataFrame with ALL features (text + engineered features)
            test_size: size of the test set
            random_state: for reproducibility
        """
        self.df = df
        self.test_size = test_size
        self.random_state = random_state
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        self.models = {}
        self.results = {}
        
    def split_data(self):
        """Splitting into train/test"""
        print("✂️  Splitting data into train/test...")
        
        # Check for target presence
        if 'target' not in self.df.columns:
            raise ValueError("Column 'target' not found!")
        
        X = self.df.drop(['target'], axis=1)
        y = self.df['target']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"   📊 Train: {len(self.X_train)} samples")
        print(f"   📊 Test:  {len(self.X_test)} samples")
        print(f"\n   Class distribution in train:")
        print(self.y_train.value_counts().sort_index())
        
    def get_numeric_features(self):
        """Get list of numeric features"""
        # Exclude utility columns
        exclude_cols = ['datetime', 'tweet_text', 'clean_text_nlp', 'categories', 
                        'is_noise', 'Market_Impact', 'is_weekend_news', 'target']
        
        numeric_features = [col for col in self.df.columns 
                           if col not in exclude_cols 
                           and self.df[col].dtype in ['int64', 'float64']]
        
        return numeric_features
        
    def create_models(self):
        """Creating models with combined features"""
        print("\n🤖 Creating models with TEXT + NUMERIC features...")
        
        numeric_features = self.get_numeric_features()
        print(f"   📊 Numeric features: {len(numeric_features)}")
        if numeric_features:
            print(f"   📝 Examples: {numeric_features[:5]}")
        
        # 1. TEXT ONLY (baseline)
        self.models['LogReg_TextOnly'] = Pipeline([
            ('text', TextSelector('clean_text_nlp')),
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, 
                                     random_state=self.random_state))
        ])
        
        # 2. TEXT + NUMERIC FEATURES (if available)
        if numeric_features:
            from sklearn.compose import ColumnTransformer
            
            # Create transformer for feature combination
            preprocessor = ColumnTransformer(
                transformers=[
                    ('text', TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), 
                     'clean_text_nlp'),
                    ('numeric', StandardScaler(), numeric_features)
                ],
                remainder='drop'
            )
            
            self.models['LogReg_Combined'] = Pipeline([
                ('features', preprocessor),
                ('clf', LogisticRegression(class_weight='balanced', max_iter=1000,
                                         random_state=self.random_state))
            ])
            
            self.models['RandomForest_Combined'] = Pipeline([
                ('features', preprocessor),
                ('clf', RandomForestClassifier(class_weight='balanced', n_estimators=100,
                                             random_state=self.random_state))
            ])
            
            self.models['GradientBoosting_Combined'] = Pipeline([
                ('features', preprocessor),
                ('clf', GradientBoostingClassifier(n_estimators=100,
                                                 random_state=self.random_state))
            ])
        
        print(f"   ✅ Created {len(self.models)} models")
        
    def train_all_models(self):
        """Training all models"""
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
                import traceback
                traceback.print_exc()
                
    def fine_tune_best_model(self):
        """
        Fine-tuning the best model using GridSearchCV.
        Finds the best model from the initial training and optimizes its hyperparameters.
        """
        print("\n" + "=" * 60)
        print("FINE-TUNING THE BEST MODEL (Grid Search)")
        print("=" * 60)
        
        if not self.results:
            print("❌ No trained models! Run train_all_models() first.")
            return
        
        # 1. Identify the winner from initial training
        best_name = max(self.results.keys(), key=lambda k: self.results[k]['f1_score'])
        print(f"🏆 Initial Winner: {best_name}")
        print(f"   Base F1-Score: {self.results[best_name]['f1_score']:.4f}")
        
        # 2. Define parameter grids for potential winners
        # Keys must allow matching step names in the Pipeline ('clf' or 'tfidf')
        param_grids = {
            'LogReg_Combined': {
                'clf__C': [0.1, 1, 10, 100],
                'clf__solver': ['liblinear', 'lbfgs']
            },
            'LogReg_TextOnly': {
                'clf__C': [0.1, 1, 10],
                'tfidf__max_features': [3000, 5000, 7000]
            },
            'RandomForest_Combined': {
                'clf__n_estimators': [100, 200, 300],
                'clf__max_depth': [None, 10, 20],
                'clf__min_samples_split': [2, 5]
            },
            'GradientBoosting_Combined': {
                'clf__n_estimators': [100, 200, 300],
                'clf__learning_rate': [0.01, 0.1, 0.2],
                'clf__max_depth': [3, 5, 7]
            }
        }
        
        # 3. Check if we have a grid for the winner
        if best_name not in param_grids:
            print(f"⚠️ No hyperparameter grid defined for {best_name}. Skipping tuning.")
            return

        print(f"\n⚙️  Starting Grid Search for {best_name}...")
        print("   (This involves training many models, please wait...)")
        
        # 4. Run GridSearchCV
        # cv=3: 3-fold cross-validation
        # n_jobs=-1: use all CPU cores
        grid_search = GridSearchCV(
            self.models[best_name], 
            param_grids[best_name], 
            cv=3, 
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # 5. Show results
        print(f"\n✅ Optimization Complete!")
        print(f"   Best Params: {grid_search.best_params_}")
        
        # 6. Evaluate optimized model on Test set
        best_model_optimized = grid_search.best_estimator_
        y_pred_opt = best_model_optimized.predict(self.X_test)
        
        from sklearn.metrics import accuracy_score, f1_score
        new_acc = accuracy_score(self.y_test, y_pred_opt)
        new_f1 = f1_score(self.y_test, y_pred_opt, average='weighted')
        
        print(f"\n📊 Results after Tuning:")
        print(f"   Old F1: {self.results[best_name]['f1_score']:.4f}")
        print(f"   New F1: {new_f1:.4f}")
        
        # 7. Update results so save_best_model uses the tuned version
        self.results[best_name]['model'] = best_model_optimized
        self.results[best_name]['f1_score'] = new_f1
        self.results[best_name]['accuracy'] = new_acc
        
    def save_best_model(self, filepath: str = 'best_model.pkl'):
        """Saving the best model"""
        if not self.results:
            print("❌ No trained models!")
            return
            
        best_name = max(self.results.keys(), key=lambda k: self.results[k]['f1_score'])
        best_model = self.results[best_name]['model']
        
        joblib.dump(best_model, filepath)
        print(f"\n💾 Model {best_name} (Best Version) saved to {filepath}")
        
    def get_comparison_table(self) -> pd.DataFrame:
        """Model comparison table"""
        comparison = []
        
        for name, result in self.results.items():
            comparison.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'F1-Score': result['f1_score']
            })
        
        df = pd.DataFrame(comparison).sort_values('F1-Score', ascending=False)
        return df


# Usage example
if __name__ == "__main__":
    from data_loader import DataLoader
    from feature_engineering import FeatureEngineer
    
    # 1. Load data
    loader = DataLoader("ready_for_ml_training.csv")
    loader.load_data()
    loader.clean_data()
    loader.create_target()
    
    # 2. CREATING ALL FEATURES
    print("\n" + "=" * 60)
    print("FEATURE CREATION")
    print("=" * 60)
    engineer = FeatureEngineer(loader.get_full_dataframe())
    df_with_features = engineer.create_all_features()
    
    # 3. Training using ALL features
    trainer = ModelTrainer(df_with_features, test_size=0.2, random_state=42)
    trainer.split_data()
    trainer.create_models()
    trainer.train_all_models()
    
    # 4. Comparison
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (Initial)")
    print("=" * 60)
    print(trainer.get_comparison_table().to_string(index=False))
    
    # 5. 🔥 FINE TUNING (Added Step)
    trainer.fine_tune_best_model()
    
    # 6. Saving
    trainer.save_best_model()