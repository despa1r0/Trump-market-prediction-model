"""
Model Evaluator - Metrics analysis and model evaluation (2p)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)



class ModelEvaluator:
    """Class for detailed model evaluation"""
    
    def __init__(self, y_true, y_pred, model_name: str = "Model"):
        """
        Args:
            y_true: true labels
            y_pred: predicted labels
            model_name: model name
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.model_name = model_name
        
    def print_classification_report(self):
        """Print classification report"""
        print("\n" + "=" * 60)
        print(f"CLASSIFICATION REPORT: {self.model_name}")
        print("=" * 60)
        
        report = classification_report(
            self.y_true, 
            self.y_pred,
            target_names=['Drop (-1)', 'Noise (0)', 'Rise (1)'],
            digits=4
        )
        print(report)
        
    def plot_confusion_matrix(self, save_path: str = None):
        """Visualize confusion matrix"""
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Drop', 'Noise', 'Rise'],
            yticklabels=['Drop', 'Noise', 'Rise']
        )
        plt.title(f'Confusion Matrix: {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"✅ Confusion matrix saved: {save_path}")
        else:
            plt.savefig(f'confusion_matrix_{self.model_name}.png')
            print(f"✅ Confusion matrix saved: confusion_matrix_{self.model_name}.png")
        
        plt.close()
        
    def calculate_metrics(self) -> dict:
        """Calculate all metrics"""
        metrics = {
            'accuracy': accuracy_score(self.y_true, self.y_pred),
            'precision_macro': precision_score(self.y_true, self.y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(self.y_true, self.y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(self.y_true, self.y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0),
        }
        
        return metrics
    
    def print_metrics_summary(self):
        """Print metrics summary"""
        metrics = self.calculate_metrics()
        
        print("\n" + "=" * 60)
        print(f"METRICS: {self.model_name}")
        print("=" * 60)
        
        print(f"\n📊 General metrics:")
        print(f"   Accuracy:           {metrics['accuracy']:.4f}")
        
        print(f"\n📊 Macro-averaged (equal class weight):")
        print(f"   Precision (macro):  {metrics['precision_macro']:.4f}")
        print(f"   Recall (macro):     {metrics['recall_macro']:.4f}")
        print(f"   F1-Score (macro):   {metrics['f1_macro']:.4f}")
        
        print(f"\n📊 Weighted-averaged (by sample count):")
        print(f"   Precision (weighted): {metrics['precision_weighted']:.4f}")
        print(f"   Recall (weighted):    {metrics['recall_weighted']:.4f}")
        print(f"   F1-Score (weighted):  {metrics['f1_weighted']:.4f}")
        
    def analyze_errors(self):
        """Analyze model errors"""
        print("\n" + "=" * 60)
        print(f"ERROR ANALYSIS: {self.model_name}")
        print("=" * 60)
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        # Error counts by type
        total = len(self.y_true)
        correct = np.sum(self.y_true == self.y_pred)
        errors = total - correct
        
        print(f"\n✅ Correct predictions: {correct} ({correct/total*100:.1f}%)")
        print(f"❌ Errors: {errors} ({errors/total*100:.1f}%)")
        
        # Detailed error analysis
        print(f"\n📉 Error types:")
        
        classes = {-1: 'Drop', 0: 'Noise', 1: 'Rise'}
        
        for true_idx, true_label in enumerate([-1, 0, 1]):
            for pred_idx, pred_label in enumerate([-1, 0, 1]):
                if true_idx != pred_idx:
                    count = cm[true_idx, pred_idx]
                    if count > 0:
                        pct = count / np.sum(cm[true_idx, :]) * 100
                        print(f"   {classes[true_label]} → {classes[pred_label]}: {count} ({pct:.1f}%)")
    
    def full_evaluation(self):
        """Full model evaluation"""
        self.print_classification_report()
        self.print_metrics_summary()
        self.plot_confusion_matrix()
        self.analyze_errors()



class MultiModelComparison:
    """Class for comparing multiple models"""
    
    def __init__(self, results: dict, y_test):
        """
        Args:
            results: dictionary {model_name: {'predictions': y_pred, ...}}
            y_test: true labels
        """
        self.results = results
        self.y_test = y_test
        
    def compare_all_models(self):
        """Compare all models"""
        print("\n" + "🏆" * 30)
        print("ALL MODELS COMPARISON")
        print("🏆" * 30 + "\n")
        
        comparison_data = []
        
        for model_name, result in self.results.items():
            y_pred = result['predictions']
            
            metrics = {
                'Model': model_name,
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
                'Recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
                'F1-Score': f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
            }
            
            comparison_data.append(metrics)
        
        df = pd.DataFrame(comparison_data).sort_values('Accuracy', ascending=False)
        
        print(df.to_string(index=False))
        
        # Visualization
        self.plot_model_comparison(df)
        
        return df
    
    def plot_model_comparison(self, df: pd.DataFrame):
        """Plot model comparison chart"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(df))
        width = 0.2
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            ax.bar(x + i * width, df[metric], width, label=metric, color=color)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        print("\n✅ Comparison chart saved: model_comparison.png")
        plt.close()



# Usage example
if __name__ == "__main__":
    from data_loader import DataLoader
    from model_trainer import ModelTrainer
    
    loader = DataLoader("ready_for_ml_training.csv")
    loader.load_data()
    loader.clean_data()
    loader.create_target(threshold=0.005)
    X, y = loader.prepare_features_and_target()
    
    trainer = ModelTrainer(X, y, test_size=0.2, random_state=42)
    trainer.split_data()
    trainer.create_models()
    trainer.train_all_models()
    
    # Evaluate each model
    print("\n" + "📊" * 30)
    for model_name, result in trainer.results.items():
        evaluator = ModelEvaluator(
            trainer.y_test, 
            result['predictions'], 
            model_name
        )
        evaluator.full_evaluation()
    
    # Compare models
    comparator = MultiModelComparison(trainer.results, trainer.y_test)
    comparator.compare_all_models()
