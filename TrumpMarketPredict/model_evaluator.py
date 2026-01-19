"""
Model Evaluator - Metrics analysis and model evaluation (2p)
Ewaluator modelu - Analiza metryk i ocena modelu (2p)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

class ModelEvaluator:
    """Class for detailed model evaluation / Klasa do szczegółowej oceny modelu"""

    def __init__(self, y_true, y_pred, model_name: str = "Model", output_dir: str = "matrix_reports"):
        """
        Args:
            y_true: true labels / prawdziwe etykiety
            y_pred: predicted labels / przewidywane etykiety
            model_name: model name / nazwa modelu
            output_dir: directory for plots / katalog na wykresy
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.model_name = model_name
        self.output_dir = output_dir

    def print_classification_report(self):
        """Print classification report / Wyświetl raport klasyfikacji"""
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

    def plot_confusion_matrix(self):
        """Visualize confusion matrix / Wizualizacja macierzy pomyłek"""
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

        save_path = os.path.join(self.output_dir, f'confusion_matrix_{self.model_name}.png')
        plt.savefig(save_path)
        print(f"✅ Confusion matrix saved: {save_path}")
        plt.close()

    def calculate_metrics(self) -> dict:
        """Calculate all metrics / Oblicz wszystkie metryki"""
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
        """Print metrics summary / Wyświetl podsumowanie metryk"""
        metrics = self.calculate_metrics()
        print("\n" + "=" * 60)
        print(f"METRICS: {self.model_name}")
        print("=" * 60)
        print(f"\n📊 General metrics:")
        print(f"   Accuracy:           {metrics['accuracy']:.4f}")
        print(f"\n📊 Weighted-averaged (by sample count):")
        print(f"   Precision (weighted): {metrics['precision_weighted']:.4f}")
        print(f"   Recall (weighted):    {metrics['recall_weighted']:.4f}")
        print(f"   F1-Score (weighted):  {metrics['f1_weighted']:.4f}")

    def analyze_errors(self):
        """Analyze model errors / Analiza błędów modelu"""
        print("\n" + "=" * 60)
        print(f"ERROR ANALYSIS: {self.model_name}")
        print("=" * 60)
        cm = confusion_matrix(self.y_true, self.y_pred)
        total = len(self.y_true)
        correct = np.sum(self.y_true == self.y_pred)
        errors = total - correct
        print(f"\n✅ Correct predictions: {correct} ({correct/total*100:.1f}%)")
        print(f"❌ Errors: {errors} ({errors/total*100:.1f}%)")

    def full_evaluation(self):
        """Full model evaluation / Pełna ocena modelu"""
        self.print_classification_report()
        self.print_metrics_summary()
        self.plot_confusion_matrix()
        self.analyze_errors()

class MultiModelComparison:
    """Class for comparing multiple models / Klasa do porównywania wielu modeli"""

    def __init__(self, results: dict, y_test, output_dir: str = "matrix_reports"):
        """
        Args:
            results: dict of predictions / słownik przewidywań
            y_test: true labels / prawdziwe etykiety
            output_dir: directory for plots / katalog na wykresy
        """
        self.results = results
        self.y_test = y_test
        self.output_dir = output_dir

    def compare_all_models(self):
        """Compare all models / Porównaj wszystkie modele"""
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
        self.plot_model_comparison(df)
        return df

    def plot_model_comparison(self, df: pd.DataFrame):
        """Plot model comparison chart / Wykres porównawczy modeli"""
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
        save_path = os.path.join(self.output_dir, 'model_comparison.png')
        plt.savefig(save_path)
        print(f"\n✅ Comparison chart saved: {save_path}")
        plt.close()