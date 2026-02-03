"""
Main Pipeline
Main - Wersja poprawiona z wykorzystaniem wszystkich cech
"""
import sys
import os # Import for path management / Import do zarządzania ścieżkami
from colorama import init, Fore, Style
from data_loader import DataLoader
from exploratory_analysis import ExploratoryAnalysis
from feature_engineering import FeatureEngineer
from model_evaluator import ModelEvaluator, MultiModelComparison

# Using updated trainer / Użycie zaktualizowanego trenera
from model_trainer import ModelTrainer

init(autoreset=True)

# --- PATH CONFIGURATION / KONFIGURACJA ŚCIEŻEK ---
# Base folder for graphs based on your architecture / Folder bazowy dla wykresów na podstawie architektury
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYTICS_PATH = os.path.join(BASE_DIR, "DataPrep and Graphs", "analytics")
MATRIX_DIR = os.path.join(ANALYTICS_PATH, "matrix_reports")
INSIGHTS_DIR = os.path.join(ANALYTICS_PATH, "market_insights")

# Create directories if they don't exist / Utwórz foldery, jeśli nie istnieją
os.makedirs(MATRIX_DIR, exist_ok=True)
os.makedirs(INSIGHTS_DIR, exist_ok=True)

def print_header(text: str):
    """Formatted header / Sformatowany nagłówek"""
    print("\n" + "=" * 70)
    print(f"{Fore.CYAN}{text}{Style.RESET_ALL}")
    print("=" * 70 + "\n")

def main():
    """Main pipeline function / Główna funkcja rurociągu"""

    print_header("[*] TRUMP TWEET MARKET IMPACT PREDICTION")
    print("Project: Predicting tweet impact on financial markets")
    print("Version: 2.0 - USING ALL FEATURES\n")

    # ===== STEP 1: DATA LOADING / KROK 1: ŁADOWANIE DANYCH =====
    print_header("[*] STEP 1: DATA LOADING AND CLEANING")

    try:
        loader = DataLoader("ready_for_ml_training.csv")
        loader.load_data()
        loader.clean_data()
        loader.create_target()
    except Exception as e:
        print(f"{Fore.RED}[X] Data loading error: {e}{Style.RESET_ALL}")
        sys.exit(1)

    # ===== STEP 2: DATA ANALYSIS / KROK 2: ANALIZA DANYCH =====
    print_header("[*] STEP 2: EXPLORATORY DATA ANALYSIS")

    try:
        # Pass INSIGHTS_DIR to save EDA results there / Przekaż INSIGHTS_DIR, aby tam zapisać wyniki EDA
        analyzer = ExploratoryAnalysis(loader.get_full_dataframe(), output_dir=INSIGHTS_DIR)
        analyzer.generate_full_report()
    except Exception as e:
        print(f"{Fore.YELLOW}[WARN] Analysis error: {e}{Style.RESET_ALL}")

    # ===== STEP 3: FEATURE ENGINEERING / KROK 3: INŻYNIERIA CECH =====
    print_header("[*] STEP 3: FEATURE ENGINEERING")

    try:
        engineer = FeatureEngineer(loader.get_full_dataframe())
        df_with_features = engineer.create_all_features()

        # Show created features / Pokaż utworzone cechy
        features = engineer.get_feature_names()
        print(f"\n[OK] Created {len(features)} new features:")
        for feat in features[:15]:
            print(f"   - {feat}")
        if len(features) > 15:
            print(f"   ... and {len(features) - 15} more features")

    except Exception as e:
        print(f"{Fore.RED}[X] Feature engineering error: {e}{Style.RESET_ALL}")
        sys.exit(1)

    # ===== STEP 4: TRAINING PREPARATION / KROK 4: PRZYGOTOWANIE DO TRENINGU =====
    print_header("[*] STEP 4: TRAIN/TEST PREPARATION")

    print(f"📊 Total features in dataset: {len(df_with_features.columns)}")
    print(f"📝 Data size: {df_with_features.shape}")

    # ===== STEP 5: MODEL TRAINING / KROK 5: TRENOWANIE MODELI =====
    print_header("[*] STEP 5: MODEL TRAINING (WITH ALL FEATURES)")

    try:
        trainer = ModelTrainer(df_with_features, test_size=0.2, random_state=42)
        trainer.split_data()
        trainer.create_models()
        trainer.train_all_models()

        print(f"\n{Fore.GREEN}[OK] Trained {len(trainer.results)} models{Style.RESET_ALL}")

    except Exception as e:
        print(f"{Fore.RED}[X] Training error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ===== STEP 6: FINE TUNING / KROK 6: DOSTRAJANIE MODELU =====
    print_header("[*] STEP 6: FINE TUNING BEST MODEL")

    try:
        trainer.fine_tune_best_model()
    except Exception as e:
        print(f"{Fore.YELLOW}[WARN] Fine tuning error: {e}{Style.RESET_ALL}")

    # ===== STEP 7: MODEL EVALUATION / KROK 7: OCENA MODELU =====
    print_header("[*] STEP 7: MODEL EVALUATION AND COMPARISON")

    try:
        # Detailed evaluation with MATRIX_DIR / Szczegółowa ocena z użyciem MATRIX_DIR
        for model_name, result in trainer.results.items():
            evaluator = ModelEvaluator(
                trainer.y_test,
                result['predictions'],
                model_name,
                output_dir=MATRIX_DIR
            )
            evaluator.full_evaluation()

        # Comparison with MATRIX_DIR / Porównanie z użyciem MATRIX_DIR
        print_header("[*] FINAL MODEL COMPARISON")
        comparator = MultiModelComparison(trainer.results, trainer.y_test, output_dir=MATRIX_DIR)
        comparison_df = comparator.compare_all_models()

        # Saving the best model / Zapisywanie najlepszego modelu
        trainer.save_best_model('best_model.pkl')

    except Exception as e:
        print(f"{Fore.RED}[X] Evaluation error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ===== FINAL / ZAKOŃCZENIE =====
    print_header("[*] PROJECT COMPLETED")

    print(f"{Fore.GREEN}[OK] All stages completed successfully!{Style.RESET_ALL}\n")
    print(f"📁 Results saved in: {ANALYTICS_PATH}")

    print(f"\n{Fore.CYAN}[INFO] Best model details:{Style.RESET_ALL}")
    best_model = comparison_df.iloc[0]
    print(f"   Name:      {best_model['Model']}")
    print(f"   Accuracy:  {best_model['Accuracy']:.4f}")
    print(f"   F1-Score:  {best_model['F1-Score']:.4f}")
    
    print("\n" + "=" * 70)
    print(f"{Fore.GREEN}[DONE] MAKE MODELS GREAT AGAIN!{Style.RESET_ALL}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}[!] Interrupted by user{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n{Fore.RED}[X] Critical error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        sys.exit(1)