"""
Main Pipeline - Main script for running the entire project
"""
import sys
from colorama import init, Fore, Style
from data_loader import DataLoader
from exploratory_analysis import ExploratoryAnalysis
from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator, MultiModelComparison


init(autoreset=True)



def print_header(text: str):
    """Beautiful header"""
    print("\n" + "=" * 70)
    print(f"{Fore.CYAN}{text}{Style.RESET_ALL}")
    print("=" * 70 + "\n")



def main():
    """Main pipeline function"""
    
    print_header("🚀 TRUMP TWEET MARKET IMPACT PREDICTION")
    print("Project: Predicting tweet impact on financial markets")
    print("Author: [Your name]")
    print("Date: 2025-01-06\n")
    
    # ===== STEP 1: DATA LOADING =====
    print_header("📂 STEP 1: DATA LOADING AND CLEANING")
    
    try:
        loader = DataLoader("ready_for_ml_training.csv")
        loader.load_data()
        loader.clean_data()
        loader.create_target(threshold=0.005)
    except Exception as e:
        print(f"{Fore.RED}❌ Data loading error: {e}{Style.RESET_ALL}")
        sys.exit(1)
    
    # ===== STEP 2: DATA ANALYSIS =====
    print_header("🔍 STEP 2: EXPLORATORY DATA ANALYSIS")
    
    try:
        analyzer = ExploratoryAnalysis(loader.get_full_dataframe())
        analyzer.generate_full_report()
    except Exception as e:
        print(f"{Fore.YELLOW}⚠️ Analysis error: {e}{Style.RESET_ALL}")
    
    # ===== STEP 3: FEATURE ENGINEERING =====
    print_header("🔧 STEP 3: FEATURE ENGINEERING")
    
    try:
        engineer = FeatureEngineer(loader.get_full_dataframe())
        df_with_features = engineer.create_all_features()
        
        # Display created features
        features = engineer.get_feature_names()
        print(f"\n✅ Created {len(features)} new features:")
        for feat in features[:10]:  # Show first 10
            print(f"   • {feat}")
        if len(features) > 10:
            print(f"   ... and {len(features) - 10} more features")
            
    except Exception as e:
        print(f"{Fore.YELLOW}⚠️ Feature engineering error: {e}{Style.RESET_ALL}")
    
    # ===== STEP 4: PREPARE DATA FOR TRAINING =====
    print_header("✂️ STEP 4: TRAIN/TEST PREPARATION")
    
    X, y = loader.prepare_features_and_target()
    
    # ===== STEP 5: MODEL TRAINING =====
    print_header("🤖 STEP 5: MODEL TRAINING")
    
    try:
        trainer = ModelTrainer(X, y, test_size=0.2, random_state=42)
        trainer.split_data()
        trainer.create_models()
        trainer.train_all_models()
        
        print(f"\n{Fore.GREEN}✅ Trained {len(trainer.results)} models{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"{Fore.RED}❌ Training error: {e}{Style.RESET_ALL}")
        sys.exit(1)
    
    # ===== STEP 6: FINE TUNING =====
    print_header("🔧 STEP 6: FINE TUNING BEST MODEL")
    
    try:
        trainer.fine_tune_best_model()
    except Exception as e:
        print(f"{Fore.YELLOW}⚠️ Fine tuning error: {e}{Style.RESET_ALL}")
    
    # ===== STEP 7: MODEL EVALUATION =====
    print_header("📊 STEP 7: MODEL EVALUATION AND COMPARISON")
    
    try:
        # Detailed evaluation of each model
        for model_name, result in trainer.results.items():
            evaluator = ModelEvaluator(
                trainer.y_test, 
                result['predictions'], 
                model_name
            )
            evaluator.full_evaluation()
        
        # Compare all models
        print_header("🏆 FINAL MODEL COMPARISON")
        comparator = MultiModelComparison(trainer.results, trainer.y_test)
        comparison_df = comparator.compare_all_models()
        
        # Save best model
        trainer.save_best_model('best_model.pkl')
        
    except Exception as e:
        print(f"{Fore.RED}❌ Evaluation error: {e}{Style.RESET_ALL}")
        sys.exit(1)
    
    # ===== FINALE =====
    print_header("🎉 PROJECT COMPLETED")
    
    print(f"{Fore.GREEN}✅ All steps completed successfully!{Style.RESET_ALL}\n")
    print("📁 Generated files:")
    print("   • correlation_matrix.png")
    print("   • market_impact_distribution.png")
    print("   • confusion_matrix_*.png (for each model)")
    print("   • model_comparison.png")
    print("   • best_model.pkl")
    
    print(f"\n{Fore.CYAN}📊 Best model:{Style.RESET_ALL}")
    best_model = comparison_df.iloc[0]
    print(f"   Name:      {best_model['Model']}")
    print(f"   Accuracy:  {best_model['Accuracy']:.4f}")
    print(f"   F1-Score:  {best_model['F1-Score']:.4f}")
    
    print(f"\n{Fore.YELLOW}💡 Next steps:{Style.RESET_ALL}")
    print("   1. Review generated charts")
    print("   2. Study confusion matrices to understand errors")
    print("   3. If needed, run tests: python test_models.py")
    print("   4. Use best_model.pkl for predictions")
    
    print("\n" + "=" * 70)
    print(f"{Fore.GREEN}🎯 DONE! MAKE MODELS GREAT AGAIN! 🇺🇸{Style.RESET_ALL}")
    print("=" * 70 + "\n")



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}⚠️ Interrupted by user{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n{Fore.RED}❌ Critical error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
