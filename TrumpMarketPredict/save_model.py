"""
Interactive Model Trainer - FULL VERSION (Sync with main.py)
Includes Feature Engineering, Training, and Fine-Tuning.
"""
import joblib
import sys
from colorama import init, Fore, Style
from data_loader import DataLoader
from feature_engineering import FeatureEngineer

# Attempt to import trainer
try:
    from model_trainer import ModelTrainer
except ImportError:
    print("⚠️  model_trainer.py not found, make sure the file is in the same directory")
    sys.exit(1)

init(autoreset=True)


class InteractiveTrainer:
    """
    Advanced Trainer: Train -> Fine Tune -> Save
    Fully mirrors the training logic from main.py
    """
    
    def __init__(self):
        self.trainer = None
        self.df_with_features = None
        self.training_columns = None
        
    def print_menu(self, title, options):
        """Print menu"""
        print("\n" + "=" * 70)
        print(f"{Fore.CYAN}{title}{Style.RESET_ALL}")
        print("=" * 70)
        for key, value in options.items():
            print(f"{Fore.YELLOW}{key}{Style.RESET_ALL} - {value}")
        print("=" * 70)
        
    def train_full_pipeline(self):
        """
        Full pipeline: Load -> Features -> Training -> Fine-Tuning
        """
        print(f"\n{Fore.GREEN}{'='*70}")
        print(f"🚀 STARTING FULL TRAINING PIPELINE")
        print(f"{'='*70}{Style.RESET_ALL}\n")
        
        # --- 1. DATA LOADING ---
        print(f"{Fore.CYAN}📂 STEP 1: Loading Data...{Style.RESET_ALL}")
        try:
            loader = DataLoader("ready_for_ml_training.csv")
            loader.load_data()
            loader.clean_data()
            loader.create_target()
        except Exception as e:
            print(f"{Fore.RED}❌ Data loading error: {e}{Style.RESET_ALL}")
            return
        
        # --- 2. FEATURE ENGINEERING ---
        print(f"\n{Fore.CYAN}🔧 STEP 2: Feature Engineering...{Style.RESET_ALL}")
        try:
            engineer = FeatureEngineer(loader.get_full_dataframe())
            self.df_with_features = engineer.create_all_features()
            
            # Save list of columns (important for production)
            self.training_columns = [col for col in self.df_with_features.columns if col != 'target']
            print(f"   ✅ Features created: {len(self.training_columns)}")
        except Exception as e:
            print(f"{Fore.RED}❌ Feature engineering error: {e}{Style.RESET_ALL}")
            return
        
        # --- 3. TRAINING ---
        print(f"\n{Fore.CYAN}🤖 STEP 3: Training Base Models...{Style.RESET_ALL}")
        try:
            self.trainer = ModelTrainer(self.df_with_features, test_size=0.2, random_state=42)
            self.trainer.split_data()
            self.trainer.create_models()
            self.trainer.train_all_models()
        except Exception as e:
            print(f"{Fore.RED}❌ Training error: {e}{Style.RESET_ALL}")
            return

        # --- 4. FINE TUNING (Same as in main.py) ---
        print(f"\n{Fore.CYAN}🎛️ STEP 4: Fine Tuning Best Model...{Style.RESET_ALL}")
        try:
            # Key step that was missing earlier
            self.trainer.fine_tune_best_model()
            print(f"{Fore.GREEN}✅ Fine tuning completed successfully!{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️ Fine tuning warning: {e}{Style.RESET_ALL}")
        
        # --- RESULT ---
        print(f"\n{Fore.GREEN}🎉 PIPELINE COMPLETED!{Style.RESET_ALL}")
        comparison = self.trainer.get_comparison_table()
        print("\n" + str(comparison))
        
    def save_model_to_disk(self):
        """Save selected model"""
        if not self.trainer or not self.trainer.results:
            print(f"{Fore.RED}❌ No trained models! Run Option 1 first.{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.GREEN}{'='*70}")
        print(f"💾 SAVE MODEL TO DISK")
        print(f"{'='*70}{Style.RESET_ALL}\n")
        
        # List of models
        models_list = list(self.trainer.results.keys())
        
        print("Available models (Optimized versions if applicable):\n")
        print(f"{'#':<4} {'Model Name':<30} {'Accuracy':<10} {'F1-Score':<10}")
        print("-" * 60)
        
        for i, model_name in enumerate(models_list, 1):
            result = self.trainer.results[model_name]
            acc = result['accuracy']
            f1 = result['f1_score']
            
            # Highlight best models
            color = Fore.WHITE
            if f1 > 0.8: 
                color = Fore.GREEN 
            
            print(f"{color}{i:<4} {model_name:<30} {acc:.4f}     {f1:.4f}{Style.RESET_ALL}")
        print("-" * 60)
        
        # Selection and saving
        while True:
            try:
                choice = input(
                    f"\n{Fore.CYAN}Enter model number (1-{len(models_list)}) or 'q' to exit: {Style.RESET_ALL}"
                )\
                
                if choice.lower() == 'q':
                    break
                
                idx = int(choice) - 1
                if 0 <= idx < len(models_list):
                    # Get model data
                    chosen_name = models_list[idx]
                    chosen_model = self.trainer.results[chosen_name]['model']
                    
                    # File name
                    default_name = f"model_{chosen_name}.pkl"
                    custom_name = input(
                        f"{Fore.YELLOW}File name [Enter for '{default_name}']: {Style.RESET_ALL}"
                    )
                    filename = custom_name.strip() if custom_name.strip() else default_name
                    if not filename.endswith('.pkl'):
                        filename += '.pkl'
                    
                    # Save dictionary with metadata (important for API/production use)
                    model_data = {
                        'model': chosen_model,
                        'training_columns': self.training_columns,  # Columns list for validation at inference time
                        'model_name': chosen_name,
                        'metrics': {
                            'accuracy': self.trainer.results[chosen_name]['accuracy'],
                            'f1': self.trainer.results[chosen_name]['f1_score']
                        }
                    }
                    
                    joblib.dump(model_data, filename)
                    
                    print(f"\n{Fore.GREEN}✅ Successfully saved: {filename}{Style.RESET_ALL}")
                    break
                else:
                    print(f"{Fore.RED}Invalid number!{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Enter a number!{Style.RESET_ALL}")

    def run(self):
        """Main menu loop"""
        while True:
            options = {
                '1': 'Train & Optimize (Full Pipeline)',
                '2': 'Save Model',
                '3': 'Exit'
            }
            
            self.print_menu("🚀 TRUMP TWEET TRAINER (FULL VERSION)", options)
            
            choice = input(f"\n{Fore.CYAN}Your choice: {Style.RESET_ALL}")
            
            if choice == '1':
                self.train_full_pipeline()
            elif choice == '2':
                self.save_model_to_disk()
            elif choice == '3':
                print(f"\n{Fore.GREEN}👋 Bye!{Style.RESET_ALL}\n")
                break
            else:
                print(f"{Fore.RED}❌ Invalid choice!{Style.RESET_ALL}")


if __name__ == "__main__":
    try:
        trainer = InteractiveTrainer()
        trainer.run()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}⚠️  Stopped by user{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
