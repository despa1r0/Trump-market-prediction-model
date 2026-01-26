import pandas as pd
import os
from typing import Tuple

# --- PATH CONFIGURATION / KONFIGURACJA ŚCIEŻEK ---
# Base directory setup based on your architecture / Konfiguracja katalogu bazowego
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(BASE_DIR, "DataPrep and Graphs")
DEFAULT_FILE = "ready_for_ml_training.csv"
DEFAULT_FULL_PATH = os.path.join(DEFAULT_DIR, DEFAULT_FILE)

class DataLoader:
    """Class for robust data loading / Klasa do solidnego ładowania danych."""

    def __init__(self, filepath: str | None = None):
        """
        Args:
            filepath: Path to CSV / Ścieżka do pliku CSV
        """
        if filepath is None:
            self.filepath = DEFAULT_FULL_PATH
        else:
            self.filepath = filepath
        self.df: pd.DataFrame | None = None
        self.X = None
        self.y = None

    def load_data(self) -> pd.DataFrame:
        """Loads data with path validation / Ładuje dane z walidacją ścieżki."""
        print(f"[*] Loading data from: {self.filepath}")

        # File search logic / Logika wyszukiwania pliku
        if os.path.exists(self.filepath):
            load_path = self.filepath
        else:
            # Search near the script / Szukaj obok skryptu
            local_path = os.path.join(os.getcwd(), os.path.basename(self.filepath))
            if os.path.exists(local_path):
                load_path = local_path
            else:
                raise FileNotFoundError(f"[ERROR] File not found: {self.filepath}")

        self.df = pd.read_csv(load_path)
        print(f"[OK] Loaded {len(self.df)} rows.")
        return self.df

    def clean_data(self) -> pd.DataFrame:
        """Data cleaning logic / Logika czyszczenia danych."""
        assert self.df is not None, "Data not loaded / Dane nie zostały załadowane"

        # Remove missing values / Usuń brakujące wartości
        self.df = self.df.dropna(subset=['clean_text_nlp', 'Market_Impact'])
        return self.df

    def create_target(self, threshold=None) -> pd.Series:
        """
        Target creation with ENHANCED type checking /
        Tworzenie celu ze WZMOCNIONĄ weryfikacją typów.
        """
        print("\n[*] Creating labels (Target)...")

        if self.df is None:
             raise ValueError("Data not loaded / Dane nie zostały załadowane")
        
        if 'is_noise' not in self.df.columns:
            raise ValueError("Column 'is_noise' not found in file! / Brak kolumny 'is_noise'!")

        def classify_row(row):
            # Get raw value / Pobierz surową wartość
            raw_val = row['is_noise']

            # --- BULLETPROOF CONVERSION / PANCERNA KONWERSJA ---
            is_noise = False
            try:
                # 1. Try converting to float then int (handles "1.0" and 1.0) /
                # 1. Spróbuj konwertować na float, a potem na int (obsługuje "1.0" i 1.0)
                if int(float(raw_val)) == 1:
                    is_noise = True
            except:
                # 2. If failed, try string comparison / 2. Jeśli nie powiodło się, spróbuj porównania ciągów
                if str(raw_val).lower() in ['true', 'yes', '1']:
                    is_noise = True

            # --- CLASSIFICATION LOGIC / LOGIKA KLASYFIKACJI ---
            if is_noise:
                return 0  # Class 0: NOISE (ignore market) / Klasa 0: SZUM (ignoruj rynek)

            # If not noise, check market reaction / Jeśli nie szum, sprawdź reakcję rynku
            if row['Market_Impact'] > 0:
                return 1  # Rise / Wzrost
            else:
                return -1 # Drop / Spadek

        self.df['target'] = self.df.apply(classify_row, axis=1)

        # Display statistics / Wyświetl statystyki
        counts = self.df['target'].value_counts().sort_index()
        total = len(self.df)
        print(f"[INFO] Final class distribution:")
        print(f"   [-1] Drop  (-1): {counts.get(-1, 0)} ({counts.get(-1, 0)/total*100:.1f}%)")
        print(f"   [0]  Noise  (0): {counts.get(0, 0)} ({counts.get(0, 0)/total*100:.1f}%) <--- SHOULD BE ~67%")
        print(f"   [+1] Rise   (1): {counts.get(1, 0)} ({counts.get(1, 0)/total*100:.1f}%)")

        return self.df['target']

    def prepare_features_and_target(self) -> Tuple[pd.Series, pd.Series]:
        """Returns X and y for training / Zwraca X i y do treningu."""
        assert self.df is not None
        self.X = self.df['clean_text_nlp'].astype(str)
        self.y = self.df['target']
        return self.X, self.y

    def get_full_dataframe(self) -> pd.DataFrame:
        """Returns the full internal dataframe / Zwraca pełną wewnętrzną ramkę danych."""
        assert self.df is not None
        return self.df