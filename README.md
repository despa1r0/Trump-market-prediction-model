Trump Tweet Market Predictor
Analiza wpływu tweetów Donalda Trumpa na rynek S&P 500
1. Wprowadzenie
Celem projektu Trump Tweet Market Predictor jest zbadanie i modelowanie zależności pomiędzy treścią tweetów Donalda Trumpa a krótkoterminowymi reakcjami rynku finansowego, reprezentowanego przez indeks S&P 500 (ETF SPY).
System automatycznie:
pobiera i czyści tweety,
synchronizuje je z danymi rynkowymi,
klasyfikuje ich treść tematycznie,
mierzy rzeczywisty wpływ na rynek,
buduje modele uczenia maszynowego przewidujące kierunek ruchu rynku:
spadek (-1)
szum (0)
wzrost (1)
Całość została zaprojektowana jako modularny ekosystem składający się z 8 współpracujących komponentów.

2. Architektura systemu
Projekt jest podzielony na dwa główne etapy:
ETAP 0 – Budowa zestawu danych (DataPrep and Graphs)
Zanim możliwe jest trenowanie modeli ML, surowe tweety muszą zostać przekształcone w uporządkowany zbiór danych.
ETAP 1 – Uczenie maszynowe i analiza predykcyjna
Gotowy zestaw danych trafia do pipeline’u ML, gdzie powstają cechy, modele i raporty ewaluacyjne.

3. ETAP 0 – Przygotowanie danych
3.1. tweets_output.csv – Dane źródłowe
Plik wejściowy zawiera surowe tweety Donalda Trumpa.
Format:
Separator: |
Kodowanie: UTF-8
Kolumny:
datetime – znacznik czasu
tweet_text – treść tweeta
Zawartość obejmuje:
oryginalne tweety,
retweety,
linki,
znaki specjalne,
treści polityczne, gospodarcze i prywatne.

3.2. data_pipeline.py – Pipeline przetwarzania danych
Klasa TrumpMarketPipeline odpowiada za pełne przygotowanie danych do uczenia maszynowego.
Główne założenia:
Ticker rynku: SPY
Data początkowa: 2025-01-01
Źródło danych rynkowych: yfinance
Interwał: 1 godzina

3.3. Logika klasyfikacji tweetów
Poziom 1 – Filtr szumu (NOISE_TRIGGERS)
Jeżeli tweet zawiera frazy typu:
happy birthday, congratulations, thank you, rally, vote for, merry christmas
jest automatycznie oznaczany jako szum (Noise).
Poziom 2 – Klasyfikacja tematyczna
Tweet otrzymuje jedną lub kilka kategorii, jeśli zawiera słowa kluczowe:
Kategoria
Przykładowe słowa kluczowe
ECONOMY
inflation, economy, tax, tariff, fed, market
GEOPOLITICS
war, china, russia, ukraine, nato
CRYPTO
bitcoin, crypto, blockchain
DOMESTIC_POLICY
law, senate, house, bill, supreme court

Jeśli nie pasuje do żadnej kategorii → OTHER + szum.

3.4. Synchronizacja z rynkiem
Dla każdego tweeta:
wyszukiwany jest najbliższy przyszły punkt notowań,
obliczana jest procentowa zmiana ceny,
jeśli przerwa przekracza 4 dni (weekendy, święta) → tweet jest ignorowany.

3.5. Finalny zestaw danych
Wynikiem ETAPU 0 jest plik:
ready_for_ml_training.csv
Zawiera on m.in.:
datetime
tweet_text
clean_text_nlp
categories
is_noise
Market_Impact

4. Wizualizacja danych – graph.py
Moduł generuje trzy główne wykresy:
1. Market Match
Pokazuje:
linię ceny S&P 500,
punkty dużych wzrostów i spadków,
pionowe linie w momentach tweetów.
2. Topics Ranking
Ranking kategorii tweetów o największym wpływie rynkowym.
3. Extended Correlation Matrix
Mapa korelacji pomiędzy:
kategoriami tweetów,
cechami szumu,
faktycznymi ruchami rynku.

5. ETAP 1 – Pipeline uczenia maszynowego
5.1. data_loader
ładuje dane,
czyści braki,
tworzy zmienną docelową:
-1 – spadek rynku
0 – szum
1 – wzrost rynku

5.2. feature_engineer
Tworzy ~15 dodatkowych cech:
długość tekstu
liczba słów
sentyment (TextBlob)
liczba słów kluczowych
emocjonalność (WIELKIE LITERY, !!!)
cechy czasowe (godzina, dzień tygodnia)
binarne kategorie tematyczne

5.3. model_trainer
Trenuje trzy modele:
Logistic Regression
Random Forest
Gradient Boosting
Każdy model działa w pipeline:
TF-IDF Vectorizer + Klasyfikator
Modele są oceniane przy pomocy:
accuracy
precision
recall
F1-score
Najlepszy model trafia do GridSearchCV w celu dalszej optymalizacji.

5.4. model_evaluator
Dla każdego modelu:
generowany jest raport klasyfikacji,
rysowana jest macierz pomyłek,
analizowane są błędy predykcji.



6. Główne skrypty sterujące projektem
Oprócz modułów przetwarzania danych i uczenia maszynowego projekt zawiera trzy kluczowe pliki sterujące, które odpowiadają za uruchamianie, testowanie oraz zapisywanie modeli.

main.py — Główny punkt wejścia (pełny pipeline)
Plik main.py jest głównym skryptem uruchamiającym cały proces end-to-end – od surowych danych aż po porównanie modeli.
Co robi krok po kroku:
Ładowanie danych
inicjalizuje DataLoader,
ładuje ready_for_ml_training.csv,
czyści dane,
tworzy zmienną docelową (klasy: -1, 0, 1).
Inżynieria cech
uruchamia FeatureEngineer,
generuje cechy tekstowe, sentyment, cechy czasowe i binarne kategorie,
tworzy rozszerzony DataFrame.
Analiza eksploracyjna (opcjonalna)
generuje wykresy rozkładów klas, długości tweetów, chmury słów,
zapisuje raport do folderu exploratory_plots/.
Trenowanie modeli
dzieli dane na train/test (stratyfikacja),
tworzy pipeline’y modeli (TF-IDF + klasyfikator),
trenuje wszystkie modele bazowe,
zapisuje metryki jakości.
Dostrajanie najlepszego modelu
automatycznie wybiera model z najwyższym F1-score,
uruchamia GridSearchCV,
zapisuje zoptymalizowany wariant.
Szczegółowa ewaluacja
generuje raporty klasyfikacji,
rysuje macierze pomyłek,
analizuje błędy predykcji.
Porównanie modeli
tworzy tabelę porównawczą,
generuje wykres model_comparison.png.
main.py jest używany, gdy chcesz jednorazowo uruchomić cały proces treningowy bez interakcji z użytkownikiem.

save_model.py — Tryb interaktywny i zapis modeli
Plik save_model.py udostępnia interaktywny interfejs tekstowy do trenowania i zapisywania modeli.
Główne funkcje:
wyświetla menu w konsoli:
Train & Optimize
uruchamia pełny pipeline (tak jak main.py),
pokazuje postęp krok po kroku z kolorowymi komunikatami,
automatycznie wybiera i optymalizuje najlepszy model.
Save Model
wyświetla listę wszystkich wytrenowanych modeli,
pokazuje ich metryki (accuracy, F1-score),
podświetla najlepszy model na zielono,
pozwala użytkownikowi wybrać model do zapisania.
zapisuje wybrany model do pliku .pkl, zawierającego:
pipeline (TF-IDF + klasyfikator),
listę użytych kolumn,
nazwę modelu,
metryki jakości.
save_model.py jest używany, gdy chcesz ręcznie zdecydować, który model zachować do dalszych eksperymentów lub wdrożenia.

test_models.py — Testy jednostkowe systemu
Plik test_models.py odpowiada za automatyczne testowanie wszystkich kluczowych komponentów projektu.
Co jest testowane:
DataLoader
poprawność ładowania CSV,
czyszczenie pustych wartości,
tworzenie zmiennej docelowej (3 klasy),
przygotowanie danych wejściowych X i y.
FeatureEngineer
obliczanie długości tekstu,
liczenie słów kluczowych,
analiza wielkich liter,
liczenie interpunkcji,
tworzenie cech czasowych.
ModelTrainer
poprawność podziału danych,
tworzenie pipeline’ów modeli,
proces trenowania i obliczania metryk.
Predykcje modeli
format wyjścia (-1, 0, 1),
zgodność liczby predykcji z liczbą próbek testowych.
Uruchamianie testów:
python test_models.py

Wynik w konsoli pokazuje:
 liczbę zaliczonych testów,
 liczbę niepowodzeń,
 liczbę błędów.
test_models.py zapewnia stabilność projektu i chroni przed regresją przy dalszym rozwoju systemu.



7. Gdzie program zapisuje wyniki?
Po zakończeniu działania pipeline’u wszystkie wygenerowane pliki są automatycznie zapisywane w uporządkowanej strukturze katalogów, co ułatwia późniejszą analizę i prezentację wyników.
Folder główny wyników
DataPrep and Graphs/analytics/
To centralne miejsce, w którym trafiają wszystkie wykresy, raporty i artefakty modeli.

 market_insights/ — Analizy ogólne danych
W tym folderze znajdują się wykresy opisujące zależności między tweetami a rynkiem:
correlation_matrix.png
Mapa cieplna pokazująca, które cechy (tematy tweetów, szum, sentyment itp.) mają największy wpływ na ruchy rynku.
market_impact_distribution.png
Histogram rozkładu wpływu tweetów na rynek – pokazuje, jak często występują duże wzrosty i spadki cen.

confusion_matrix/ — Skuteczność modeli
Tutaj zapisywane są raporty jakości predykcji:
confusion_matrix_*.png
Macierze pomyłek dla każdego modelu.
Pokazują:
ile razy model trafił poprawnie,
ile razy pomylił klasy (Drop / Noise / Rise).
model_comparison.png
Duży wykres porównujący wszystkie modele pod względem:
accuracy
precision
recall
F1-score
Dzięki niemu widać na pierwszy rzut oka, który model radzi sobie najlepiej.

Główny wynik projektu
best_model.pkl
Plik binarny zawierający najlepszy, zoptymalizowany model.
Zawiera:
wytrenowany pipeline (TF-IDF + klasyfikator),
listę użytych cech,
nazwę modelu,
metryki jakości.
Ten plik jest używany później do:
przewidywania wpływu nowych tweetów,
wdrażania modelu w aplikacji lub API,
dalszych eksperymentów i fine-tuningu.


8. Tryby uruchamiania projektu
# Pełny pipeline
python main.py

# Tryb interaktywny (trening + zapis modelu)
python save_model.py

# Testy jednostkowe
python test_models.py


9. Podsumowanie
Projekt Trump Tweet Market Predictor stanowi kompletny, end-to-end system do:
przetwarzania danych tekstowych,
integracji z danymi rynkowymi,
analizy wpływu wydarzeń politycznych na giełdę,
trenowania modeli predykcyjnych,
wizualizacji wyników.
Architektura modułowa umożliwia:
łatwe rozszerzanie systemu,
podmianę źródeł danych,
eksperymentowanie z nowymi modelami NLP.





