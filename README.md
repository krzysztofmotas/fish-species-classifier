# Rozpoznawanie gatunków ryb

Projekt ten służy do klasyfikacji gatunków ryb na podstawie zdjęć. Aktualnie wykorzystywane obrazy pochodzą z zestawu danych:  
[Fish Species Image Data - Kaggle](https://www.kaggle.com/datasets/sripaadsrinivasan/fish-species-image-data/code).  
Model wykorzystuje sieć neuronową EfficientNet-B0 i został zaimplementowany w języku Python przy użyciu biblioteki PyTorch.

## Wykorzystane biblioteki
- `torch` – główna biblioteka do uczenia maszynowego
- `torchvision` – obsługa transformacji obrazów i modeli wstępnie trenowanych
- `PIL` – ładowanie i przetwarzanie obrazów
- `pandas` – przetwarzanie plików z danymi o gatunkach ryb
- `tqdm` – wyświetlanie paska postępu podczas treningu modelu

## Zawartość plików

### `main.py`
Główny plik programu. Pozwala na:
- Trenowanie modelu na zbiorze danych
- Załadowanie wcześniej zapisanego modelu
- Ewaluację skuteczności modelu
- Klasyfikację ryby na podstawie obrazu podanego przez użytkownika

### `model.py`
Zawiera implementację modelu EfficientNet-B0, który został dostosowany do klasyfikacji gatunków ryb.

### `train.py`
Zawiera funkcję do trenowania modelu. Wykorzystuje automatyczną precyzję mieszaną (AMP) oraz harmonogram zmiany tempa uczenia.

### `evaluate.py`
Zawiera funkcję oceniającą skuteczność modelu na zbiorze testowym.

### `predict_image.py`
Funkcja do przewidywania gatunku ryby na podstawie pojedynczego zdjęcia.

### `dataset.py`
Definiuje klasę `FishDataset`, która ładuje zbiór danych oraz generuje odpowiednie indeksy dla poszczególnych gatunków ryb.

### `match_fish_images.py`
Sprawdza, czy wszystkie obrazy ryb w katalogu pasują do wpisów w pliku indeksowym.

### `gpu_test.py`
Testuje dostępność GPU i sprawdza liczbę rdzeni procesora.

## Uruchamianie projektu

Aby uruchomić model, należy wykonać:

```sh
python main.py
```
