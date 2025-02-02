import torch
from model import get_model
from train import train_model
from evaluate import evaluate_model
from dataset import get_dataloaders, FishDataset
from predict_image import predict_fish

# Włączenie optymalizacji CUDA dla szybszego działania na GPU
torch.backends.cudnn.benchmark = True

# Pobieranie danych (ładowanie datasetu i liczby klas)
train_loader, val_loader, test_loader, num_classes = get_dataloaders()

if __name__ == '__main__':
    if input("Czy chcesz załadować zapisany model? (tak/nie): ").strip().lower() == "tak":
        model = get_model(num_classes)  # Tworzenie modelu EfficientNet-B0
        model.load_state_dict(torch.load("efficientnet_b0_fish_classifier.pth", weights_only=True))  # Wczytanie wag
        print("[INFO] Model załadowany.")
    else:
        model = get_model(num_classes)  # Tworzenie nowego modelu
        train_model(model, train_loader, val_loader)  # Trening modelu

    if input("Czy chcesz sprawdzić dokładność modelu? (tak/nie): ").strip().lower() == "tak":
        evaluate_model(model, test_loader)  # Ewaluacja modelu

    # Pytanie użytkownika o obraz do klasyfikacji
    while True:
        image_path = input("Podaj ścieżkę do obrazu (lub wpisz 'exit' aby zakończyć): ").strip()
        if image_path.lower() == "exit":
            print("Zakończono działanie programu.")
            break

        dataset = FishDataset("fishes/images/numbered", "fishes/final_all_index.txt")
        class_to_idx = dataset.class_to_idx

        index, name = predict_fish(model, image_path, class_to_idx)
        print(f"[INFO] Przewidywana klasa: {index} - {name}")


