import os
import torch
from data_preprocessing import load_data
from model import FishClassifier
from train import train_model
from evaluate import evaluate_model
from predict import predict_fish

MODEL_PATH = "models/fish_classifier.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DEBUG] Uruchamianie na urządzeniu: {device}")

print("[DEBUG] Wczytywanie zbioru danych...")
train_loader, val_loader, test_loader, num_classes = load_data()
print(f"[DEBUG] Liczba klas w zbiorze: {num_classes}")

model_exists = os.path.exists(MODEL_PATH)

if model_exists:
    print(f"[INFO] Znaleziono zapisany model: {MODEL_PATH}")

    decision = input("[?] Czy chcesz ponownie trenować model? (tak/nie): ").strip().lower()

    if decision == "tak":
        print("[INFO] Rozpoczynam ponowne trenowanie...")
    elif decision == "nie":
        print("[INFO] Wczytywanie zapisanego modelu...")
        model = FishClassifier(num_classes).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model.eval()
        print("[INFO] Model został wczytany i gotowy do użycia!")
    else:
        print("[ERROR] Nieznana opcja. Wczytywanie modelu...")
        model = FishClassifier(num_classes).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model.eval()

else:
    print("[INFO] Nie znaleziono modelu. Rozpoczynam trenowanie od zera...")
    model = FishClassifier(num_classes).to(device)
    train_model(model, train_loader, val_loader, device)

    print(f"[INFO] Zapisywanie modelu do {MODEL_PATH}...")
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print("[INFO] Model zapisany pomyślnie!")

print("[DEBUG] Rozpoczęcie ewaluacji modelu...")
evaluate_model(model, test_loader, device)
print("[DEBUG] Ewaluacja zakończona.")

image_path = "auxis_rochei.jpg"
print(f"[DEBUG] Testowanie modelu na nowym obrazie: {image_path}")
predicted_class = predict_fish(model, image_path, device)
print(f"[DEBUG] Przewidywana klasa: {predicted_class}")
