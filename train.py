import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm  # Progress bar
import os

def train_model(model, train_loader, val_loader, device, num_epochs=10):
    """
    Trenuje model z walidacją i zapisem po każdej epoce.

    Args:
        model (nn.Module): Model do trenowania.
        train_loader: DataLoader dla zbioru treningowego.
        val_loader: DataLoader dla zbioru walidacyjnego.
        device: CPU lub GPU.
        num_epochs (int): Liczba epok treningu.
    """

    # Przeniesienie modelu na GPU, jeśli dostępne
    model.to(device)

    # Definiowanie funkcji straty i optymalizatora
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Ścieżka do zapisu modelu
    MODEL_PATH = "models/fish_classifier.pth"
    os.makedirs("models", exist_ok=True)  # Tworzenie folderu jeśli nie istnieje

    # Pętla treningowa
    for epoch in range(num_epochs):
        model.train()  # Tryb trenowania
        running_loss = 0.0  # Resetowanie strat

        # Tworzenie paska postępu
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoka {epoch+1}/{num_epochs}")

        for batch_idx, (images, labels) in progress_bar:
            images, labels = images.to(device), labels.to(device)  # Przeniesienie batcha na CPU/GPU

            optimizer.zero_grad()  # Zerowanie gradientów
            outputs = model(images)  # Przepuszczenie przez sieć
            loss = criterion(outputs, labels)  # Obliczenie straty
            loss.backward()  # Obliczenie gradientów
            optimizer.step()  # Aktualizacja wag modelu

            running_loss += loss.item()  # Sumowanie strat

            # Aktualizacja progress bara co batch
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        print(f"[INFO] Epoka {epoch+1} zakończona, średnia strata: {avg_train_loss:.4f}")

        model.eval()  # Tryb ewaluacji
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        print(f"[INFO] Epoka {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {accuracy:.4f}")

        torch.save(model.state_dict(), MODEL_PATH)
        print(f"[INFO] Model zapisany do {MODEL_PATH}")

    print("[INFO] Trening zakończony!")
