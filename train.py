import torch
import torch.optim as optim
import torch.nn as nn
import torch.cuda.amp as amp  # Automatyczna precyzja mieszana dla lepszej wydajności
from tqdm import tqdm  # Pasek postępu

def train_model(model, train_loader, val_loader, num_epochs=15, save_path="efficientnet_b0_fish_classifier.pth"):
    # Sprawdzenie, czy dostępna jest karta graficzna (GPU), jeśli nie, używamy CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Używane urządzenie: {device}")

    # Funkcja błędu - CrossEntropyLoss jest często używana w klasyfikacji
    criterion = nn.CrossEntropyLoss()

    # Optymalizator - Adam to popularny algorytm aktualizacji wag
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Planowanie uczenia - zmniejsza stopniowo szybkość uczenia się
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Skaler AMP (Automatic Mixed Precision), przyspiesza trening na GPU
    scaler = amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()  # Ustawienie modelu w tryb treningowy
        running_loss = 0.0
        correct, total = 0, 0

        print(f"\n[INFO] Epoka {epoch + 1}/{num_epochs}")

        # Pasek postępu dla treningu
        train_bar = tqdm(train_loader, desc=f"Trening {epoch + 1}/{num_epochs}", leave=True)

        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)  # Przeniesienie danych na GPU/CPU

            optimizer.zero_grad()  # Zerowanie gradientów przed kolejną iteracją

            # Obliczenia z użyciem automatycznej precyzji mieszanej
            with amp.autocast():
                outputs = model(images)  # Przepuszczenie obrazów przez sieć neuronową
                loss = criterion(outputs, labels)  # Obliczenie straty (błędu predykcji)

            # Skalowanie gradientów przed propagacją wsteczną
            scaler.scale(loss).backward()
            scaler.step(optimizer)  # Aktualizacja wag
            scaler.update()  # Aktualizacja skali dla AMP

            running_loss += loss.item()  # Sumowanie strat
            _, predicted = torch.max(outputs, 1)  # Pobranie klasy o najwyższym prawdopodobieństwie
            correct += (predicted == labels).sum().item()  # Zliczanie poprawnych klasyfikacji
            total += labels.size(0)  # Liczba wszystkich próbek

            train_bar.set_postfix(loss=f"{loss.item():.4f}")  # Aktualizacja paska postępu

        # Obliczenie średniej straty i dokładności dla treningu
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        model.eval()  # Ustawienie modelu w tryb ewaluacji (wyłącza dropout i batchnorm)
        val_loss = 0.0
        correct, total = 0, 0

        # Pasek postępu dla walidacji
        val_bar = tqdm(val_loader, desc=f"Walidacja {epoch + 1}/{num_epochs}", leave=True)

        with torch.no_grad():  # Podczas walidacji nie zapisujemy gradientów (oszczędność pamięci)
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)

                with amp.autocast():  # AMP również dla walidacji
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Obliczenie średniej straty i dokładności dla walidacji
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        print(
            f"[INFO] Epoka {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.4f}")

        scheduler.step()  # Aktualizacja harmonogramu uczenia się

        # Co 5 epok zapisujemy stan modelu
        if epoch % 5 == 0:
            torch.save(model.state_dict(), save_path)

    print("[INFO] Trening zakończony!")
    torch.save(model.state_dict(), save_path)  # Zapis końcowy modelu
