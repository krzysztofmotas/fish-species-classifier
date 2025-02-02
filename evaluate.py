import torch

def evaluate_model(model, test_loader):
    """
    Funkcja do oceny skuteczności modelu na zbiorze testowym.
    """
    # Ustawienie urządzenia na GPU, jeśli jest dostępne, w przeciwnym razie CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Przełączenie modelu w tryb ewaluacji (wyłącza dropout, batch normalization itp.)
    model.eval()

    # Inicjalizacja liczników poprawnych predykcji i całkowitej liczby próbek
    correct, total = 0, 0

    # Wyłączenie obliczania gradientów dla oszczędności pamięci i przyspieszenia obliczeń
    with torch.no_grad():
        # Iteracja po zbiorze testowym
        for images, labels in test_loader:
            # Przeniesienie danych wejściowych i etykiet na to samo urządzenie co model (CPU/GPU)
            images, labels = images.to(device), labels.to(device)

            # Przekazanie obrazów do modelu i uzyskanie wyników
            outputs = model(images)

            # Pobranie indeksu klasy z największą wartością predykcji (najbardziej prawdopodobna klasa)
            _, predicted = torch.max(outputs, 1)

            # Zliczanie poprawnych predykcji
            correct += (predicted == labels).sum().item()

            # Zwiększanie całkowitej liczby przykładów testowych
            total += labels.size(0)

    # Obliczenie i wyświetlenie dokładności modelu na zbiorze testowym
    print(f"Dokładność na zbiorze testowym: {100 * correct / total:.2f}%")

