import torch

def evaluate_model(model, test_loader, device):
    # Przeniesienie modelu w tryb oceny
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():  # Wyłączamy obliczanie gradientów
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)  # Pobranie klasy z najwyższym wynikiem
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Dokładność na zbiorze testowym: {100 * correct / total:.2f}%")
