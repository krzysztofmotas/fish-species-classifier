import torch
import torchvision.transforms as transforms
from PIL import Image

def predict_fish(model, image_path, class_to_idx):
    # Ustawienie rozdzielczości wejściowej na 300x300
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Odwrócenie słownika, tak aby indeksy wskazywały nazwy klas
    idx_to_class = {}

    for class_name, class_index in class_to_idx.items():
        idx_to_class[class_index] = class_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ładowanie obrazu i konwersja do RGB
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device) # Dodanie wymiaru batch

    model.eval()

    # Użycie `torch.inference_mode()` dla lepszej wydajności
    with torch.inference_mode():
        output = model(image)
        _, predicted = torch.max(output, 1)

    class_index = predicted.item()
    class_name = idx_to_class.get(class_index, "Nieznana klasa?")

    return class_index, class_name # Zwracanie indeksu i nazwy klasy
