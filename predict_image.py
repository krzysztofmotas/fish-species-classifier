import torch
import torchvision.transforms as transforms
from PIL import Image

def predict_fish(model, image_path):
    # Ustawienie rozdzielczości wejściowej na 300x300
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ładowanie obrazu i konwersja do RGB
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device) # Dodanie wymiaru batch

    model.eval()

    # Użycie `torch.inference_mode()` dla lepszej wydajności
    with torch.inference_mode():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item() # Zwracanie numeru klasy
