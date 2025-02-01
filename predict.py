import torch
from PIL import Image
import torchvision.transforms as transforms

def predict_fish(model, image_path, device):
    # Przeniesienie modelu w tryb oceny
    model.eval()

    # Transformacja obrazu (taka sama jak w treningu)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Wczytanie obrazu i przekształcenie do tensorów
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    # Przewidywanie klasy
    output = model(image)
    _, predicted = torch.max(output, 1)

    return predicted.item()
