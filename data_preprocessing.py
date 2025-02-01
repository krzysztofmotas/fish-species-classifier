import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

class FishDataset(Dataset):
    def __init__(self, images_folder, index_file, transform=None):
        """
        Niestandardowy Dataset dla obrazów ryb.

        Args:
            images_folder (str): Ścieżka do katalogu z obrazami.
            index_file (str): Ścieżka do pliku final_all_index.txt z etykietami.
            transform (callable, optional): Transformacje dla obrazów.
        """
        self.images_folder = images_folder
        self.transform = transform

        # Wczytanie pliku indexowego z informacjami o etykietach
        columns = ["species_id", "species_name", "image_type", "file_name", "index_number"]
        self.data = pd.read_csv(index_file, sep="=", names=columns)

        # Konwersja index_number na string i dodanie rozszerzenia .png
        self.data["index_number"] = self.data["index_number"].astype(str) + ".png"

        # Lista dostępnych plików w katalogu numbered
        available_files = set(os.listdir(images_folder))
        self.data = self.data[self.data["index_number"].isin(available_files)]  # Usunięcie brakujących plików

        # Tworzenie mapowania species_name -> indeks klasy
        self.class_to_idx = {species: idx for idx, species in enumerate(self.data["species_name"].unique())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Pobiera obraz i jego etykietę.

        Args:
            idx (int): Indeks próbki.

        Returns:
            Tensor obrazu, etykieta (int)
        """
        row = self.data.iloc[idx]
        image_path = os.path.join(self.images_folder, row["index_number"])
        image = Image.open(image_path).convert("RGB")  # Wczytanie obrazu jako RGB
        label = self.class_to_idx[row["species_name"]]  # Pobranie indeksu klasy

        if self.transform:
            image = self.transform(image)

        return image, label

def load_data(batch_size=32):
    """
    Wczytuje zbiór danych i dzieli go na zestawy treningowy, walidacyjny i testowy.

    Args:
        batch_size (int): Rozmiar batcha.

    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Skalowanie do 224x224
        transforms.RandomHorizontalFlip(),  # Odbicie lustrzane
        transforms.RandomRotation(10),  # Obrót o losowe 10 stopni
        transforms.ToTensor(),  # Konwersja do tensora
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizacja pikseli do [-1,1]
    ])

    dataset = FishDataset(images_folder="fishes/images/numbered", index_file="fishes/final_all_index.txt", transform=transform)
    num_classes = len(dataset.class_to_idx)  # Pobranie liczby klas

    train_size = int(0.7 * len(dataset))  # 70% trening
    val_size = int(0.2 * len(dataset))  # 20% walidacja
    test_size = len(dataset) - train_size - val_size  # 10% testy

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, num_classes
