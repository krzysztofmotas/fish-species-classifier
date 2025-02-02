import os
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image

class FishDataset(Dataset):
    """
    Dataset dla klasyfikacji gatunków ryb.
    Ładuje obrazy oraz ich odpowiadające etykiety.
    """

    def __init__(self, images_folder, index_file, transform=None):
        self.images_folder = images_folder
        self.transform = transform

        # Definiujemy kolumny pliku
        columns = ["species_id", "species_name", "image_type", "file_name", "index_number"]

        # Wczytanie danych z pliku
        self.data = pd.read_csv(index_file, sep="=", names=columns)

        # Przekształcenie nazw plików do formatu .png
        self.data["index_number"] = self.data["index_number"].astype(str) + ".png"

        # Filtrujemy tylko pliki obecne w folderze z obrazami
        available_files = set(os.listdir(images_folder))
        self.data = self.data[self.data["index_number"].isin(available_files)]

        # Pobieramy unikalne nazwy gatunków
        unique_species = self.data["species_name"].unique()

        # Tworzymy pusty słownik do mapowania nazw gatunków na indeksy
        self.class_to_idx = {}

        # Przypisujemy każdemu gatunkowi unikalny indeks
        for idx, species in enumerate(unique_species):
            self.class_to_idx[species] = idx

    def __len__(self):
        """Zwraca liczbę próbek w zbiorze danych."""
        return len(self.data)

    def __getitem__(self, idx):
        """Zwraca obraz i jego etykietę dla danego indeksu."""
        row = self.data.iloc[idx]
        image_path = os.path.join(self.images_folder, row["index_number"])
        image = Image.open(image_path).convert("RGB")
        label = self.class_to_idx[row["species_name"]]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(batch_size=32):
    """
    Tworzy DataLoadery dla zbiorów treningowego, walidacyjnego i testowego.
    """

    # Definiujemy transformacje dla obrazów
    transform = transforms.Compose([
        transforms.Resize((300, 300)),  # Zmiana rozmiaru obrazów
        transforms.RandomHorizontalFlip(),  # Losowe odbicie poziome
        transforms.RandomRotation(30),  # Losowy obrót o maksymalnie 30 stopni
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),  # Zmiana kolorystyki
        transforms.ToTensor(),  # Konwersja do tensora
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizacja
    ])

    # Tworzymy instancję zbioru danych
    dataset = FishDataset("fishes/images/numbered", "fishes/final_all_index.txt", transform=transform)
    num_classes = len(dataset.class_to_idx)

    # Podział na zbiór treningowy, walidacyjny i testowy
    train_size = int(0.7 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Tworzenie DataLoaderów
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, num_classes