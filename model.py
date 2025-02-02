import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def get_model(num_classes):
    """
    Tworzy i zwraca model EfficientNet-B0 przystosowany do klasyfikacji obrazów na określoną liczbę klas.
    """
    # Wczytanie modelu EfficientNet-B0 z pretrenowanymi wagami ImageNet
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    # EfficientNet-B0 posiada wbudowaną warstwę klasyfikacyjną model.classifier, która składa się z kilku elementów.
    # Domyślnie ostatnia warstwa to warstwa w pełni połączona (fully connected), dostosowana do 1000 klas z ImageNet.
    # Jednak my chcemy dostosować model do naszej własnej liczby klas (num_classes), dlatego musimy ją zamienić.

    # Pobieramy liczbę wejściowych neuronów z oryginalnej warstwy (czyli tyle, ile neuronów przychodzi z poprzedniej warstwy modelu).
    # Następnie tworzymy nową warstwę w pełni połączoną (Linear), która ma tyle wyjść, ile klas mamy w naszym zadaniu.
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    # Przeniesienie modelu na odpowiednie urządzenie (GPU, jeśli dostępne, inaczej CPU)
    return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

