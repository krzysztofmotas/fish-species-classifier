import torch
import torch.nn as nn
import torch.nn.functional as F

class FishClassifier(nn.Module):
    def __init__(self, num_classes):
        super(FishClassifier, self).__init__()

        # Warstwy konwolucyjne + MaxPooling
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Dodatkowa warstwa konwolucyjna
        self.pool = nn.MaxPool2d(2, 2)  # Pooling 2x2 zmniejsza rozmiary o połowę

        # Obliczenie wejścia do fc1 dynamicznie
        self.fc1_input_size = self._get_fc_input_size()
        self.fc1 = nn.Linear(self.fc1_input_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def _get_fc_input_size(self):
        """
        Oblicza liczbę cech wejściowych do fc1 na podstawie danych wejściowych.
        """
        with torch.no_grad():
            sample_input = torch.zeros(1, 3, 224, 224)  # Przykładowy obraz
            sample_output = self.pool(F.relu(self.conv1(sample_input)))
            sample_output = self.pool(F.relu(self.conv2(sample_output)))
            sample_output = self.pool(F.relu(self.conv3(sample_output)))  # Nowa warstwa konwolucyjna
            return sample_output.view(1, -1).size(1)  # Pobranie liczby cech

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))  # Dodatkowy pooling przed fc1
        x = x.view(x.size(0), -1)  # Spłaszczenie do 1D
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
