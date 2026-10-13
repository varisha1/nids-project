import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier


class MLP(nn.Module):
    """d -> 128 -> 64 -> 2 (NN aur ADV_NN dono ka architecture)."""

    def __init__(self, d, hidden=(128, 64), n_classes=2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d, hidden[0]), nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
        )
        self.head = nn.Linear(hidden[1], n_classes)

    def features(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.body(x))


class TransformerNIDS(nn.Module):
    """Har flow ek 128-d token (assumption, paper clear nahi)."""

    def __init__(self, d, d_model=128, layers=3, heads=4, ff=256, dropout=0.2, n_classes=2):
        super().__init__()
        self.proj = nn.Linear(d, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        h = self.proj(x).unsqueeze(1)
        return self.head(self.encoder(h).squeeze(1))


def make_surrogate(d):
    return nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 2))


def make_rf(seed):
    return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)