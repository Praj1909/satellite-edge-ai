"""SatNet — Lightweight CNN for EuroSAT (64x64, 10 classes, ~1.2M params)."""
import os, time
import torch, torch.nn as nn, torch.nn.functional as F

NUM_CLASSES = 10
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "models", "satnet.pth")

CLASS_NAMES = [
    "AnnualCrop","Forest","HerbaceousVeg","Highway",
    "Industrial","Pasture","PermanentCrop","Residential",
    "River","SeaLake"
]

class ConvBlock(nn.Module):
    def __init__(self, ci, co, pool=True):
        super().__init__()
        layers = [nn.Conv2d(ci,co,3,padding=1,bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                  nn.Conv2d(co,co,3,padding=1,bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True)]
        if pool: layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class SatNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.enc1 = ConvBlock(3,  32, pool=True)
        self.enc2 = ConvBlock(32, 64, pool=True)
        self.enc3 = ConvBlock(64,128, pool=True)
        self.enc4 = ConvBlock(128,256,pool=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.4)
        self.fc1  = nn.Linear(256,128)
        self.fc2  = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.enc1(x); x = self.enc2(x)
        x = self.enc3(x); x = self.enc4(x)
        x = self.pool(x).flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

    def predict_single(self, img_tensor):
        self.eval()
        with torch.no_grad():
            x = img_tensor.unsqueeze(0)
            t0 = time.perf_counter()
            logits = self(x)
            ms = (time.perf_counter()-t0)*1000
            probs = F.softmax(logits,dim=1)[0]
            idx = probs.argmax().item()
        return idx, float(probs[idx]), ms, probs.numpy()

    def count_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

def load_model(path=MODEL_PATH, device="cpu"):
    m = SatNet(); m.load_state_dict(torch.load(path, map_location=device)); m.eval(); return m

def save_model(model, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
