import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from PIL import Image

# ── 경로 설정 ──
BASE_DATA  = r"C:\Users\sj123\Doit_DeepLearning\2025_딥러닝_전선\PPE_Data\processed"
TRAIN_CSV  = os.path.join(BASE_DATA, "train_multilabels.csv")

MODEL_DIR  = r"C:\Users\sj123\Doit_DeepLearning\2025_딥러닝_전선\PPE_Project\models"
os.makedirs(MODEL_DIR, exist_ok=True)
SAVE_PATH  = os.path.join(MODEL_DIR, "cnn_model.pth")

# ── 하이퍼파라미터 ──
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_LABELS= ["Face Mask","Gloves","Helmet","No Gloves","No Helmet","No Mask"]
BATCH_SIZE  = 24
EPOCHS      = 5
LR          = 1e-3

# ── 이미지 전처리 ──
tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# ── Dataset 정의 ──
class PPEDataset(Dataset):
    def __init__(self, csv_path, transform):
        df = pd.read_csv(csv_path)
        self.paths  = df["filename"].tolist()
        self.labels = df.drop(columns="filename").values
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        label = torch.FloatTensor(self.labels[idx])
        return img, label

# ── DataLoader 생성 ──
train_ds     = PPEDataset(TRAIN_CSV, tfm)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# ── 모델 정의 ──
class CNNMulti(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = models.resnet18(pretrained=True)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x):
        return self.base(x)

model     = CNNMulti(len(CLASS_LABELS)).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

# ── 학습 루프 ──
for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_ds)
    print(f"[Epoch {epoch}] Loss: {epoch_loss:.4f}")

# ── 가중치 저장 ──
torch.save(model.state_dict(), SAVE_PATH)
print(f"Saved CNN weights to {SAVE_PATH}")
