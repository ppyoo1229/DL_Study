DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNMulti(...).to(DEVICE)
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score

# ── 경로 설정 ──
BASE_DATA = r"C:\PPE_Data\processed"
TRAIN_CSV = os.path.join(BASE_DATA, "train_multilabels.csv")
VALID_CSV = os.path.join(BASE_DATA, "valid_multilabels.csv")
MODEL_DIR = r"C:\PPE_Project\models"
BEST_PATH = os.path.join(MODEL_DIR, "best_cnn.pth")
RESULTS_DIR = os.path.join(BASE_DATA, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
HIST_CSV = os.path.join(RESULTS_DIR, "cnn_epoch15.csv")

# ── 하이퍼파라미터 ──
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_LABELS = ["Face Mask","Gloves","Helmet",
                "No Mask","No Gloves","No Helmet",
                "Goggles","No Goggles","Shoes","No Shoes"]
BATCH_SIZE = 24
EPOCHS = 15
LR = 1e-3
PATIENCE = 3

# ── 전처리 ──
tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ── Dataset & DataLoader ──
class PPEDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform):
        df = pd.read_csv(csv_path)
        self.names = df['filename'].tolist()
        self.labels = df.drop(columns='filename').values
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        fname = self.names[idx]
        path = os.path.join(self.img_dir, fname)
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        label = torch.FloatTensor(self.labels[idx])
        return img, label


# train
IMG_TRAIN = r"C:\PPE_Data\raw\train"
train_ds = PPEDataset(TRAIN_CSV, IMG_TRAIN, tfm)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
# valid
IMG_VALID = r"C:\PPE_Data\raw\valid"
val_ds = PPEDataset(VALID_CSV, IMG_VALID, tfm)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)


# ── 모델 정의 ──
class CNNMulti(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x):
        return self.base(x)


model = CNNMulti(len(CLASS_LABELS)).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

# ── Early Stopping 변수 ──
best_f1 = 0.0
no_improve = 0
history = []

# ── 학습 + 검증 루프 ──
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    # --- Train ---
    model.train()
    train_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_ds)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labels)
            val_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
    val_loss /= len(val_ds)
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)
    val_acc = accuracy_score(y_true, y_pred)
    val_f1 = f1_score(y_true, y_pred, average="macro")

    elapsed = time.time() - t0
    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "time_s": elapsed
    })
    print(f"[Epoch {epoch:2d}] "
          f"train_loss={train_loss:.6f}  "
          f"val_loss={val_loss:.6f}  "
          f"val_acc={val_acc:.6f}  "
          f"val_f1={val_f1:.6f}  "
          f"time={elapsed:.1f}s")

    # --- Early Stopping 체크 ---
    if val_f1 > best_f1:
        best_f1 = val_f1
        no_improve = 0
        torch.save(model.state_dict(), BEST_PATH)
        print(f"  ▶ New best f1: {best_f1:.4f}, model saved.")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"  ▶ No improvement for {PATIENCE} epochs. Stopping.")
            break

# ── 최종 베스트 모델 로드 ──
model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE))

# ── 기록 CSV 저장 ──
pd.DataFrame(history).to_csv(HIST_CSV, index=False)
print(f"Saved history to {HIST_CSV}")
print(f"Best model saved to {BEST_PATH} at epoch with f1={best_f1:.4f}")
