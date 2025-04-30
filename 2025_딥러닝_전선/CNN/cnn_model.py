import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import f1_score, classification_report

df = pd.read_csv("train/train_multilabels.csv")
df_sample = df.head(100)
df_sample.to_csv("train/train_sample.csv", index=False)

# --------- 클래스 목록 (순서 고정) ---------
CLASS_LABELS = ["Face Mask", "Gloves", "Helmet", "No Gloves", "No Helmet", "No Mask"]

# --------- 커스텀 Dataset ---------
class PPEDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        df = pd.read_csv(csv_file)
        self.image_paths = df['filename'].values
        self.labels = df.drop(columns='filename').values
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = torch.FloatTensor(self.labels[idx])  # [0,1,0,1,...]
        if self.transform:
            image = self.transform(image)
        return image, label

# --------- 이미지 전처리 ---------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --------- 데이터셋 로딩 ---------
csv_path = "C:/Users/sj123/Doit_DeepLearning/2025_딥러닝_전선/train_sample.csv"
dataset = PPEDataset(csv_path, transform=transform)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True) #CPU기반 환경이므로 16~ 실행해보고 최대32

# --------- 모델 정의 (ResNet18 기반) ---------
class CNNMultilabel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.base = models.resnet18(pretrained=True)
        self.base.fc = nn.Linear(self.base.fc.in_features, num_classes)

    def forward(self, x):
        return self.base(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNMultilabel(num_classes=len(CLASS_LABELS)).to(device)

# --------- 손실 함수 / 옵티마이저 ---------
criterion = nn.BCEWithLogitsLoss() # 멀티라벨용 손실 함수
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------- 학습 루프 ---------
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

# --------- 성능 평가 ---------
# model.eval()
# all_labels = []
# all_preds = []
#
# with torch.no_grad():
#     for images, labels in train_loader:
#         images = images.to(device)
#         outputs = model(images)
#         probs = torch.sigmoid(outputs).cpu().numpy()
#         preds = (probs > 0.5).astype(int) # 클래스별 확률을 이진 결과로 변환
#         all_preds.extend(preds)
#         all_labels.extend(labels.numpy())
#
# print("\n[Classification Report]") # F1-score, precision, recall 평가
# print(classification_report(all_labels, all_preds, target_names=CLASS_LABELS))
