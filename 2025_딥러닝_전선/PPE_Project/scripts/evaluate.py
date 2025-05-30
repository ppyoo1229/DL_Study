import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import ViTFeatureExtractor, ViTForImageClassification

# ── 설정 ──
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_LABELS = ["Face Mask","Gloves","Helmet","No Gloves","No Helmet","No Mask"]
THRESHOLD = 0.5
BATCH_SIZE = 16

# 실제 프로젝트 경로에 맞게 수정하세요
BASE_DIR    = r"C:\Users\sj123\Doit_DeepLearning\2025_딥러닝_전선\PPE_Project"
CNN_CKPT    = os.path.join(BASE_DIR, "models", "cnn_model.pth")
VIT_DIR     = os.path.join(BASE_DIR, "models", "vit")              # HF 형식 폴더
VALID_CSV   = os.path.join(BASE_DIR, "data", "processed", "valid_multilabels.csv")

# ── 평가용 Dataset ──
class EvalDataset(Dataset):
    def __init__(self, csv_path, feature_extractor=None, use_cnn=True):
        df = pd.read_csv(csv_path)
        self.img_paths = df["filename"].tolist()
        self.labels    = df.drop(columns="filename").values.astype(np.float32)
        self.use_cnn   = use_cnn
        self.fe        = feature_extractor
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.use_cnn:
            return self.transform(img), torch.from_numpy(self.labels[idx])
        else:
            enc = self.fe(images=img, return_tensors="pt")
            item = {k: v.squeeze() for k,v in enc.items()}
            item["labels"] = torch.from_numpy(self.labels[idx])
            return item

# ── CNN 평가 함수 ──
def eval_cnn():
    from scripts.train_cnn import CNNMulti   # cnn_model 클래스 정의된 곳
    ds     = EvalDataset(VALID_CSV, use_cnn=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    model = CNNMulti().to(DEVICE)
    model.load_state_dict(torch.load(CNN_CKPT, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(imgs)
            preds  = (torch.sigmoid(logits) > THRESHOLD).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)

    print("── CNN Evaluation ──")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("F1-score :", f1_score(y_true, y_pred, average="macro"))
    print(classification_report(y_true, y_pred, target_names=CLASS_LABELS))

# ── ViT 평가 함수 ──
def eval_vit():
    fe    = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    model = ViTForImageClassification.from_pretrained(VIT_DIR).to(DEVICE)

    ds     = EvalDataset(VALID_CSV, feature_extractor=fe, use_cnn=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            pixel_vals = batch["pixel_values"].to(DEVICE)
            labels     = batch["labels"].to(DEVICE)
            logits     = model(pixel_vals).logits
            preds      = (torch.sigmoid(logits) > THRESHOLD).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)

    print("── ViT Evaluation ──")
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("F1-score :", f1_score(y_true, y_pred, average="macro"))
    print(classification_report(y_true, y_pred, target_names=CLASS_LABELS))

# ── 메인 ──
if __name__ == "__main__":
    eval_cnn()
    eval_vit()