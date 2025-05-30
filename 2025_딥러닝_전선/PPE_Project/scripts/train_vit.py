import torch
import transfomers
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import ViTFeatureExtractor, ViTForImageClassification, TrainingArguments, Trainer, default_data_collator
import numpy as np

# 클래스·레이블
CLASS_LABELS = ["Face Mask","Gloves","Helmet","No Gloves","No Helmet","No Mask"]
label2id = {l:i for i,l in enumerate(CLASS_LABELS)}
id2label = {i:l for l,i in label2id.items()}

# FE & 모델
fe = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    problem_type="multi_label_classification",
    num_labels=len(CLASS_LABELS),
    label2id=label2id, id2label=id2label
)

# Dataset
class PPEHFDataset(Dataset):
    def __init__(self, csv, fe):
        df = pd.read_csv(csv)
        self.imgs = df['filename'].tolist()
        self.labels = df.drop(columns='filename').values.astype(np.float32)
        self.fe = fe
    def __len__(self): return len(self.imgs)
    def __getitem__(self,i):
        img = Image.open(self.imgs[i]).convert("RGB")
        enc = self.fe(images=img, return_tensors="pt")
        item = {k:v.squeeze() for k,v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item

ds = PPEHFDataset("2025_딥러닝_전선/PPE_Data/processed/train_multilabels.csv", fe)
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="/content/models/vit",
        per_device_train_batch_size=16,
        num_train_epochs=5,
        learning_rate=5e-5,
        logging_steps=10,
        save_strategy="no"
    ),
    train_dataset=ds,
    data_collator=default_data_collator,
    tokenizer=fe
)

# 학습
trainer.train()
trainer.save_model("/content/models/vit")
