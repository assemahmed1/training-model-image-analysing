import ssl
ssl._create_default_https_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
ssl._create_default_https_context.check_hostname = False
ssl._create_default_https_context.verify_mode = ssl.CERT_NONE

import os, torch, pandas as pd
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ======= CONFIG =======
CSV_PATH    = '/Users/assemahmed/Downloads/labeled_dataset.csv'
MODEL_PATH  = '/Users/assemahmed/Downloads/severity_model.pt'
BATCH_SIZE  = 32
EPOCHS      = 10
LR          = 1e-4
NUM_WORKERS = 0

# ======= DEVICE =======
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"✅ Device: {device}")

# ======= LOAD CSV =======
df = pd.read_csv(CSV_PATH)
df = df[df['severity_score'] >= 0].reset_index(drop=True)
print(f"✅ صور: {len(df):,}")

train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
print(f"   Train: {len(train_df):,}  |  Val: {len(val_df):,}")

# ======= DATASET =======
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

class SkinDataset(Dataset):
    def __init__(self, df, transform):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row['image_path']).convert("RGB")
            img = self.transform(img)
        except:
            img = torch.zeros(3, 224, 224)
        score = torch.tensor(row['severity_score'], dtype=torch.float32)
        return img, score

train_loader = DataLoader(SkinDataset(train_df, transform_train),
                          batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS)
val_loader   = DataLoader(SkinDataset(val_df, transform_val),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS)

# ======= MODEL =======
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 1),
    nn.Sigmoid()
)

state_dict = torch.load('/Users/assemahmed/Downloads/resnet18-f37072fd.pth',
                        map_location=device)
model.load_state_dict(state_dict, strict=False)
print(f"✅ ResNet18 اتحمل من الملف")

model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.MSELoss()

# ======= TRAIN =======
best_val_loss = float('inf')
print("\n🚀 التدريب بدأ...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0
    for imgs, scores in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
        imgs   = imgs.to(device)
        scores = scores.to(device).unsqueeze(1)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, scores)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, scores in val_loader:
            imgs   = imgs.to(device)
            scores = scores.to(device).unsqueeze(1)
            preds  = model(imgs)
            val_loss += criterion(preds, scores).item()

    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)
    scheduler.step()

    print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"   💾 Model اتحفظ (val_loss={val_loss:.4f})")

print(f"\n✅ خلص! Model في: {MODEL_PATH}")