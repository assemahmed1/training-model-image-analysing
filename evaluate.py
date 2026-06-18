import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# ======= CONFIG =======
MODEL_PATH = '/Users/assemahmed/Downloads/severity_model.pt'
CSV_PATH   = '/Users/assemahmed/Downloads/labeled_dataset.csv'

device = "mps" if torch.backends.mps.is_available() else "cpu"

# ======= LOAD MODEL =======
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 1),
    nn.Sigmoid()
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model = model.to(device)

# ======= TRANSFORM =======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ======= DATASET =======
class SkinDataset(Dataset):
    def __init__(self, df, transform):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = row['image_path'].replace(
            '/content/drive/MyDrive/Dermalyze/dataset/images',
            '/Users/assemahmed/Downloads/skin images '
        )
        try:
            img   = Image.open(path).convert("RGB")
            img   = self.transform(img)
            valid = True
        except:
            img   = torch.zeros(3, 224, 224)
            valid = False
        score = torch.tensor(row['severity_score'], dtype=torch.float32)
        return img, score, valid

# ======= LOAD CSV =======
df        = pd.read_csv(CSV_PATH)
df        = df[df['severity_score'] >= 0].reset_index(drop=True)
df_sample = df.sample(n=2000, random_state=42)

loader = DataLoader(SkinDataset(df_sample, transform),
                    batch_size=32, shuffle=False, num_workers=0)

# ======= EVALUATE =======
total_mae = 0
total_mse = 0
count     = 0

with torch.no_grad():
    for imgs, scores, valids in tqdm(loader, desc="Evaluating"):
        imgs   = imgs.to(device)
        scores = scores.to(device)
        preds  = model(imgs).squeeze(1)

        for i in range(len(preds)):
            if valids[i]:
                mae        = abs(preds[i].item() - scores[i].item())
                mse        = (preds[i].item() - scores[i].item()) ** 2
                total_mae += mae
                total_mse += mse
                count     += 1

mae      = total_mae / count
rmse     = (total_mse / count) ** 0.5
accuracy = (1 - mae) * 100

print(f"\n📊 Evaluation on {count} images:")
print(f"   MAE:      {mae:.4f}  → average error ±{mae:.2f}")
print(f"   RMSE:     {rmse:.4f}")
print(f"   Accuracy: {accuracy:.1f}%")