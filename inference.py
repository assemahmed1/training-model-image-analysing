import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ======= CONFIG =======
MODEL_PATH = '/Users/assemahmed/Downloads/severity_model.pt'

# ======= DEVICE =======
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

# ======= FUNCTIONS =======
def get_severity(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        score = model(img).item()
    return round(score, 4)

def calculate_improvement(image_path_1, image_path_2):
    score1 = get_severity(image_path_1)
    score2 = get_severity(image_path_2)

    if score1 == 0:
        improvement = 0
    else:
        improvement = ((score1 - score2) / score1) * 100

    print(f"زيارة 1 → severity: {score1}")
    print(f"زيارة 2 → severity: {score2}")
    print(f"نسبة التحسن: {improvement:.1f}%")
    return improvement

# ======= TEST =======
calculate_improvement(
    '/Users/assemahmed/Downloads/skin images /Acne And Rosacea Photos/acne-infantile-7.jpg',
    '/Users/assemahmed/Downloads/skin images /Acne And Rosacea Photos/acne-infantile-10.jpg'
)