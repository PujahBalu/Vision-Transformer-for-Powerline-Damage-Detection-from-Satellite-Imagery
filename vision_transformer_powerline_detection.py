
# Vision Transformer for Powerline Damage Detection (XView2 Dataset)

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to dataset - should contain 'damage' and 'no_damage' folders
DATASET_PATH = "data/xview2/"
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5

# Feature extractor from HuggingFace
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# Custom Dataset using PIL images and Albumentations
class CustomXView2Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.image_paths = []
        for label in self.classes:
            label_dir = os.path.join(root_dir, label)
            for img in os.listdir(label_dir):
                self.image_paths.append((os.path.join(label_dir, img), self.classes.index(label)))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((224, 224))
        inputs = feature_extractor(images=image, return_tensors="pt")
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'label': torch.tensor(label)
        }

# Load dataset
train_dataset = CustomXView2Dataset(os.path.join(DATASET_PATH, "train"))
val_dataset = CustomXView2Dataset(os.path.join(DATASET_PATH, "val"))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Load ViT model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=2
)
model.to(device)

# Optimizer and loss
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
def train(model, dataloader):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        inputs = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)

        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# Validation loop
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy

# Run training
for epoch in range(EPOCHS):
    train_loss = train(model, train_loader)
    val_accuracy = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")
