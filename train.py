import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os

class CustomDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgs = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])
        assert len(self.imgs) == len(self.masks), "Images and masks count mismatch"
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        img = img.resize((512,512))
        mask = mask.resize((512,512), resample=Image.NEAREST)

        mask = np.array(mask).astype(np.uint8)

        # remap 255 to 0 (background)
        mask[mask == 255] = 0

        if self.transform:
            img = self.transform(img)

        mask = torch.from_numpy(mask).long()  # shape [H,W]
        return img, mask

# directories
IMG_DIR = "./dataset/images"
MASK_DIR = "./dataset/annotations"

# transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# dataset & loader
dataset = CustomDataset(IMG_DIR, MASK_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# model
model = deeplabv3_resnet50(pretrained=False, num_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# training loop
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for batch_idx, (imgs, masks) in enumerate(dataloader):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        print(f"Batch {batch_idx+1}/{len(dataloader)} Loss: {loss.item():.4f}")

    print(f"Epoch [{epoch+1}/{EPOCHS}] Average Loss: {total_loss/len(dataloader):.4f}")
