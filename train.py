# train.py
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn
import torch.optim as optim

# your dataset (as VOC-like)
dataset = VOCSegmentation(
    root="./deeplab_dataset",
    year="2012",
    image_set="train",
    download=False,
    transforms=transforms.Compose([
        transforms.ToTensor()
    ])
)

train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deeplabv3_resnet50(num_classes=2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(10):
    model.train()
    for images, targets in train_loader:
        images = images.to(device)
        targets = targets["segmentation"].to(device)
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, targets.long())
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} done")

torch.save(model.state_dict(), "deeplabv3_clouds.pth")
