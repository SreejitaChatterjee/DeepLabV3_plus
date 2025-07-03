# predict.py
import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deeplabv3_resnet50(num_classes=2)
model.load_state_dict(torch.load("deeplabv3_clouds.pth"))
model.to(device)
model.eval()

image = Image.open("test_sample.png")
image = np.array(image)
image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float()/255.0
image = image.to(device)

with torch.no_grad():
    out = model(image)['out']
    pred = torch.argmax(out.squeeze(), dim=0).cpu().numpy()

plt.imshow(pred, cmap="gray")
plt.title("Predicted TCCs")
plt.show()
