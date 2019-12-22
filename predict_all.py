from path import IMAGE_PATH
from torchvision.datasets import ImageFolder

from torchvision import transforms
from model import Model
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
imageSet = ImageFolder(IMAGE_PATH, transform=transform)
data = DataLoader(imageSet, batch_size=64, shuffle=False)
model = Model(data)
correct, Sum, labels = model.predict_all()
