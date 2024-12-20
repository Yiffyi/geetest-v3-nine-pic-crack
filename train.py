from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torchvision

from PIL import Image

import os

data_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)), # scale to match resnet input shape
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # normalize to match pretrained weight of IMAGENET1K_V1
        ),
    ]
)

TOTAL_CATEGORIES = 90
BATCH_SIZE = 64
LEARNING_RATE = 0.001

HINTS_DIR = './dataset/hints'
ANNOTED_DIR = './dataset/annoted'
RAW_DIR = './dataset/raw'
UNKNOWN_DIR = './dataset/unknown'
MODEL_DIR = './model'

class MyDataset(ImageFolder):
    def is_valid_image(self, filename: str):
        return os.path.basename(filename).lower().startswith('option')
    
    def find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: int(cls_name) for cls_name in classes}
        return classes, class_to_idx

    def __init__(self, data_dir):
        super().__init__(root=data_dir, transform=data_transform, is_valid_file=self.is_valid_image)
        self.dataset = ImageFolder(root=data_dir, transform=data_transform)

class MyModel(nn.Module):
    def __init__(self, output_size: int):
        super(MyModel, self).__init__()
        self.model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        self.model.fc = nn.Linear(512, output_size)

    def forward(self, x):
        return self.model(x)

def train(target_epochs: int):
    print(f"torch.version.cuda={torch.version.cuda}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using torch.device({str(device)})")

    assert len(os.listdir(f"{ANNOTED_DIR}/")) == TOTAL_CATEGORIES
    model = MyModel(TOTAL_CATEGORIES)
    model.to(device)

    dataset = MyDataset(ANNOTED_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    for epoch in range(target_epochs):
        correct = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}/{target_epochs}, Loss: {loss.item()}")

            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).float().sum()
        
        accuracy = correct / len(dataset)
        # trainset, not train_loader
        # probably x in your case

        print(f"Accuracy = {accuracy*100 : .2f}%")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"{MODEL_DIR}/model_e{epoch}.pth")
    torch.save(model.state_dict(), f"{MODEL_DIR}/model.pth")

def inference(model_path: str, image_dir: str):
    model = MyModel(TOTAL_CATEGORIES)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    for filename in os.listdir(image_dir):
        if not filename.startswith('option'):
            continue
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path)
        image = data_transform(image)
        image = image.unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            predicted = torch.argmax(output, 1)
            print(f"Predicted class for {filename}: {predicted.item()}")

def convert(model_path: str, output_path: str):
    model = MyModel(TOTAL_CATEGORIES)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # dummy_input = torch.randn(1, 3, 224, 224)
    # torch.onnx.export(model, dummy_input, output_path, verbose=True,
    #                   input_names = ['input'], output_names = ['output'],
    #                   dynamic_axes={'input': {0: 'batch_size'}})
    dummy_input = torch.randn(9, 3, 224, 224)
    torch.onnx.export(model, dummy_input, output_path, verbose=True)

if __name__ == "__main__":
    # pass
    train(10)
    # inference(f"{MODEL_DIR}/model.pth", UNKNOWN_DIR)
    # convert(f"{MODEL_DIR}/model_e10.pth", f"{MODEL_DIR}/model_e10.onnx")
