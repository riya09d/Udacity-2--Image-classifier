# Imports
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Load data
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
# Load datasets
image_datasets = {
    'train': datasets.ImageFolder(train_dir, data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, data_transforms['valid']),
    'test': datasets.ImageFolder(test_dir, data_transforms['test'])
}

# Define dataloaders
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
    'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=True),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32, shuffle=True)
}

# Load pre-trained model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.classifier = nn.Sequential(nn.Linear(25088, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),
                                 nn.Linear(1024, 102))

# Train model
def train(model, device, dataloaders, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        running_loss = 0
        running_corrects = 0
        for images, labels in dataloaders['train']:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(image_datasets['train'])
        epoch_acc = running_corrects.double() / len(image_datasets['train'])
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

train(model, device, dataloaders, criterion, optimizer, epochs=5)

# Test model
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for images, labels in dataloaders['test']:
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

accuracy = correct / len(image_datasets['test'])
print(f'Test Acc: {accuracy:.4f}')

# Save model
torch.save(model.state_dict(), 'model.pth')
# Load checkpoint
def load_checkpoint(model_path):
    model.load_state_dict(torch.load(model_path))
    return model

# Process image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    '''
    img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(image)
    return img

def imshow(image, ax=None, title=None):
    """
    Displays a PyTorch tensor as an image.

    Args:
        image (Tensor): The image tensor to display.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. Defaults to None.
        title (str, optional): The title of the plot. Defaults to None.

    Returns:
        matplotlib.axes.Axes: The axes used for plotting.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes it's the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    if title:
        ax.set_title(title)
    return ax


# You can use this function to display images like this:


image = process_image(your_image)
imshow(image, title="1")
plt.show()