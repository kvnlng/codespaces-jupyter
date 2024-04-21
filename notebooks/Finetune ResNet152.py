# Databricks notebook source
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

import torch
from torch import nn
from torchinfo import summary
from torch.optim.lr_scheduler import ExponentialLR
# from torchvision import transforms
from torchinfo import summary
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2, InterpolationMode
from torchvision import datasets
from torch.utils.data import DataLoader

print(torch.__version__)

# COMMAND ----------

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# COMMAND ----------

width = 224
height = 224

data_transform = v2.Compose([
    v2.PILToTensor(),
    v2.Resize(size=(232, 232), interpolation=InterpolationMode.BILINEAR, antialias=True),
    v2.CenterCrop(size=(224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# COMMAND ----------

data_path = Path("/Volumes/catalog1/schema1/data1/food-101")
image_path = data_path / "images"
meta_path = data_path / "meta"
train_path = data_path / "train"
test_path = data_path / "test"
model_path = data_path / 'model_data'

# COMMAND ----------

image_path_list = list(image_path.glob("*/*.jpg"))

# COMMAND ----------

train_data = datasets.ImageFolder(root=train_path,              # target folder of images 
                                  transform=data_transform,     # transforms to perform on data (images)
                                  target_transform=None)        # transforms to perform on labels (if necessary)

test_data = datasets.ImageFolder(root=test_path, 
                                 transform=data_transform, 
                                 target_transform=None)

class_names = train_data.classes
class_dict = train_data.class_to_idx

BATCH_SIZE = 512
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                            #   num_workers=os.cpu_count(),
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                            #  num_workers=os.cpu_count(),
                             shuffle=False)

# COMMAND ----------

img, label = next(iter(train_dataloader))
print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")

# COMMAND ----------

# DBTITLE 1,Obtain ResNet152 model architecture
from torchvision.models import ResNet152_Weights, resnet152

weights = ResNet152_Weights.IMAGENET1K_V2

# Instantiate model and initialize with ImageNet weights
model = resnet152(weights=weights).to(device)

# for name, child in model.named_children():
#     print(name)

# for name, parameters in model.named_parameters():
#     print(name)

# COMMAND ----------

# DBTITLE 1,Freeze all layers of the model
for name, child in model.named_children():
    for param in child.parameters():
        param.requires_grad = False

# COMMAND ----------

# Apply it to an input image
preprocess = weights.transforms()
img_transformed = preprocess(img)
weights.meta["categories"]
summary(model, input_size=[BATCH_SIZE, 3, width, height])

# COMMAND ----------

# Create results dictionary
results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

lr = 0.0001

# COMMAND ----------

# DBTITLE 1,Function Definitions
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):

    model.train() # Put the model in train mode

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader and data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)
        # print(y_pred)

        # 2. Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad 
        optimizer.zero_grad()

        # 4. Loss backward 
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and average accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):

    model.eval() # Put model in eval mode

    # Setup the test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X).to(device)
            # print(test_pred_logits)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def plot_loss_curves(results: Dict[str, List[float]]):
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

# from tqdm.auto import tqdm
from tqdm.notebook import tqdm

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          results: Dict,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          lr_scheduler_gamma: float = 0.1,
          epochs: int = 5):
    
    scheduler = ExponentialLR(optimizer, gamma=lr_scheduler_gamma)

    # Loop through the training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer)
        test_loss, test_acc = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn)

        print(f"Epoch: {epoch + 1} | "
              f"train_loss: {train_loss:.4f} | "
              f"train_acc: {train_acc:.4f} | "
              f"test_loss: {test_loss:.4f} | "
              f"test_acc: {test_acc:.4f} | "
              f"lr: {optimizer.param_groups[0]['lr']}"
              )

        # Update the results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        scheduler.step() # Update the learning rate scheduler

    # Return the results dictionary
    return results

# COMMAND ----------

# Unfreeze Layer 3 and 4
for name, child in model.named_children():
    if name in ['layer4', 'layer3']:
        for param in child.parameters():
            param.requires_grad = True
        print(name + ' is unfrozen')
    else: # Freeze layers
        for param in child.parameters():
            param.requires_grad = False
            
optimizer = torch.optim.Adam(params = filter(lambda p: p.requires_grad, model.parameters()), 
                             lr=lr)

# optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()), lr=0.006, momentum=0.9)
model_results = train(model=model, 
                      train_dataloader=train_dataloader, 
                      test_dataloader=test_dataloader, 
                      results=results,
                      optimizer=optimizer, epochs=5)

plot_loss_curves(model_results)

# COMMAND ----------

# Unfreeze Layer 2 and 1, refreeze 3 and 4
for name, child in model.named_children():
    if name in ['layer2', 'layer1']:
        for param in child.parameters():
            param.requires_grad = True
        print(name + ' is unfrozen')
    else: # Freeze Layers
        for param in child.parameters():
            param.requires_grad = False
            
# optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()), lr=0.006, momentum=0.9)
optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), 
                             lr= lr/10)

model_results = train(model=model, 
                      train_dataloader=train_dataloader, 
                      test_dataloader=test_dataloader,
                      results=results,
                      optimizer=optimizer, epochs=5)

plot_loss_curves(model_results)

# COMMAND ----------

torch.save(model.state_dict(), 'models/ft_resnet152.pth')

# COMMAND ----------


