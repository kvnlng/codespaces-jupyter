# Databricks notebook source
# DBTITLE 1,Grab a few random images to infer
import requests

custom_image = "pizza_dad.jpeg"
with open("pizza_dad.jpeg", "wb") as f:
    request = requests.get(
        "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
    f.write(request.content)

# COMMAND ----------

import torchvision

img = torchvision.io.read_image(custom_image)

plt.imshow(img.permute(1, 2, 0)) # permutate to HWC (height, width, color_channels) format 
plt.axis(False);

# COMMAND ----------

from torchvision.models import resnet152

model = resnet152().to(device)
model.load_state_dict(torch.load('models/ft_resnet152.pth'))

# COMMAND ----------

# Make a prediction on the image
model.eval()

with torch.inference_mode():
    # Get image pixels into float + between 0 and 1
    img = img / 255.

    # Resize image to 64x64
    resize = v2.Resize((width, height))
    img = resize(img)

    # Turn image in single batch and pass to target device
    batch = img.unsqueeze(0)

    # Predict on image
    y_pred_logit = model(batch.to(device))

    # Convert pred logit to pred label
    # pred_label = torch.argmax(torch.softmax(y_pred_logit, dim=1), dim=1)
    pred_label = torch.argmax(y_pred_logit, dim=1)  # get same results as above without torch.softmax

# Plot the image and prediction
plt.imshow(img.permute(1, 2, 0))
plt.title(f"Pred label: {class_names[pred_label]}")
plt.axis(False);
