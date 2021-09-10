from data import data
from utils import annotate
from models import model as m
from training import training
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
print(torch.__version__)

# load data
annoations_filepath = "data/mpii_human_pose_v1_u12_1.mat"
images_base_filepath_local = "/Users/danny/Code/Data/SHPE/images/"
images_base_filepath_server = "/home/paperspace/Code/Data/SHPE/images/"

train_loader, val_loader = data.load_data(227, 128, images_base_filepath_server, annoations_filepath, True)

# create model and optimizer
model = m.get_model_resnet(34, device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# train
training.train_single_lr(model, train_loader, val_loader, 0.001, 75, device, 'visibility_with_sigmoid_34_norm')

# evaluate
loss, _, o, l= training.evaluate(model, val_loader, nn.MSELoss(), device)

# load saved model and test it
checkpoint = torch.load('saved_models/cyclical.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Train loss from checkpoint: {checkpoint['train_loss']}")
print(f"Val loss from checkpoint: {checkpoint['val_loss']}")

loss, _ = training.evaluate(model, val_loader, nn.MSELoss(), device)
print(f'Val loss after loading: {loss}')
 
itr = iter(val_loader)
x, y = itr.next()
for i in range(x.shape[0]):
    img = np.asarray(transforms.ToPILImage()(x[i].squeeze_(0)))
    coordinates = training.predict(model, x[i].unsqueeze(0).to(device))
    annotate.annotate_flat_image(coordinates.squeeze(), image=img)






