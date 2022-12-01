# %% [markdown]
# ## 1. 사용할 패키지 불러오기

# %%
from utils import ConstructDataset, ConstructInferenceDataset, prepare_dataset
from models.visualize import plot_loss, plot_score, plot_acc, visualize_inference
from models import ModelFactory, LossFactory, fit 
from models.inference import get_samples, predict_image_mask_miou, miou_score, pixel_acc
import albumentations as A
from torch.utils.data import DataLoader
from torchvision import transforms as T
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import os
import torch
import torch.nn as nn

# %% [markdown]
# ## 2. 데이터셋 불러오기

# %% [markdown]
# ### (1) Dataset Directory Setting

# %%
train_image_dir = './dataset/train/JPEGImages'
test_image_dir = './dataset/test/JPEGImages'

train_mask_dir = './dataset/train/SegmentationClass'
test_mask_dir = './dataset/test/SegmentationClass'

# %% [markdown]
# ### (2) Prepare Dataset

# %%
train_dataset = prepare_dataset(image_dir = train_image_dir, mask_dir = train_mask_dir)
test_dataset = prepare_dataset(image_dir = test_image_dir, mask_dir = test_mask_dir, label_exist = True)

# %% [markdown]
# ### (3) Train Dataset으로 부터 Validation Dataset 생성

# %%
valid_split = 0.2

# %%
train_image_list, valid_image_list, train_mask_list, valid_mask_list = train_test_split(train_dataset['Image'], train_dataset['Mask'], test_size=valid_split, random_state=1004)
test_image_list = test_dataset['Image']
test_mask_list = test_dataset['Mask']

# %% [markdown]
# ## 3. Parameter Setting

# %% [markdown]
# ### (1) Transformation Setting

# %%
input_size = (384, 512)  # devided by 32
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

# %%
train_transform = A.Compose([A.Resize(input_size[0], input_size[1], interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(), A.VerticalFlip(), 
                     A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0,0.5),(0,0.5)),
                     A.GaussNoise()])

validation_transform = A.Compose([A.Resize(input_size[0], input_size[1], interpolation=cv2.INTER_NEAREST)])

# %% [markdown]
# ### (2) Make Torch DataLoader

# %%
batch_size = 16

# %%
train_set = ConstructDataset(train_image_list, train_mask_list, mean, std, train_transform)
val_set = ConstructDataset(valid_image_list, valid_mask_list, mean, std, validation_transform)

# %%
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)     

# %% [markdown]
# ## 4. Model

# %% [markdown]
# ### (1) Parameter Setting

# %%
architecture = 'DeepLabV3+' ## ['unet', 'fpn', 'DeepLabV3+']
encoder_name = 'mobilenet_v2'
in_channels = 3
class_num = 22
experiment_path = './result'
os.makedirs(experiment_path, exist_ok=True)
os.makedirs(os.path.join(experiment_path, architecture + '+' + encoder_name), exist_ok=True)
save_model_path = os.path.join(experiment_path, architecture + '+' + encoder_name)


# %% [markdown]
# ### (2) Construct Model

# %%
model = ModelFactory(architecture = architecture, encoder_name = encoder_name, in_channels = in_channels, class_num = class_num)

# %% [markdown]
# ## 5. Training

# %% [markdown]
# ### (1) Loss, Optimizer 정의

# %%
max_lr = 1e-3
epoch = 15
weight_decay = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = LossFactory(loss_name = 'crossentropy').get_loss_fn()
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epoch,
                                            steps_per_epoch=len(train_dataloader))

# %% [markdown]
# ### (2) Fit

# %%
history = fit(epoch, model, train_dataloader, valid_dataloader, criterion, optimizer, sched, save_model_path, device)

# %% [markdown]
# ### (3) Plot Training Process

# %%
plot_loss(history)
plot_score(history)
plot_acc(history)

# %% [markdown]
# ### (4) Load Best model

# %%
best_model = torch.load(os.path.join(save_model_path, 'best_model.pt'))

# %% [markdown]
# ## 6. Inference

# %% [markdown]
# ### (1) Construct Inference Dataset

# %%
test_set = ConstructInferenceDataset(valid_image_list, valid_mask_list, validation_transform)

# %% [markdown]
# ### (2) Sample Testset Visualize

# %%
sample_testset = get_samples(test_set, 5)

# %%
for (image, mask) in sample_testset:
    pred_mask, score = predict_image_mask_miou(best_model, image, mask, device)
    visualize_inference(image, mask, pred_mask, architecture = architecture, encoder_name = encoder_name, score = score)

# %%



