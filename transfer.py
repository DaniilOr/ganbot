import torch
import torch.nn as nn
import torch.optim as optim

import copy
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
import itertools

import copy


def loss_fn(target, resource):
  return torch.mean((resource - target).pow(2))


IMG_SIZE = 128
CONTENT_LAYER = 'conv1'
STYLE_LAYERS = ['layer1',
                 'layer2',
                 'layer3']
CONTENT_COEF = 10
STYLE_COEF = 1e3
device = ("cuda" if torch.cuda.is_available() else "cpu")

def img_to_tensor(img):
  #img = Image.open(path).convert('RGB')
  transformation = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

  img = transformation(img)[:3, :, :].unsqueeze(0)

  return img

def tensor_to_img(tensor):
  image = tensor.to("cpu").clone().detach()
  image = image.numpy().squeeze()
  image = image.transpose(1, 2, 0)
  image = image * np.array((0.229, 0.224, 0.225)) + np.array(
    (0.485, 0.456, 0.406))
  image = image.clip(0, 1)

  return image

def get_features_resnet(img, model):
  x = img
  layers = ['conv1', 'layer1', 'layer2', 'layer3']
  features = {}
  children = list(model.named_children())
  for name, child in children:
    x = child(x)
    if name in layers:
        features[name] = x
    if name == layers[-1]:
        break

  return features

def set_trainability(model, trainable=False):
  for param in model.parameters():
    param.requires_grad = trainable


def gram(tensor):
  _, n_filters, h, w = tensor.size()
  tensor = tensor.view(n_filters, h * w)
  return torch.mm(tensor, tensor.t())


def style_transfer(content, style1,  n_epoches=101):
  content = img_to_tensor(content).to(device)
  style1 = img_to_tensor(style1).to(device)
  model = models.resnet50(pretrained=True)
  torch.save(model.state_dict(), 'resnet.pt')

  model.load_state_dict(torch.load('resnet.pt'))
  model.eval()
  set_trainability(model)

  model = model.to(device)
  content_features = get_features_resnet(content, model)
  style1_features = get_features_resnet(style1, model)
  style1_weights = {'layer1': 0.8,
                 'layer2': 0.7,
                 'layer3': 0.5}
  target = content.clone().requires_grad_(True).to(device)
  opt = optim.AdamW([target], lr=0.01)
  for epoch in range(n_epoches):
    if epoch % 10 == 0:
        print(epoch)
    style_loss = 0
    opt.zero_grad()
    target_features = get_features_resnet(target, model)
    #print(target_features)
    content_loss = loss_fn(target_features[CONTENT_LAYER], content_features[CONTENT_LAYER])
    for layer in STYLE_LAYERS:
      tg = gram(target_features[layer])
      source1 = style1_features[layer]
      gram1 = gram(source1)
      current_loss = style1_weights[layer] * loss_fn(tg, gram1) * STYLE_COEF

      style_loss += current_loss
    style_loss /= target.shape[1] * target.shape[2] * target.shape[3]

    loss = content_loss * CONTENT_COEF + style_loss * STYLE_COEF
    #first_ratio = first_ratio.requires_grad_(True)
    loss.backward(retain_graph=True)

    opt.step()
  img_to_show = tensor_to_img(target)
      #img_to_show = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)

      # imshow() only accepts float [0,1] or int [0,255]
  img_to_show = np.array(img_to_show).clip(0,1)
  return (img_to_show * 255).astype('uint8')
