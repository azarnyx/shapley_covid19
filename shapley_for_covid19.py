import shap
import torch
import torchvision
from torch.autograd import Variable
from PIL import Image
import numpy as np
import glob
import pylab as plt

# change line 198 after shap installation   in python3.7/site-packages/shap/explainers/_deep/
#From:
# phis[l][j] = (torch.from_numpy(sample_phis[l][self.data[l].shape[0]:]).to(self.device) * (X[l][j: j + 1] - self.data[l])).cpu().numpy().mean(0)
# To:
# phis[l][j] = (torch.from_numpy(sample_phis[l][self.data[l].shape[0]:]).to(self.device) * (X[l][j: j + 1] - self.data[l])).cpu().detach().numpy().mean(0)
# TODO: fill out pull request


model = torch.load('./model.pth')
model.eval()

loader = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size = (224,224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean =[0.485,0.456,0.406],
                                     std=[0.229,0.224,0.225]),
])


def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name).convert('RGB')
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image #pass the path of the image to be tested


list_images = glob.glob("data/COVID-19 Radiography Database/NORMAL/*png")[0:30]+\
              glob.glob("data/COVID-19 Radiography Database/Viral Pneumonia/*png")[0:30]+\
              glob.glob("data/COVID-19 Radiography Database/COVID-19/*png")[0:30]


list_tensors = []
for i in list_images:
    im = image_loader(i)
    list_tensors.append(im)

images = torch.cat(list_tensors)

e = shap.DeepExplainer(model, images)


image = image_loader('data/sample_images/COVID-19 (36).png')
shap_values = e.shap_values(image)


def preprocess(im):
    im = im.swapaxes(0,2)
    im = im.swapaxes(0,1)
    return im

im = image.detach().numpy()
im = im.swapaxes(1, 3)
a = 255*(im-im.min())/(-im.min()+im.max())
a = a.swapaxes(1,2)
shap_values1 = [i.swapaxes(1, 3) for i in shap_values]
shap_values1 = [i.swapaxes(1, 2) for i in shap_values1]
shap.image_plot(shap_values1, a)
