'''
This package takes care of recognizing
politician faces , using transfer Learning on
resnet model.

v2 - OpenFace weights
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np 
import torchvision 
from torchvision import datasets , models , transforms
import matplotlib.pyplot as plt
import os
import time 
import copy

# Data transforms

data_transforms = {
	'train' : transforms.Compose([
		transforms.RandomSizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		# transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229,0.224,0.225]) # Given Weights -> Move to OpenFace weights after initial run
		]),
	'val' : transforms.Compose([
		transforms.Scale(256),
		transforms.RandomSizedCrop(224),
		transforms.ToTensor(),
		# transforms.Normalize([0.485 , 0.456 , 0.406] , [0.229,0.224,0.225])
		])
}

data_dir = '../data'

dsets = {x : datasets.ImageFolder(os.path.join(data_dir , x) , data_transforms[x])
		for x in ['train' , 'val']}

dset_loaders = { x: torch.utils.data.DataLoader(dsets[x] , batch_size = 4 ,
												shuffle=True , num_workers=4)
				for x in ['train' , 'val']}
dset_sizes = {x: len(dsets[x]) for x in ['train' , 'val']}
dset_classes = dsets['train'].classes

def show_samples(inp , title = None):
	inp = inp.numpy().transpose((1,2,0))
	mean = np.array([0.485,0.456,0.406])
	sd = np.array([0.229,0.224,0.225])
	inp = inp*mean + sd 
	plt.imshow(inp)
	plt.pause(5)

inputs , classes = next(iter(dset_loaders['train']))

out = torchvision.utils.make_grid(inputs)

# show_samples(out , title = [dset_classes[x] for x in classes])

def train_model(model , criterion , optimizer , lr_scheduler , num_epochs =25):
	t = time.time()

	best_model = model
	best_acc = 0.0

	for epoch in range(num_epochs):
		print('Epoch {} / {}'.format(epoch , num_epochs -1))
		print('-'*10)

		for phase in ['tain' , 'val']:
			if phase == 'train':
				