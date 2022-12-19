# -*- coding: utf-8 -*-
"""
DNMF model
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch.nn as nn

class DNMF(nn.Module):
	def __init__(self, config, k):
		super(DNMF, self).__init__()
		self.num_feature = config.mFea
		self.num_sample = config.nSmp
		self.img_h = config.img_size[0]
		self.img_w = config.img_size[1]
		self.F_net = F_net(config.nSmp, k = k)
		self.G_net = G_net(config.nSmp, num_res = 2, k = k, end_dim = config.end_dim)
	def forward(self, X): # nfeature * nsample
		U = self.F_net(X)
		img_X = X.reshape(self.num_sample,1,self.img_w, self.img_h) 
		V = self.G_net(img_X)
		return U, V

class F_net(nn.Module):
    def __init__(self, num_sample, k):
        super(F_net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_sample, 1024),
            nn.ReLU(inplace=True),
			nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
			nn.Linear(256, 64),
            nn.ReLU(inplace=True),    
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, k),
            nn.ReLU(inplace=True))
    def forward(self, X): # nsample * nfeature 
        V = self.layers(X)
        return V

class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out, eps = 1e-5, momentum = 0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out, eps = 1e-5, momentum = 0.1))
    def forward(self, x):
        return x + self.main(x)

class G_net(nn.Module):
	def __init__(self, num_sample, num_res, k, end_dim):
		super(G_net, self).__init__()
		self.num_sample = num_sample
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 7, stride = 1, padding=3),
            nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1),  # 28 * 28 --> 14 * 14
            nn.BatchNorm2d(16, eps = 1e-5, momentum = 0.1),
            nn.ReLU(inplace=True),
			nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 1, padding=1),
            nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1),  # 28 * 28 --> 14 * 14
            nn.BatchNorm2d(32, eps = 1e-5, momentum = 0.1),
            nn.ReLU(inplace=True))
		layer2 = []
		for i in range(num_res):
			layer2.append(ResidualBlock(32, 32))
		self.layer2 = nn.Sequential(*layer2)
		self.layer3 = nn.Sequential(
			nn.Conv2d(in_channels = 32, out_channels = 1, kernel_size = 3, stride = 1, padding=1),
			nn.AvgPool2d(kernel_size = 3, stride = 2, padding = 1))
		self.outlayer = nn.Linear(end_dim, k)
	def forward(self, X): # nsample * nfeature
		output1 = self.layer1(X)
		output2 = self.layer2(output1)
		output3 = self.layer3(output2)
		output3 = output3.view(output3.size(0), -1)
		output4 = self.outlayer(output3)
		return output4