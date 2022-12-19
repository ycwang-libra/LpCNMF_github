# -*- coding: utf-8 -*-
"""
selfsupervised training for DNMF to generate U
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from data_loader import get_loader
from solver import Solver
import argparse
from dependencies.utils import Config_Adjust

def main(config):
	# Data loader.
	train_loader = None
	train_loader = get_loader(config)
	solver = Solver(train_loader, config)
	if config.mode == 'pretrain':
		solver.pretrain()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name', type=str, default='LpCNMF_dl')
	parser.add_argument('--mode', type=str, default='pretrain', choices=['pretrain'])
	parser.add_argument('--dataset', type=str, default='MNIST', \
		choices=['MNIST','AR','COIL20','COIL100','Yale','YaleB','UMIST','USPS'])
	parser.add_argument('--label_ratio', type=float, default=0.3, \
		help='Proportion of label data in total标')
	# model parameter
	parser.add_argument('--num_layer', type=int, default=10, \
		help='num of model layers')	
	parser.add_argument('--lr', type=float, default=0.001, \
		help='learning rate')
	parser.add_argument('--beta1', type=float, default=0.5, \
		help='beta1 for Adam optimizer')
	parser.add_argument('--beta2', type=float, default=0.999, \
		help='beta2 for Adam optimizer')
	# train parameter
	parser.add_argument('--nCluster', type=list, default=[2,3,4,5,6,7,8,9,10], \
		help='Number of class in experiment')
	parser.add_argument('--num_epoch', type=int, default=100, \
		help='num of epoch')
	parser.add_argument('--nEach', type=int, default=500, \
		help='Number of samples in each class')
	parser.add_argument('--percent', type=float, default=0.3)
	# other parameter
	parser.add_argument('--WeightMode', type=str, default='HeatKernel')
	parser.add_argument('--NeighborMode', type=str, default='KNN')
	parser.add_argument('--k', type=int, default=5)
	parser.add_argument('--t', type=int, default=1)
	parser.add_argument('--maxIter', type=int, default=200)
	parser.add_argument('--alpha', type=int, default=10)
	# path
	parser.add_argument('--data_root_path', type=str, \
		 default='/Datasets_path/',\
		 help = 'dataset path')
	parser.add_argument('--save_root_path', type=str, \
		 default='LpCNMF_dl/model_UV_results/',\
		 help = 'model and UV saving path') 
	parser.add_argument('--log_root_path', type=str, \
		 default='LpCNMF_dl/Logging/',\
		 help = 'log path')
	config = parser.parse_args()
	config = Config_Adjust(config)
	print(config)
	main(config)

