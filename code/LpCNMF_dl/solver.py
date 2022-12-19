# -*- coding: utf-8 -*-

import numpy as np
import torch
from time import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from data_loader import prepare_data_label, sample_label, shuffle_datalabel, gen_shuffleClasses, gen_shuffleIndexes
import datetime
import os
from model import DNMF

class Solver():
	def __init__(self, train_loader, config):
		self.train_loader = train_loader
		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	def build_DNMF(self, k):
		"""Create DNMF."""
		self.DNMF = DNMF(self.config, k = k)
		self.optimizer = torch.optim.Adam(self.DNMF.parameters(),\
						self.config.lr, [self.config.beta1, self.config.beta2])
		self.print_network(self.DNMF, 'DNMF')
		self.DNMF.to(self.device)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(name)
		print("The number of parameters: {}".format(num_params))
		
	def reset_grad(self):
		"""Reset the gradient buffers."""
		self.optimizer.zero_grad()

	def Save_rand(self, UV_root_path, shuffleClasses, shuffleIndexes, orgsemiSplit, k):
		# save rand in DNMF for reconstruction the datasets
		shuffleClasses_path = UV_root_path + 'shuffleClasses_{}.npy'.format(k)
		shuffleIndexes_path = UV_root_path + 'shuffleIndexes_{}.npy'.format(k)
		orgsemiSplit_path = UV_root_path + 'orgsemiSplit_{}.npy'.format(k)
		np.save(shuffleClasses_path, shuffleClasses)
		np.save(shuffleIndexes_path, shuffleIndexes)
		np.save(orgsemiSplit_path, orgsemiSplit)

	def save_print_log(self, txt_path, *args):
		# save and print the log information
		f = open(txt_path,'a')
		for log_content in args:
			print(log_content)
			f.write(log_content)
		f.close()

	def pretrain(self): # train DNMF
		'''pretrain use DNMF and save the model, rand vector, U and V'''
		# path setting
		UV_root_path = self.config.save_root_path + '/' + self.config.mode + '/UV_rand/'+ self.config.dataset  + '/'
		model_root_path = self.config.save_root_path + '/' + self.config.mode + '/DNMF_model/' + self.config.dataset + '/'
		log_path = self.config.log_root_path + self.config.mode + '/'
		if not os.path.exists(UV_root_path):
			os.makedirs(UV_root_path)
		if not os.path.exists(model_root_path):
			os.makedirs(model_root_path)	
		if not os.path.exists(log_path):
			os.makedirs(log_path)
		pretrain_txt_path = os.path.join(log_path, 'Log_'+self.config.dataset+'_output.txt')		
		
		# origin data loading
		fea, gnd = self.train_loader

		nCluster = self.config.nCluster
		nCase = len(nCluster)
		num_epoch  = self.config.num_epoch

		caseIter = 0
		start_time = time()
		for k in nCluster:
			caseIter = caseIter + 1
			log1 = '################' + '\n'
			log2 = 'The k is : ' + str(k) + ' The dataset is: ' + self.config.dataset + '\n'
			self.save_print_log(pretrain_txt_path, log1, log2)
			
			# prepare the datasets and save the random for reconstruct data
			# once random generation for k
			shuffleClasses = gen_shuffleClasses(gnd, self.config)
			shuffleIndexes = gen_shuffleIndexes(k, self.config)
			# dataset prepare
			orgfeaSet, orggndSet = prepare_data_label(fea, gnd, self.config, k, shuffleClasses)
			orgsemiSplit = sample_label(orggndSet, self.config.percent)
			feaSet, gndSet, semiSplit = shuffle_datalabel(orgfeaSet, orggndSet, orgsemiSplit, shuffleIndexes)
			# save random for construct data
			self.Save_rand(UV_root_path, shuffleClasses, shuffleIndexes, orgsemiSplit, k)

			# build DNMF
			self.config.nSmp = k * self.config.nEach
			self.build_DNMF(k)

			for epoch in range(num_epoch):
				self.DNMF.train()
				log = 'The epoch is : ' + str(epoch) + '\n'
				self.save_print_log(pretrain_txt_path, log)
				
				X = np.transpose(feaSet)
				X = torch.Tensor(X).to(self.device)
				U,V = self.DNMF.train()(X)

				X_hat = torch.mm(U.detach(), (V.detach()).permute(1,0))
				rec_loss = torch.mean((X_hat - X)**2)
				
				rec_loss.requires_grad_(True)
				self.reset_grad()
				rec_loss.backward()
				self.optimizer.step()

				used_time = str(datetime.timedelta(seconds = time() - start_time))[:-7]
				now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
				time_log = 'Used time is [{}]'.format(used_time) + ' And now time is: {}'.format(now_time) + '\n'
				self.save_print_log(pretrain_txt_path, time_log)
			
			# save the UV
			self.DNMF.eval()
			U,V = self.DNMF.eval()(X)
			U_path = os.path.join(UV_root_path, 'U_{}.npy'.format(k))
			V_path = os.path.join(UV_root_path, 'V_{}.npy'.format(k))
			np_U = np.array(U.detach().cpu())
			np_V = np.array(V.detach().cpu())
			np.save(U_path, np_U)
			np.save(V_path, np_V)
			save_log = 'UV saved to {}.'.format(UV_root_path) + '\n'
			self.save_print_log(pretrain_txt_path, save_log)

			# save the pretrained model
			model_path = os.path.join(model_root_path, 'DNMF_{}.pth'.format(k))
			torch.save({'model':self.DNMF.state_dict()},model_path)
			save_log = 'Model saved to {}.'.format(model_path) + '\n'
			self.save_print_log(pretrain_txt_path, save_log)