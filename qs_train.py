# add LDDMM shooting code into path
import sys
sys.path.append('../vectormomentum/Code/Python')
sys.path.append('../library')
from subprocess import call
import argparse
import os.path
import gc

#Add deep learning related libraries
from collections import Counter
import torch
from torch.utils.serialization import load_lua
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import prediction_network
import util
from own_function import *
import numpy as np

#Add LDDMM registration related libraries
# library for importing LDDMM formulation configs
# others
import logging
import copy
import math

#parse command line input
parser = argparse.ArgumentParser(description='Deformation predicting given set of moving and target images.')

##required parameters
requiredNamed = parser.add_argument_group('required named arguments')
requiredNamed.add_argument('--moving-image-dataset', nargs='+', required=True, metavar=('m1', 'm2, m3...'),
						   help='List of moving images datasets stored in .pth.tar format (or .t7 format for the old experiments in the Neuroimage paper). File names are seperated by space.')
requiredNamed.add_argument('--target-image-dataset', nargs='+', required=True, metavar=('t1', 't2, t3...'),
						   help='List of target images datasets stored in .pth.tar format (or .t7 format for the old experiments in the Neuroimage paper). File names are seperated by space.')
requiredNamed.add_argument('--deformation-parameter', nargs='+', required=True, metavar=('o1', 'o2, o3...'),
						   help='List of target deformation parameter files to predict to, stored in .pth.tar format (or .t7 format for the old experiments in the Neuroimage paper). File names are seperated by space.')
requiredNamed.add_argument('--output-name', required=True, metavar=('file_name'),
						   help='output directory + name of the network parameters, in .pth.tar format.')
##optional parameters
parser.add_argument('--features', type=int, default=32, metavar='N',
					help='number of output features for the first layer of the deep network (default: 64)')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
					help='input batch size for prediction network (default: 32)')
parser.add_argument('--patch-size', type=int, default=15, metavar='N',
					help='patch size to extract patches (default: 15)')
parser.add_argument('--stride', type=int, default=14, metavar='N',
					help='sliding window stride to extract patches for training (default: 14)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
					help='number of epochs to train the network (default: 10)')
parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='N',
					help='learning rate for the adam optimization algorithm (default: 0.0001)')
parser.add_argument('--use-dropout', action='store_true', default=False,
					help='Use dropout to train the probablistic version of the network')
parser.add_argument('--n-GPU', type=int, default=1, metavar='N',
					help='number of GPUs used for training.')
parser.add_argument('--continue-from-parameter', metavar=('parameter_name'),
						   help='file directory+name of the existing parameter if want to start')
args = parser.parse_args()

# finish command line input


# check validity of input arguments from command line
def check_args(args):
	# number of input images/output prefix consistency check
	n_moving_images = len(args.moving_image_dataset)
	n_target_images = len(args.target_image_dataset)
	n_deformation_parameter = len(args.deformation_parameter)
	if (n_moving_images != n_target_images):
		print('The number of moving image datasets is not consistent with the number of target image datasets!')
		sys.exit(1)
	elif (n_moving_images != n_deformation_parameter ):
		print('The number of moving image datasets is not consistent with the number of deformation parameter datasets!')
		sys.exit(1)

	# number of GPU check (positive integers)
	if (args.n_GPU <= 0):
		print('Number of GPUs must be positive!')
		sys.exit(1)
#enddef



def create_net(args):
	net_single = prediction_network.net(args.features, args.use_dropout).cuda();
	
	if (args.continue_from_parameter != None):
		print('Loading existing parameter file!')
		config = torch.load(args.continue_from_parameter)
		net_single.load_state_dict(config['state_dict'])

	if (args.n_GPU > 1) :
		device_ids=range(0, args.n_GPU)
		net = torch.nn.DataParallel(net_single, device_ids=device_ids).cuda()
	else:
		net = net_single

	net.train()
	return net;
#enddef

# define functions used in flow warping
def make_one_hot(labels, C=args.patch_size*args.patch_size*args.patch_size):
	one_hot = torch.cuda.FloatTensor(labels.size(0), C).zero_()
	target = one_hot.scatter_(1, labels.data, 1)
	target = Variable(target)
	return target

class Round(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x):
		return x.round_()
	@staticmethod
	def backward(ctx, g):
		return g

class Clamp(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x, a, b):
		ctx.save_for_backward(x)
		ctx.constant = a, b
		return torch.clamp(x, a, b)
	@staticmethod
	def backward(ctx, g_out):
		x, = ctx.saved_tensors
		a, b = ctx.constant
		g_in = g_out.clone()
		g_in[x<a] = 0
		g_in[x>b] = 0
		return g_in, None, None

class Flow_mul(torch.autograd.Function):
	@staticmethod
	def forward(ctx, f1, f2):
		ctx.save_for_backward(f1, f2)
		flow_01_est1 = torch.ones(args.batch_size, 1, args.patch_size, args.patch_size, args.patch_size).cuda()
		for i in range(f1.size(0)):
			one_hot_0A1 = make_one_hot(f1[i].long())
			one_hot_A11 = make_one_hot(f2[i].long())
			one_hot_011 = torch.matmul(one_hot_A11, one_hot_0A1)
			index_011 = torch.nonzero(one_hot_011)
			index_011 = index_011[:,1]
			flow_01_est1[i,0] = index_011.view(args.patch_size, args.patch_size, args.patch_size)
		#flow_01_est1 = Variable(flow_01_est1, requires_grad=True)
		return flow_01_est1
	@staticmethod
	def backward(ctx, g_out):
		f1, f2 = ctx.saved_tensors
		g_in = g_out.clone().view(-1, args.patch_size*args.patch_size*args.patch_size, 1)
		print "hello world!"
		return g_in, g_in


def train_cur_data(cur_epoch, datapart, moving_file, target_file, parameter, output_name, net, criterion, optimizer,  args):
	old_experiments = False
	if moving_file[-3:] == '.t7' :
		old_experiments = True
		#only for old data used in the Neuroimage paper. Do not use .t7 format for new data and new experiments.
		moving_appear_trainset = load_lua(moving_file).float()
		target_appear_trainset = load_lua(target_file).float()
		train_m0 = 10*torch.ones(100, 3, moving_appear_trainset.size()[1], moving_appear_trainset.size()[1],moving_appear_trainset.size()[1] )#load_lua(parameter).float()
	else :
		moving_appear_trainset = torch.load(moving_file).float()
		target_appear_trainset = torch.load(target_file).float()
		train_m0 = 10*torch.ones(100, 3, moving_appear_trainset.size()[1], moving_appear_trainset.size()[1], moving_appear_trainset.size()[1])  #torch.load(parameter).float()

	input_batch_0 = torch.zeros(args.batch_size, 1, args.patch_size, args.patch_size, args.patch_size).cuda()
	input_batch_1 = torch.zeros(args.batch_size, 1, args.patch_size, args.patch_size, args.patch_size).cuda()
	output_batch = torch.zeros(args.batch_size, 3, args.patch_size, args.patch_size, args.patch_size).cuda()

	dataset_size = moving_appear_trainset.size()
	flat_idx = util.calculatePatchIdx3D(dataset_size[0], args.patch_size*torch.ones(3), dataset_size[1:], args.stride*torch.ones(3));
	flat_idx_select = torch.zeros(flat_idx.size());
	
	for patch_idx in range(1, flat_idx.size()[0]):
		patch_pos = util.idx2pos_4D(flat_idx[patch_idx], dataset_size[1:])
		moving_patch = moving_appear_trainset[patch_pos[0], patch_pos[1]:patch_pos[1]+args.patch_size, patch_pos[2]:patch_pos[2]+args.patch_size, patch_pos[3]:patch_pos[3]+args.patch_size]
		target_patch = target_appear_trainset[patch_pos[0], patch_pos[1]:patch_pos[1]+args.patch_size, patch_pos[2]:patch_pos[2]+args.patch_size, patch_pos[3]:patch_pos[3]+args.patch_size]
		if (torch.sum(moving_patch) + torch.sum(target_patch) != 0):
			flat_idx_select[patch_idx] = 1; 
	
	flat_idx_select = flat_idx_select.byte();

	flat_idx = torch.masked_select(flat_idx, flat_idx_select);
	N = flat_idx.size()[0] / args.batch_size;	


	for iters in range(0, N):
		train_idx = (torch.rand(args.batch_size).double() * flat_idx.size()[0])
		train_idx = torch.floor(train_idx).long()
		for slices in range(0, args.batch_size):
			patch_pos = util.idx2pos_4D(flat_idx[train_idx[slices]], dataset_size[1:])
			input_batch_0[slices,0] =  moving_appear_trainset[patch_pos[0], patch_pos[1]:patch_pos[1]+args.patch_size, patch_pos[2]:patch_pos[2]+args.patch_size, patch_pos[3]:patch_pos[3]+args.patch_size].cuda()
			input_batch_1[slices,0] =  target_appear_trainset[patch_pos[0], patch_pos[1]:patch_pos[1]+args.patch_size, patch_pos[2]:patch_pos[2]+args.patch_size, patch_pos[3]:patch_pos[3]+args.patch_size].cuda()
			output_batch[slices] =	train_m0[ 1, :, patch_pos[1]:patch_pos[1]+args.patch_size, patch_pos[2]:patch_pos[2]+args.patch_size, patch_pos[3]:patch_pos[3]+args.patch_size].cuda()

		input_batch_0_variable = Variable(input_batch_0).cuda()
		input_batch_1_variable = Variable(input_batch_1).cuda()
		output_batch_variable = Variable(output_batch).cuda()

		optimizer.zero_grad()
		x_list, y_list, z_list, x_inv_list, y_inv_list, z_inv_list = net([input_batch_0_variable, input_batch_1_variable])

		#construct flow among images
		flow_0A = Round.apply(args.patch_size*args.patch_size*args.patch_size * x_list[0].view(-1, args.patch_size*args.patch_size*args.patch_size, 1))
		flow_A1 = Round.apply(args.patch_size*args.patch_size*args.patch_size * x_inv_list[1].view(-1, args.patch_size*args.patch_size*args.patch_size, 1))
		flow_0A = Clamp.apply(flow_0A, 0, args.patch_size*args.patch_size*args.patch_size-1)
		flow_A1 = Clamp.apply(flow_A1, 0, args.patch_size * args.patch_size * args.patch_size - 1)

		flow_01_est = Flow_mul.apply(flow_0A, flow_A1)

		x2 = Clamp.apply(x_list[0], -1000, args.patch_size*args.patch_size*args.patch_size-1)
		loss = criterion(flow_01_est[:,0,:,:,:], output_batch_variable[:, 0, :,:,:])
		loss.backward(retain_graph=True)
		loss_value = loss.data[0]
		optimizer.step()
		print('====> Epoch: {}, datapart: {}, iter: {}/{}, loss: {:.4f}'.format(
			cur_epoch+1, datapart+1, iters, N, loss_value/args.batch_size))
		if iters % 100 == 0 or iters == N-1:
			if args.n_GPU > 1:
				cur_state_dict = net.module.state_dict()
			else:
				cur_state_dict = net.state_dict()			
			
			modal_name = output_name
			
			model_info = {
				'patch_size' : args.patch_size,
				'network_feature' : args.features,
				'state_dict': cur_state_dict,
			}
			if old_experiments :
				model_info['matlab_t7'] = True
			#endif
			torch.save(model_info, modal_name)	
#enddef 

def train_network(args):
	net = create_net(args)
	net.train()
	criterion = nn.L1Loss().cuda()
	optimizer = optim.Adam(net.parameters(), args.learning_rate)
	for cur_epoch in range(0, args.epochs) :
		for datapart in range(0, len(args.moving_image_dataset)) :
			train_cur_data(
				cur_epoch, 
				datapart,
				args.moving_image_dataset[datapart], 
				args.target_image_dataset[datapart], 
				args.deformation_parameter[datapart], 
				args.output_name,
				net, 
				criterion, 
				optimizer,
				args
			)
			gc.collect()
#enddef

if __name__ == '__main__':
	check_args(args);
	#registration_spec = read_spec(args)
	train_network(args)