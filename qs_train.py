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
import timeit
import numpy as np
import nibabel as nib

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
requiredNamed.add_argument('--image-dataset', nargs='+', required=True, metavar=('m0', 'm2, m3...'),
						   help='List of moving images datasets stored in .pth.tar format (or .t7 format for the old experiments in the Neuroimage paper). File names are seperated by space.')
requiredNamed.add_argument('--truth-flow', nargs='+', required=True, metavar=('o1', 'o2, o3...'),
						   help='List of target deformation parameter files to predict to, stored in .pth.tar format (or .t7 format for the old experiments in the Neuroimage paper). File names are seperated by space.')
requiredNamed.add_argument('--output-dir', required=True, metavar=('file_name'),
						   help='output directory + name of the network parameters, in .pth.tar format.')
##optional parameters
parser.add_argument('--features', type=int, default=16, metavar='N',
					help='number of output features for the first layer of the deep network (default: 64)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
					help='input batch size for prediction network (default: 32)')
parser.add_argument('--patch-size', type=int, default=15, metavar='N',
					help='patch size to extract patches (default: 15)')
parser.add_argument('--stride', type=int, default=14, metavar='N',
					help='sliding window stride to extract patches for training (default: 14)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
					help='number of epochs to train the network (default: 10)')
parser.add_argument('--group-size', type=int, default=5, metavar='N',
					help='size of the group (k input images at a time) (default: 10)')
parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='N',
					help='learning rate for the adam optimization algorithm (default: 0.0001)')
parser.add_argument('--use-dropout', action='store_true', default=False,
					help='Use dropout to train the probablistic version of the network')
parser.add_argument('--n-GPU', type=int, default=1, metavar='N',
					help='number of GPUs used for training.')
parser.add_argument('--restore', type=str, default=None, metavar='N',
					help='restore path')
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
	
	if (args.restore != None):
		print('Restoring from checkpoint!')
		net_single.load_state_dict(torch.load(args.restore))
		print('Restore success!')

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
		return flow_01_est1

	@staticmethod
	def backward(ctx, g_out):
		t1 = timeit.default_timer()
		f1, f2 = ctx.saved_tensors


		g_out = g_out.clone().view(-1, args.patch_size*args.patch_size*args.patch_size, 1).cpu().data.numpy()
		g_in1 = np.zeros([args.batch_size, args.patch_size * args.patch_size * args.patch_size, 1])
		g_in2 = np.zeros([args.batch_size, args.patch_size * args.patch_size * args.patch_size, 1])


		f1_cpu = f1.clone().cpu().data.numpy()
		f2_cpu = f2.clone().cpu().data.numpy()
		shape0 = g_out.shape[0]
		shape1 = g_out.shape[1]

		for i in range(shape0):
			for j in range(shape1):
				a = int(f1_cpu[i,j,0])
				b = int(f2_cpu[i, max(a-1,0), 0])
				c = int(f2_cpu[i,min(a+1,shape1-1), 0])
				# g_in1
				if b<=c:
					g_in1[i,j,0] = g_out[i,j,0]
				else:
					g_in1[i,j,0] = -g_out[i,j,0]
				# g_in2
				g_in2[i, a, 0] = g_out[i, j, 0]


		g_in1 = torch.from_numpy(g_in1).float().cuda()
		g_in2 = torch.from_numpy(g_in2).float().cuda()



		return g_in1, g_in2


def train_cur_data(cur_epoch, datapart, train_dir, labels_dir, output_dir, net, criterion, optimizer,  args):
	old_experiments = False

	#only for old data used in the Neuroimage paper. Do not use .t7 format for new data and new experiments.
	appear_trainset_list = []
	for i in range(args.group_size):
		appear_trainset_list.append(torch.load(train_dir+'%02d.pth.tar'%i).float())

	labels_list = [[] for i in range(args.group_size)]
	for i in range(args.group_size):
		for j in range(args.group_size):
			if i == j:
				labels_list[i].append(0)
			else:
				labels_list[i].append(torch.load(labels_dir+'%02d_%02d.pth.tar'%(i, j)))

	input_batch_list = []
	for i in range(args.group_size):
		input_batch_list.append(torch.zeros(args.batch_size, 1, args.patch_size, args.patch_size, args.patch_size).cuda())

	label_batch_list = [[]for i in range(args.group_size)]
	for i in range(args.group_size):
		for j in range(args.group_size):
			label_batch_list[i].append(torch.zeros(args.batch_size, 3, args.patch_size, args.patch_size, args.patch_size).cuda())


	dataset_size = appear_trainset_list[0].size()

	flat_idx = util.calculatePatchIdx3D(dataset_size[0], args.patch_size*torch.ones(3), dataset_size[1:], args.stride*torch.ones(3));
	flat_idx_select = torch.zeros(flat_idx.size());

	for patch_idx in range(1, flat_idx.size()[0]):
		patch_pos = util.idx2pos_4D(flat_idx[patch_idx], dataset_size[1:])
		patch_list=[]
		for i in range(args.group_size):
			patch_list.append(appear_trainset_list[i][patch_pos[0], patch_pos[1]:patch_pos[1]+args.patch_size, patch_pos[2]:patch_pos[2]+args.patch_size, patch_pos[3]:patch_pos[3]+args.patch_size])
		if (torch.sum(torch.cat(patch_list, 0))!= 0):
			flat_idx_select[patch_idx] = 1; 
	
	flat_idx_select = flat_idx_select.byte();

	flat_idx = torch.masked_select(flat_idx, flat_idx_select);
	N = flat_idx.size()[0] / args.batch_size;

	t = timeit.default_timer()
	for iters in range(0, N):
		#train_idx = (torch.rand(args.batch_size).double() * N * args.batch_size)
		train_idx = torch.arange(iters*args.batch_size, (iters+1)*args.batch_size)
		train_idx = torch.floor(train_idx).long()
		for slices in range(0, args.batch_size):
			patch_pos = util.idx2pos_4D(flat_idx[train_idx[slices]], dataset_size[1:])
			for i in range(args.group_size):
				input_batch_list[i][slices,0] =  appear_trainset_list[i][patch_pos[0], patch_pos[1]:patch_pos[1]+args.patch_size, patch_pos[2]:patch_pos[2]+args.patch_size, patch_pos[3]:patch_pos[3]+args.patch_size].cuda()
				for j in range(args.group_size):
					if i != j:
						label_batch_list[i][j][slices] = labels_list[i][j][train_idx[slices]]

				#nii_img = nib.Nifti1Image(input_batch_list[i][slices,0].cpu().data.numpy(), np.eye(4))
				#nii_img.to_filename('./tmp/%d/%02d_%02d.nii' %(i,iters, slices))
		#continue

		#input_batch_0_variable = Variable(input_batch_0).cuda()
		#input_batch_1_variable = Variable(input_batch_1).cuda()
		#label_batch_variable = Variable(label_batch).cuda()

		optimizer.zero_grad()
		x_list, y_list, z_list, x_inv_list, y_inv_list, z_inv_list = net(input_batch_list)

		#construct flow among images
		for i in range(args.group_size):
			for j in range(args.group_size):
				if i != j:
					flow_0A_x = Round.apply(args.patch_size*args.patch_size*args.patch_size * x_list[i].view(-1, args.patch_size*args.patch_size*args.patch_size, 1))
					flow_A1_x = Round.apply(args.patch_size*args.patch_size*args.patch_size * x_inv_list[j].view(-1, args.patch_size*args.patch_size*args.patch_size, 1))
					flow_0A_x = Clamp.apply(flow_0A_x, 0, args.patch_size*args.patch_size*args.patch_size-1)
					flow_A1_x = Clamp.apply(flow_A1_x, 0, args.patch_size * args.patch_size * args.patch_size - 1)
					flow_01_est_x = Flow_mul.apply(flow_0A_x, flow_A1_x)

					flow_0A_y = Round.apply(args.patch_size*args.patch_size*args.patch_size * y_list[i].view(-1, args.patch_size*args.patch_size*args.patch_size, 1))
					flow_A1_y = Round.apply(args.patch_size*args.patch_size*args.patch_size * y_inv_list[j].view(-1, args.patch_size*args.patch_size*args.patch_size, 1))
					flow_0A_y = Clamp.apply(flow_0A_y, 0, args.patch_size*args.patch_size*args.patch_size-1)
					flow_A1_y = Clamp.apply(flow_A1_y, 0, args.patch_size * args.patch_size * args.patch_size - 1)
					flow_01_est_y = Flow_mul.apply(flow_0A_y, flow_A1_y)

					flow_0A_z = Round.apply(args.patch_size*args.patch_size*args.patch_size * z_list[i].view(-1, args.patch_size*args.patch_size*args.patch_size, 1))
					flow_A1_z = Round.apply(args.patch_size*args.patch_size*args.patch_size * z_inv_list[j].view(-1, args.patch_size*args.patch_size*args.patch_size, 1))
					flow_0A_z = Clamp.apply(flow_0A_z, 0, args.patch_size*args.patch_size*args.patch_size-1)
					flow_A1_z = Clamp.apply(flow_A1_z, 0, args.patch_size * args.patch_size * args.patch_size - 1)
					flow_01_est_z = Flow_mul.apply(flow_0A_z, flow_A1_z)

					flow_01_est = torch.cat([flow_01_est_x, flow_01_est_y, flow_01_est_z], dim=1)
					if i==0 and j==1:
						loss = criterion(flow_01_est,label_batch_list[i][i])
					else:
						loss = loss + criterion(flow_01_est, label_batch_list[i][i])
		loss.backward(retain_graph=True)
		loss_value = loss.data.item()
		optimizer.step()
		print timeit.default_timer() - t
		t = timeit.default_timer()
		print('====> Epoch: {}, datapart: {}, iter: {}/{}, loss: {:.4f}'.format(
			cur_epoch+1, datapart+1, iters, N, loss_value/args.batch_size))

		if iters  == N/2 or iters == N-1:
			if args.n_GPU > 1:
				cur_state_dict = net.module.state_dict()
			else:
				cur_state_dict = net.state_dict()
			torch.save(cur_state_dict, output_dir+"epoch_%d_iters_%d.pt"%(cur_epoch+1, iters))
			print "Save model success!"
#enddef 

def train_network(args):
	net = create_net(args)
	net.train()
	criterion = nn.L1Loss().cuda()
	optimizer = optim.Adam(net.parameters(), args.learning_rate)
	for cur_epoch in range(0, args.epochs) :
		for datapart in range(0, len(args.image_dataset)):
			train_cur_data(
				cur_epoch, 
				datapart,
				args.image_dataset[datapart],
				args.truth_flow[datapart],
				args.output_dir,
				net, 
				criterion, 
				optimizer,
				args
			)
			gc.collect()

		#break
#enddef

if __name__ == '__main__':
	#check_args(args);
	#registration_spec = read_spec(args)
	train_network(args)
