import torch
import PyCA.Core as ca
import PyCA.Common as common

from subprocess import call
import argparse
import os.path
import numpy as np
import nibabel as nib
parser = argparse.ArgumentParser(description='Gather 3D images / 4D momentums to .pth.tar files. Used to create files for training.')

requiredNamed = parser.add_argument_group('required named arguments')

requiredNamed.add_argument('--files', nargs='+', required=True, metavar=('f1', 'f2, f3...'),
                           help='List of 3D image / 4D momentum file directories, seperated by space.')

requiredNamed.add_argument('--output', required=True, metavar=('output.pth.tar'),
                           help='resulting .pth.tar file containing the 3D image / 4D momentums')
parser.add_argument('--momentum', action='store_true', default=False,
                    help='Use 4D momentums instead of 3D images as the input files')

args = parser.parse_args()


def gather_file(args):
	if args.momentum:
		file = common.AsNPCopy(common.LoadITKField(args.files[0], ca.MEM_HOST))
	else:
		#file = common.AsNPCopy(common.LoadITKImage(args.files[0], ca.MEM_HOST))
		file = nib.load(args.files[0]).get_data()
	
	all_size = (len(args.files),)+ (15,15,15,1,3) #file.shape;

	data = torch.zeros(all_size)
	for i in range(0, len(args.files)):
		if args.momentum:
			cur_slice = torch.from_numpy(common.AsNPCopy(common.LoadITKField(args.files[i], ca.MEM_HOST)))
		else:
			cur_slice = nib.load(args.files[i]).get_data()
			if cur_slice.size == 3375:
				cur_slice=np.zeros([15,15,15,1,3])
			cur_slice = torch.from_numpy(cur_slice)
		data[i] = cur_slice

	if args.momentum:
		# transpose the dataset to fit the training format
		data = data.numpy()
		data = np.transpose(data, [0, 4, 1, 2, 3])
		data = torch.from_numpy(data);
		
	torch.save(data, args.output)

if __name__ == '__main__':
	gather_file(args)
