import torch
import torch.nn as nn
import torch.nn.functional as F

cube_size = 5

class encoder_block(nn.Module):
	def __init__(self, input_feature, output_feature, use_dropout):
		super(encoder_block, self).__init__()
		self.conv_input = nn.Conv3d(input_feature, output_feature, 3, 1, 1, 1).cuda()
		self.conv_inblock1 = nn.Conv3d(output_feature, output_feature, 3, 1, 1, 1).cuda()
		self.conv_inblock2 = nn.Conv3d(output_feature, output_feature, 3, 1, 1, 1).cuda()
		self.conv_pooling = nn.Conv3d(output_feature, output_feature, 2, 2, 1, 1).cuda()
		self.prelu1 = nn.PReLU().cuda()
		self.prelu2 = nn.PReLU().cuda()
		self.prelu3 = nn.PReLU().cuda()
		self.prelu4 = nn.PReLU().cuda()
		self.use_dropout = use_dropout;
		self.dropout = nn.Dropout(0.2).cuda()
	def apply_dropout(self, input):
		if self.use_dropout:
			return self.dropout(input)
		else:
			return input;
	def forward(self, x):
		output = self.conv_input(x)
		output = self.apply_dropout(self.prelu1(output));
		output = self.apply_dropout(self.prelu2(self.conv_inblock1(output)));
		output = self.apply_dropout(self.prelu3(self.conv_inblock2(output)));
		return self.prelu4(self.conv_pooling(output));


class decoder_block(nn.Module):
	def __init__(self, input_feature, output_feature, pooling_filter, use_dropout):
		super(decoder_block, self).__init__()
		self.conv_unpooling = nn.ConvTranspose3d(input_feature, input_feature, pooling_filter, 2, 1).cuda()
		self.conv_inblock1 = nn.Conv3d(input_feature, input_feature, 3, 1, 1, 1).cuda()
		self.conv_inblock2 = nn.Conv3d(input_feature, input_feature, 3, 1, 1, 1).cuda()
		self.conv_output = nn.Conv3d(input_feature, output_feature, 3, 1, 1, 1).cuda()
		self.prelu1 = nn.PReLU().cuda()
		self.prelu2 = nn.PReLU().cuda()
		self.prelu3 = nn.PReLU().cuda()
		self.prelu4 = nn.PReLU().cuda()
		self.use_dropout = use_dropout;
		self.dropout = nn.Dropout(0.2).cuda()
		self.output_feature = output_feature;
	def apply_dropout(self, input):
		if self.use_dropout:
			return self.dropout(input);
		else:
			return input;
	def forward(self, x):
		output = self.prelu1(self.conv_unpooling(x));
		output = self.apply_dropout(self.prelu2(self.conv_inblock1(output)));	
		output = self.apply_dropout(self.prelu3(self.conv_inblock2(output)));
		#if self.output_feature == 1: # generates final momentum
		return self.conv_output(output);
		#else: # generates intermediate results
	#		return self.apply_dropout(self.prelu4(self.conv_output(output)));

class MPNN(nn.Module):
	def __init__(self, input_dim, output_dim, edge_dim, message_dim):
		super(MPNN, self).__init__()
		self.edge_linear = nn.Linear(input_dim, edge_dim).cuda()
		self.edge_relu = nn.ReLU().cuda()
		self.message_linear = nn.Linear(input_dim, message_dim).cuda()
		self.message_relu = nn.ReLU().cuda()
		self.gru = nn.GRUCell(message_dim, output_dim).cuda()


	def forward(self, x_list):
		input_dim = x_list[0].size(1)
		edge_list = [[] for i in range(len(x_list))]
		print len(edge_list)
		for i in range(len(x_list)):
			for j in range(len(x_list)):
				if j<=i:
					edge_list[i].append(0)
				else :
					edge_list[i].append(self.edge_relu(self.edge_linear(x_list[i]+x_list[j])))

		message_list = []
		for i in range(len(x_list)):
			m_flag=0
			for j in range(len(x_list)):
				if i != j:
					if m_flag==0:
						message = self.message_relu(self.message_linear(x_list[j]+edge_list[min(i,j)][max(i,j)]))
						m_flag=1
					else:
						message = message + self.message_relu(self.message_linear(x_list[j]+edge_list[min(i,j)][max(i,j)]))
			message_list.append(message)

		out_list = []
		for i in range(len(x_list)):
			out = self.gru(message_list[i], x_list[i]).view(-1, input_dim/(cube_size**3), cube_size,cube_size,cube_size)
			out_list.append(out)
		return out_list


class net(nn.Module):
	def __init__(self, feature_num, use_dropout = False):
		super(net, self).__init__()
		self.encoder_1 = encoder_block(1, feature_num, use_dropout);
		self.encoder_2 = encoder_block(feature_num, feature_num*2, use_dropout)

		m_dim = 2*feature_num*cube_size*cube_size*cube_size
		self.mpnn = MPNN(m_dim, m_dim, m_dim, m_dim)

		#self.decoder_x_1 = decoder_block(feature_num * 2, feature_num * 2, 2, use_dropout);
		#self.decoder_y_1 = decoder_block(feature_num * 2, feature_num * 2, 2, use_dropout);
		#self.decoder_z_1 = decoder_block(feature_num * 2, feature_num * 2, 2, use_dropout);

		#self.decoder_x_2 = decoder_block(feature_num*2, 1, 3, use_dropout);
		#self.decoder_y_2 = decoder_block(feature_num*2, 1, 3, use_dropout);
		#self.decoder_z_2 = decoder_block(feature_num*2, 1, 3, use_dropout);

		#self.decoder_x_inv_1 = decoder_block(feature_num * 2, feature_num * 2, 2, use_dropout);
		#self.decoder_y_inv_1 = decoder_block(feature_num * 2, feature_num * 2, 2, use_dropout);
		#self.decoder_z_inv_1 = decoder_block(feature_num * 2, feature_num * 2, 2, use_dropout);

		#self.decoder_x_inv_2 = decoder_block(feature_num*2, 1, 3, use_dropout);
		#self.decoder_y_inv_2 = decoder_block(feature_num*2, 1, 3, use_dropout);
		#self.decoder_z_inv_2 = decoder_block(feature_num*2, 1, 3, use_dropout);

		self.decoder_1 = decoder_block(feature_num * 2, feature_num * 2, 2, use_dropout);
		self.decoder_2 = decoder_block(feature_num * 2, 3, 3, use_dropout);

		self.decoder_inv_1 = decoder_block(feature_num * 2, feature_num * 2, 2, use_dropout);
		self.decoder_inv_2 = decoder_block(feature_num * 2, 3, 3, use_dropout);

	def forward(self, img_list):
		encoder_output_list = []
		batch_size = img_list[0].size(0)
		for i in range(len(img_list)):
			encoder_output = self.encoder_2(self.encoder_1(img_list[i])).view(batch_size, -1)
			encoder_output_list.append(encoder_output)
		print encoder_output_list[0].size()
		mpnn_output_list = self.mpnn(encoder_output_list)

		predict_result_x_list = []
		predict_result_y_list = []
		predict_result_z_list = []
		predict_result_x_inv_list = []
		predict_result_y_inv_list = []
		predict_result_z_inv_list = []
		for i in range(len(img_list)):
			predict_result = self.decoder_2(self.decoder_1(mpnn_output_list[i]))
			predict_result_x_list.append(predict_result[:,0,:,:,:].view(-1, 1, 15,15,15))
			predict_result_y_list.append(predict_result[:,1,:,:,:].view(-1, 1, 15,15,15))
			predict_result_z_list.append(predict_result[:,2,:,:,:].view(-1, 1, 15,15,15))

			predict_result_inv = self.decoder_inv_2(self.decoder_inv_1(mpnn_output_list[i]))
			predict_result_x_inv_list.append(predict_result_inv[:,0,:,:,:].view(-1, 1, 15,15,15))
			predict_result_y_inv_list.append(predict_result_inv[:,1,:,:,:].view(-1, 1, 15,15,15))
			predict_result_z_inv_list.append(predict_result_inv[:,2,:,:,:].view(-1, 1, 15,15,15))

			#predict_result_x_list.append(self.decoder_x_2(self.decoder_x_1(mpnn_output_list[i])))
			#predict_result_y_list.append(self.decoder_y_2(self.decoder_y_1(mpnn_output_list[i])))
			#predict_result_z_list.append(self.decoder_z_2(self.decoder_z_1(mpnn_output_list[i])))

			#predict_result_x_inv_list.append(self.decoder_x_inv_2(self.decoder_x_inv_1(mpnn_output_list[i])))
			#predict_result_y_inv_list.append(self.decoder_y_inv_2(self.decoder_y_inv_1(mpnn_output_list[i])))
			#predict_result_z_inv_list.append(self.decoder_z_inv_2(self.decoder_z_inv_1(mpnn_output_list[i])))

		#print(predict_result)
		return predict_result_x_list, predict_result_y_list, predict_result_z_list, \
		       predict_result_x_inv_list, predict_result_y_inv_list, predict_result_z_inv_list

