import torch

# Must ensure the two lists share the same length
cnn_channel_num_list = [3, 4, 5]
cnn_kernel_size_list = [768, 768, 768]

embed_size = 768
class_num = 7
dropout = 0.5

class ModifiedMaxPool1d(torch.nn.Module):
	def __init__(self):
		super(ModifiedMaxPool1d, self).__init__()
	
	def forward(self, x):
		return torch.nn.functional.max_pool1d(
			x,
			kernel_size = x.size()[2]
		)

class BaseCNN(torch.nn.Module):
	def __init__(self):
		super(BaseCNN, self).__init__()
		self.cnn_core_list = [
			torch.nn.Conv1d(
				in_channels = embed_size,
				out_channels = channel_num,
				kernel_size = kernel_size,
			) for channel_num, kernel_size in zip(
				cnn_channel_num_list, cnn_kernel_size_list
			)
		]
		self.pool = ModifiedMaxPool1d()
		self.dropout = torch.nn.Dropout(dropout)
		self.classifier = torch.nn.Linear(
			in_features = sum(cnn_channel_num_list),
			out_features = class_num,
			bias = False,
		)
	
	def forward(self, x):
		# Concat activated features
		x = torch.concat([
			self.pool(torch.nn.functional.relu(
				cnn_layer(x)
			)).squeeze(-1) for cnn_layer in self.cnn_core_list
		], dim = 1)

		# Dropout, classify and normalization
		x = self.dropout(x)
		x = self.classifier(x)
		return torch.nn.functional.softmax(x, dim = 1)

def main():
	pass

if __name__ == '__main__':
	main()