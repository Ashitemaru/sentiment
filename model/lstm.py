import torch

embed_dim = 768
lstm_hidden_size = 768
class_num = 7
lstm_layers = 6
is_bidirectional = True

# This param controls how many layers will be picked
filter_layer_num = 6

class LSTMModel(torch.nn.Module):
	def __init__(self):
		super(LSTMModel, self).__init__()
		self.core = torch.nn.LSTM(
			input_size = embed_dim,
			hidden_size = lstm_hidden_size,
			num_layers = lstm_layers,
			bidirectional = is_bidirectional,
		)
		# Cancel the bias of linear layer to ensure treating different classes fairly
		self.classifier = torch.nn.Linear(
			in_features = lstm_hidden_size * filter_layer_num * (2 if is_bidirectional else 1),
			out_features = class_num,
			bias = False
		)

	def forward(self, x):
		_, (x, __) = self.core(x)
		x = torch.cat([
			x[ind] for ind in range(
				(lstm_layers - filter_layer_num) * (2 if is_bidirectional else 1),
				lstm_layers * (2 if is_bidirectional else 1)
			)
		], dim = 1)
		x = self.classifier(x)
		return torch.nn.functional.softmax(x, dim = 1)

def main():
	pass

if __name__ == '__main__':
	main()