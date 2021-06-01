import torch

embed_dim = 768
class_num = 7
lstm_layers = 6

class LSTMModel(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.core = torch.nn.LSTM(
			input_size = embed_dim,
			hidden_size = embed_dim,
			num_layers = lstm_layers,
		)
		self.classifier = torch.nn.Linear(embed_dim * lstm_layers, class_num)

	def forward(self, x):
		_, (x, __) = self.core(x)
		x = torch.cat([x[ind] for ind in range(lstm_layers)], dim = 1)
		x = self.classifier(x)
		return torch.nn.functional.softmax(x, dim = 1)

def main():
	pass

if __name__ == '__main__':
	main()