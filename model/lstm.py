import torch
from config import config

embed_dim = config()['embed_dim']
class_num = config()['class_num']

class LSTMModel(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.core = torch.nn.LSTM(
			input_size = embed_dim,
			hidden_size = embed_dim,
			num_layer = 6,
		)
		self.classifier = torch.nn.Linear(embed_dim, class_num)
		self.norm = torch.nn.Softmax()

	def forward(self, x):
		x = self.core(x)
		x = self.classifier(x)
		return torch.nn.functional.softmax(x)

def main():
	pass

if __name__ == '__main__':
	main()