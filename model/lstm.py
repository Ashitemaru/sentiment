import torch

class LSTMModel(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.core = None

	def forward(self, x):
		return self.core(x)

def main():
	pass

if __name__ == '__main__':
	main()