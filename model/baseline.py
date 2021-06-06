import torch

embed_size = 768
class_num = 7

class NaiveBaseline(torch.nn.Module):
	def __init__(self):
		super(NaiveBaseline, self).__init__()
		self.classifier = torch.nn.Linear(
			in_features = embed_size,
			out_features = class_num,
			bias = False,
		)

	def forward(self, x):
		x = torch.sum(x, dim = 0)
		return torch.nn.functional.softmax(
			self.classifier(x),
			dim = 1
		)

def main():
	pass

if __name__ == '__main__':
	main()