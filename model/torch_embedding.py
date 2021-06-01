import torch

# Config
vocab_path = '/home/qianhoude/sentiment/data/vocab.txt'
embed_size = 128
device = 'cuda:0'

def load_vocab():
	vocab_handler = open(vocab_path, 'r')
	raw_vocab = vocab_handler.readlines()
	vocab = {}

	# Load the vocab
	for line in raw_vocab:
		index = int(line.split(' # ')[0])
		word = line.split(' # ')[1].strip()
		vocab[word] = index

	return vocab

class TorchEmbedding(torch.nn.Module):
	def __init__(self):
		super().__init__()
		# Get the vocab
		self.vocab = load_vocab()

		# Init the torch embedding
		self.core_embedding = torch.nn.Embedding(
			num_embeddings = len(self.vocab),
			embedding_dim = embed_size
		)

	def forward(self, text_list):
		# Convert words into ids
		input_ids = [
			list(map(
				lambda word: self.vocab[word if word in self.vocab else '[UNK]'],
				sentence.split(' ')
			)) for sentence in text_list
		]

		# Padding these input ids
		seq_len = max(list(map(len, input_ids)))
		input_ids = [
			g + [0] * (seq_len - len(g)) for g in input_ids
		]

		return self.core_embedding(torch.LongTensor(input_ids).to(device)).permute(1, 0, 2)

def main():
	pass

if __name__ == '__main__':
	main()