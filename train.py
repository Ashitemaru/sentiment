from model.bert_embedding import BERTEmbedding
from model.lstm import LSTMModel

file_path_prefix = '/home/luoyuqi/holder/sentiment/data/processed_isear_v2/'
file_path_suffix = ['isear_test.txt', 'isear_train.txt', 'isear_valid.txt']

def batch_gen(batch_size):
	pass

def main():
	# Create the model
	model = LSTMModel()

	# Load the data
	test_data = open(file_path_prefix + file_path_suffix[0], 'r').readlines()
	train_data = open(file_path_prefix + file_path_suffix[1], 'r').readlines()
	valid_data = open(file_path_prefix + file_path_suffix[2], 'r').readlines()

	x = BERTEmbedding(['I love you!', 'Fuck you!'])
	print(model(x))

if __name__ == '__main__':
	main()